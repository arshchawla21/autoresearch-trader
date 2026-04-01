"""
v11-ml-calibrated: ML-driven direction prediction with calibrated tight stops.

Step 1: Analyze historical bar ranges to set SL/TP that actually trigger.
Step 2: Engineer features (multi-timeframe returns, volume, spread, cross-asset).
Step 3: Train a gradient-boosted tree (via simple NN) to predict next-bar direction.
Step 4: Trade all stocks every bar with tight symmetric SL/TP sized to ~50-70%
        of the typical 15m high-low range, so most trades resolve at SL or TP.

Goal: >50% win rate on symmetric stops = compounding edge.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import O, H, L, C, V, evaluate

t_start = time.time()


class DirectionModel(nn.Module):
    """Predicts probability that the next bar closes up from open."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def compute_features(ohlcv_np, tidx, bar_idx, n_stocks):
    """
    Compute per-stock features at bar_idx from history.
    Returns (n_stocks, n_features) array.
    """
    if bar_idx < 20:
        return None

    closes = ohlcv_np[:bar_idx, tidx, C]  # (T, N)
    opens_hist = ohlcv_np[:bar_idx, tidx, O]
    highs = ohlcv_np[:bar_idx, tidx, H]
    lows = ohlcv_np[:bar_idx, tidx, L]
    vols = ohlcv_np[:bar_idx, tidx, V]

    curr_open = ohlcv_np[bar_idx, tidx, O]  # current bar open

    features = []

    # 1. Returns at multiple lookbacks (1, 2, 3, 5, 10 bars)
    for lb in [1, 2, 3, 5, 10]:
        if bar_idx >= lb + 1:
            ret = (closes[-1] - closes[-1 - lb]) / np.maximum(np.abs(closes[-1 - lb]), 1e-8)
        else:
            ret = np.zeros(n_stocks)
        features.append(ret)

    # 2. Open-to-close of last 3 bars (intrabar momentum)
    for i in [1, 2, 3]:
        if bar_idx >= i:
            oc = (closes[-i] - opens_hist[-i]) / np.maximum(np.abs(opens_hist[-i]), 1e-8)
        else:
            oc = np.zeros(n_stocks)
        features.append(oc)

    # 3. High-low range of last 3 bars (volatility)
    for i in [1, 2, 3]:
        if bar_idx >= i:
            hl = (highs[-i] - lows[-i]) / np.maximum(closes[-i], 1e-8)
        else:
            hl = np.zeros(n_stocks)
        features.append(hl)

    # 4. Close position within last bar's range (0=at low, 1=at high)
    if bar_idx >= 1:
        bar_range = highs[-1] - lows[-1]
        close_pos = np.where(bar_range > 1e-8,
                             (closes[-1] - lows[-1]) / bar_range,
                             0.5)
    else:
        close_pos = np.full(n_stocks, 0.5)
    features.append(close_pos)

    # 5. Gap: current open vs previous close
    gap = (curr_open - closes[-1]) / np.maximum(np.abs(closes[-1]), 1e-8)
    features.append(gap)

    # 6. Volume ratio: last bar vs 10-bar avg
    if bar_idx >= 10:
        avg_vol = np.mean(vols[-10:], axis=0)
        vol_ratio = vols[-1] / np.maximum(avg_vol, 1e-8)
    else:
        vol_ratio = np.ones(n_stocks)
    features.append(np.clip(vol_ratio, 0, 10))

    # 7. Rolling volatility (5-bar std of returns)
    if bar_idx >= 6:
        rets_5 = np.diff(closes[-6:], axis=0) / np.maximum(np.abs(closes[-6:-1]), 1e-8)
        roll_vol = np.std(rets_5, axis=0)
    else:
        roll_vol = np.zeros(n_stocks)
    features.append(roll_vol)

    # 8. Cross-sectional rank of last return (how this stock compares to peers)
    ret1 = features[0]  # 1-bar return
    ranks = np.argsort(np.argsort(ret1)).astype(np.float32) / max(n_stocks - 1, 1)
    features.append(ranks)

    # 9. Consecutive up/down bars (streak)
    if bar_idx >= 3:
        streak = np.sign(closes[-1] - opens_hist[-1]) + np.sign(closes[-2] - opens_hist[-2]) + np.sign(closes[-3] - opens_hist[-3])
    else:
        streak = np.zeros(n_stocks)
    features.append(streak / 3.0)

    # Stack: (n_features, n_stocks) -> (n_stocks, n_features)
    return np.stack(features, axis=1).astype(np.float32)  # (N, F)


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    # Compute per-stock ATR for stop calibration
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    closes = ohlcv[:, tidx, C]
    bar_range_pct = (highs - lows) / np.maximum(closes, 1e-8)
    # Median bar range per stock (more robust than mean)
    median_range = np.nanmedian(bar_range_pct, axis=0)

    print(f"  Median 15m bar range: {np.mean(median_range)*100:.3f}% across stocks")

    # Build training data
    X_list = []
    y_list = []

    for t in range(20, T - 1):
        feats = compute_features(ohlcv, tidx, t, n_stocks)
        if feats is None:
            continue

        # Target: did the stock go up from open to close in the NEXT bar?
        next_open = ohlcv[t, tidx, O]
        next_close = ohlcv[t, tidx, C]
        up = (next_close > next_open).astype(np.float32)

        valid = ~np.isnan(feats).any(axis=1) & (next_open > 0) & ~np.isnan(next_close)
        if valid.sum() > 0:
            X_list.append(feats[valid])
            y_list.append(up[valid])

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"  Training samples: {len(X)}, features: {X.shape[1]}")
    print(f"  Base rate (up): {y.mean():.3f}")

    # Normalize features
    feat_mean = np.nanmean(X, axis=0)
    feat_std = np.nanstd(X, axis=0) + 1e-8
    X_norm = (X - feat_mean) / feat_std
    X_norm = np.nan_to_num(X_norm, 0.0)

    # Train model
    n_features = X.shape[1]
    model = DirectionModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    batch_size = 256
    n = len(X_t)

    best_loss = float('inf')
    patience = 0
    for epoch in range(200):
        perm = torch.randperm(n)
        epoch_loss = 0
        batches = 0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            pred = model(X_t[batch_idx])
            loss = criterion(pred, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience > 20:
                break

    model.eval()

    # Check training accuracy
    with torch.no_grad():
        train_pred = model(X_t).numpy().flatten()
        train_acc = ((train_pred > 0.5) == y).mean()
    print(f"  Train accuracy: {train_acc:.3f}, loss: {best_loss:.4f}, epochs: {epoch+1}")

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "median_range": median_range,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_stocks": n_stocks,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 20:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    median_range = strategy["median_range"]
    model = strategy["model"]
    n_stocks = strategy["n_stocks"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv

    feats = compute_features(ohlcv_np, tidx.numpy(), bar_idx, n_stocks)
    if feats is None:
        return []

    opens = ohlcv_np[bar_idx, tidx.numpy(), O]
    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(feats).any(axis=1))[0]
    if len(valid) == 0:
        return []

    # Get model predictions
    feat_norm = (feats[valid] - strategy["feat_mean"]) / strategy["feat_std"]
    feat_norm = np.nan_to_num(feat_norm, 0.0)

    with torch.no_grad():
        probs = model(torch.from_numpy(feat_norm)).numpy().flatten()

    # Only trade stocks where the model is confident (>55% or <45%)
    orders = []
    total_weight = 0.0
    candidates = []

    for i, idx in enumerate(valid):
        p = float(probs[i])
        confidence = abs(p - 0.5)
        if confidence < 0.05:  # skip uncertain predictions
            continue
        direction = "long" if p > 0.5 else "short"
        candidates.append((idx, direction, confidence))

    if not candidates:
        return []

    # Sort by confidence, take top stocks
    candidates.sort(key=lambda x: -x[2])
    top = candidates[:min(10, len(candidates))]
    w = 1.0 / len(top)

    for idx, direction, conf in top:
        ticker = tickers[idx]
        op = float(opens[idx])
        mr = float(median_range[idx])

        # SL and TP set to ~50-60% of typical bar range
        # This means they should trigger within the bar roughly half the time
        stop_dist = max(mr * 0.5, 0.001) * op

        if direction == "long":
            sl = op - stop_dist
            tp = op + stop_dist
        else:
            sl = op + stop_dist
            tp = op - stop_dist

        orders.append({
            "ticker": ticker,
            "direction": direction,
            "weight": w,
            "stop_loss": sl,
            "take_profit": tp,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
