"""
v13-ml-minimal: Tiny model, few features, heavy regularization.

Key insight from v11 vs v12: v11 (simple) worked, v12 (complex) overfit.
With only ~4000 training samples, we need extreme simplicity.

Approach: logistic regression with a single hidden layer (8 units),
only 7 features, high weight decay, 50% dropout. Target: open-to-close
direction (simpler target than TP/SL hit).

Retain cross-asset features (SPY, VIX) since they're not stock-specific
and add real information.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import O, H, L, C, V, evaluate

t_start = time.time()


class TinyModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def compute_features(ohlcv_np, tidx, midx, bar_idx, n_stocks):
    """Minimal feature set — only the most robust signals."""
    if bar_idx < 10:
        return None

    closes = ohlcv_np[:bar_idx, tidx, C]
    opens_h = ohlcv_np[:bar_idx, tidx, O]
    highs = ohlcv_np[:bar_idx, tidx, H]
    lows = ohlcv_np[:bar_idx, tidx, L]
    curr_open = ohlcv_np[bar_idx, tidx, O]

    features = []

    # 1. 1-bar return (most important MR signal)
    ret1 = (closes[-1] - closes[-2]) / np.maximum(np.abs(closes[-2]), 1e-8)
    features.append(ret1)

    # 2. 2-bar return
    ret2 = (closes[-1] - closes[-3]) / np.maximum(np.abs(closes[-3]), 1e-8) if bar_idx >= 3 else np.zeros(n_stocks)
    features.append(ret2)

    # 3. Open-to-close of last bar (intrabar momentum)
    oc = (closes[-1] - opens_h[-1]) / np.maximum(np.abs(opens_h[-1]), 1e-8)
    features.append(oc)

    # 4. Close position within bar range
    bar_range = highs[-1] - lows[-1]
    close_pos = np.where(bar_range > 1e-8, (closes[-1] - lows[-1]) / bar_range, 0.5)
    features.append(close_pos)

    # 5. Cross-sectional rank of 1-bar return
    ranks = np.argsort(np.argsort(ret1)).astype(np.float32) / max(n_stocks - 1, 1)
    features.append(ranks)

    # 6. SPY 1-bar return (market context)
    spy_ret = float(ret1[0]) if n_stocks > 0 else 0.0
    features.append(np.full(n_stocks, spy_ret, dtype=np.float32))

    # 7. Stock excess return vs SPY
    excess = ret1 - spy_ret
    features.append(excess)

    return np.stack(features, axis=1).astype(np.float32)


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    midx = train_data["macro_indices"].numpy()
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    # Bar range for stop calibration
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    closes = ohlcv[:, tidx, C]
    bar_range_pct = (highs - lows) / np.maximum(closes, 1e-8)
    median_range = np.nanmedian(bar_range_pct, axis=0)
    print(f"  Median 15m bar range: {np.mean(median_range)*100:.3f}%")

    # Build training data
    X_list = []
    y_list = []

    for t in range(10, T - 1):
        feats = compute_features(ohlcv, tidx, midx, t, n_stocks)
        if feats is None:
            continue

        next_open = ohlcv[t, tidx, O]
        next_close = ohlcv[t, tidx, C]
        up = (next_close > next_open).astype(np.float32)

        valid = ~np.isnan(feats).any(axis=1) & (next_open > 0) & ~np.isnan(next_close)
        if valid.sum() > 0:
            X_list.append(feats[valid])
            y_list.append(up[valid])

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    n_features = X.shape[1]
    print(f"  Training: {len(X)} samples, {n_features} features, base rate={y.mean():.3f}")

    feat_mean = np.nanmean(X, axis=0)
    feat_std = np.nanstd(X, axis=0) + 1e-8
    X_norm = np.nan_to_num((X - feat_mean) / feat_std, 0.0)

    # Train with HEAVY regularization
    model = TinyModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCELoss()

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    n = len(X_t)

    # Simple training with early stopping
    best_loss = float('inf')
    patience = 0
    for epoch in range(100):
        perm = torch.randperm(n)
        total_loss = 0
        batches = 0
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            pred = model(X_t[idx])
            loss = criterion(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg = total_loss / batches
        if avg < best_loss - 1e-5:
            best_loss = avg
            patience = 0
        else:
            patience += 1
            if patience > 15:
                break

    model.eval()
    with torch.no_grad():
        preds = model(X_t).numpy().flatten()
        acc = ((preds > 0.5) == y).mean()
    print(f"  Train acc: {acc:.3f} (want ~55-60%, NOT 90%+), epochs: {epoch+1}")

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "macro_idx": train_data["macro_indices"],
        "median_range": median_range,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_stocks": n_stocks,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 10:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    midx = strategy["macro_idx"]
    ohlcv = data["ohlcv"]
    median_range = strategy["median_range"]
    model = strategy["model"]
    n_stocks = strategy["n_stocks"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx
    midx_np = midx.numpy() if isinstance(midx, torch.Tensor) else midx

    feats = compute_features(ohlcv_np, tidx_np, midx_np, bar_idx, n_stocks)
    if feats is None:
        return []

    opens = ohlcv_np[bar_idx, tidx_np, O]
    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(feats).any(axis=1))[0]
    if len(valid) == 0:
        return []

    feat_norm = np.nan_to_num((feats[valid] - strategy["feat_mean"]) / strategy["feat_std"], 0.0)

    with torch.no_grad():
        probs = model(torch.from_numpy(feat_norm)).numpy().flatten()

    # All stocks with any confidence — trade broadly
    candidates = []
    for i, idx in enumerate(valid):
        p = float(probs[i])
        confidence = abs(p - 0.5)
        if confidence < 0.02:
            continue
        direction = "long" if p > 0.5 else "short"
        candidates.append((idx, direction, confidence))

    if not candidates:
        return []

    # Trade all confident predictions, equal weight
    w = 1.0 / len(candidates)
    orders = []

    for idx, direction, conf in candidates:
        ticker = tickers[idx]
        op = float(opens[idx])
        mr = float(median_range[idx])

        # Symmetric stops at 50% of bar range
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
