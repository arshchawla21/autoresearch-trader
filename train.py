"""
v19-selective-ml: ML model that makes selective bets targeting ~1% moves.

Philosophy:
- Don't trade every bar. Wait for high-conviction setups.
- SL/TP at ~1% — expect these to resolve over 4+ hours (16+ bars).
- Train an ML model on rich features to predict which setups have >50% chance
  of hitting TP before SL.
- The model should be SELECTIVE — only trade when confident.

Features engineered:
- Multi-timeframe price momentum (1, 3, 5, 10, 20 bars)
- Intrabar microstructure (close position, shadow ratios, body size)
- Volume dynamics (relative volume, volume trend)
- Volatility regime (recent vs historical)
- Cross-sectional rank (how extreme is this stock vs peers)
- Cross-asset context (SPY direction, VIX level)
- Mean-reversion z-score (deviation from short-term moving average)

Target: will the stock move 1% in the MR direction before hitting 1% against?
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

# Trade parameters
SL_PCT = 0.01   # 1% stop loss
TP_PCT = 0.01   # 1% take profit (symmetric for clean WR signal)
MIN_CONFIDENCE = 0.58  # Only trade when model says >58%


class SelectiveModel(nn.Module):
    """Predicts P(TP hit before SL) for a mean-reversion trade."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def compute_features(ohlcv_np, tidx, bar_idx, n_stocks):
    """Rich feature set for a single bar across all stocks."""
    if bar_idx < 21:
        return None

    closes = ohlcv_np[:bar_idx, tidx, C]
    opens_h = ohlcv_np[:bar_idx, tidx, O]
    highs = ohlcv_np[:bar_idx, tidx, H]
    lows = ohlcv_np[:bar_idx, tidx, L]
    vols = ohlcv_np[:bar_idx, tidx, V]
    curr_open = ohlcv_np[bar_idx, tidx, O]

    feats = []

    # === Multi-timeframe returns ===
    for lb in [1, 2, 3, 5, 10, 20]:
        ret = (closes[-1] - closes[-1 - lb]) / np.maximum(np.abs(closes[-1 - lb]), 1e-8)
        feats.append(ret)

    # === Intrabar microstructure (last bar) ===
    bar_range = highs[-1] - lows[-1]
    safe_range = np.maximum(bar_range, 1e-8)

    # Close position in bar (0=low, 1=high)
    close_pos = (closes[-1] - lows[-1]) / safe_range
    feats.append(close_pos)

    # Body size relative to range
    body = np.abs(closes[-1] - opens_h[-1])
    body_ratio = body / safe_range
    feats.append(body_ratio)

    # Upper shadow ratio
    body_top = np.maximum(closes[-1], opens_h[-1])
    upper_shadow = (highs[-1] - body_top) / safe_range
    feats.append(upper_shadow)

    # Lower shadow ratio
    body_bot = np.minimum(closes[-1], opens_h[-1])
    lower_shadow = (body_bot - lows[-1]) / safe_range
    feats.append(lower_shadow)

    # === Volume dynamics ===
    avg_vol_10 = np.mean(vols[-10:], axis=0)
    vol_ratio = vols[-1] / np.maximum(avg_vol_10, 1e-8)
    feats.append(np.clip(vol_ratio, 0, 10))

    # Volume trend (last 3 vs prev 3)
    recent_vol = np.mean(vols[-3:], axis=0)
    older_vol = np.mean(vols[-6:-3], axis=0)
    vol_trend = recent_vol / np.maximum(older_vol, 1e-8)
    feats.append(np.clip(vol_trend, 0, 10))

    # === Volatility ===
    rets_5 = np.diff(closes[-6:], axis=0) / np.maximum(np.abs(closes[-6:-1]), 1e-8)
    recent_vol_5 = np.std(rets_5, axis=0)
    feats.append(recent_vol_5)

    rets_20 = np.diff(closes[-21:], axis=0) / np.maximum(np.abs(closes[-21:-1]), 1e-8)
    hist_vol_20 = np.std(rets_20, axis=0)
    vol_regime = recent_vol_5 / np.maximum(hist_vol_20, 1e-8)
    feats.append(vol_regime)

    # === Mean-reversion z-score ===
    sma_10 = np.mean(closes[-10:], axis=0)
    std_10 = np.std(closes[-10:], axis=0)
    z_score = (closes[-1] - sma_10) / np.maximum(std_10, 1e-8)
    feats.append(z_score)

    # === Cross-sectional ===
    ret1 = feats[0]  # 1-bar return
    ranks = np.argsort(np.argsort(ret1)).astype(np.float32) / max(n_stocks - 1, 1)
    feats.append(ranks)

    # === Gap ===
    gap = (curr_open - closes[-1]) / np.maximum(np.abs(closes[-1]), 1e-8)
    feats.append(gap)

    # === Cross-asset ===
    spy_ret_5 = float((closes[-1, 0] - closes[-6, 0]) / max(abs(closes[-6, 0]), 1e-8))
    feats.append(np.full(n_stocks, spy_ret_5, dtype=np.float32))

    # Streak (consecutive up/down bars)
    streak = (np.sign(closes[-1] - opens_h[-1]) +
              np.sign(closes[-2] - opens_h[-2]) +
              np.sign(closes[-3] - opens_h[-3]))
    feats.append(streak / 3.0)

    return np.stack(feats, axis=1).astype(np.float32)  # (N, F)


def simulate_trade_outcome(ohlcv_np, tidx, entry_bar, stock_idx, direction, sl_pct, tp_pct, max_bars=16):
    """
    Simulate: starting at entry_bar open, does TP or SL hit first
    over the next max_bars? Returns 1.0 if TP hits, 0.0 if SL hits,
    0.5 if neither (held to end).
    """
    entry_price = ohlcv_np[entry_bar, tidx[stock_idx], O]
    if entry_price <= 0 or np.isnan(entry_price):
        return None

    T = ohlcv_np.shape[0]
    end_bar = min(entry_bar + max_bars, T)

    for t in range(entry_bar, end_bar):
        hi = ohlcv_np[t, tidx[stock_idx], H]
        lo = ohlcv_np[t, tidx[stock_idx], L]

        if direction == "long":
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            if hi >= tp_price:
                return 1.0
            if lo <= sl_price:
                return 0.0
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
            if lo <= tp_price:
                return 1.0
            if hi >= sl_price:
                return 0.0

    # Neither hit — use final close
    final_close = ohlcv_np[end_bar - 1, tidx[stock_idx], C]
    if direction == "long":
        return 1.0 if final_close > entry_price else 0.0
    else:
        return 1.0 if final_close < entry_price else 0.0


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    print(f"  Building training dataset from {T} bars, {n_stocks} stocks...")

    # Build training data
    X_list = []
    y_list = []
    dir_list = []  # track direction for debugging

    for t in range(21, T - 16):  # need 16 bars lookahead for outcome
        feats = compute_features(ohlcv, tidx, t, n_stocks)
        if feats is None:
            continue

        # For each stock, determine MR direction and simulate outcome
        # MR direction: fade the 2-bar return
        ret2 = (ohlcv[t - 1, tidx, C] - ohlcv[t - 3, tidx, C]) / np.maximum(
            np.abs(ohlcv[t - 3, tidx, C]), 1e-8)

        for si in range(n_stocks):
            if np.isnan(feats[si]).any():
                continue

            direction = "short" if ret2[si] > 0 else "long"
            outcome = simulate_trade_outcome(ohlcv, tidx, t, si, direction, SL_PCT, TP_PCT)
            if outcome is None:
                continue

            X_list.append(feats[si])
            y_list.append(outcome)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    n_features = X.shape[1]

    print(f"  Samples: {len(X)}, features: {n_features}")
    print(f"  MR TP hit rate: {y.mean():.3f} (>0.5 means MR has edge)")

    # Normalize
    feat_mean = np.mean(X, axis=0)
    feat_std = np.std(X, axis=0) + 1e-8
    X_norm = np.nan_to_num((X - feat_mean) / feat_std, 0.0)

    # Train with moderate regularization
    model = SelectiveModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    criterion = nn.BCELoss()

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    n = len(X_t)
    best_loss = float('inf')
    patience = 0

    for epoch in range(200):
        perm = torch.randperm(n)
        total_loss = 0
        batches = 0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
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
            if patience > 20:
                break

    model.eval()
    with torch.no_grad():
        preds = model(X_t).numpy().flatten()
        acc = ((preds > 0.5) == y).mean()
        # Show confidence distribution
        confident = preds[preds > MIN_CONFIDENCE]
        if len(confident) > 0:
            confident_acc = y[preds > MIN_CONFIDENCE].mean()
            print(f"  Train acc: {acc:.3f}, confident ({MIN_CONFIDENCE}+) acc: {confident_acc:.3f} ({len(confident)} samples)")
        else:
            print(f"  Train acc: {acc:.3f}, no confident predictions")

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_stocks": n_stocks,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 21:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    model = strategy["model"]
    n_stocks = strategy["n_stocks"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    feats = compute_features(ohlcv_np, tidx_np, bar_idx, n_stocks)
    if feats is None:
        return []

    opens = ohlcv_np[bar_idx, tidx_np, O]

    # MR direction: fade 2-bar return
    closes = ohlcv_np[:bar_idx, tidx_np, C]
    ret2 = (closes[-1] - closes[-3]) / np.maximum(np.abs(closes[-3]), 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(feats).any(axis=1))[0]
    if len(valid) == 0:
        return []

    # Get model predictions
    feat_norm = np.nan_to_num((feats[valid] - strategy["feat_mean"]) / strategy["feat_std"], 0.0)
    with torch.no_grad():
        probs = model(torch.from_numpy(feat_norm)).numpy().flatten()

    # Only trade high-confidence predictions
    candidates = []
    for i, idx in enumerate(valid):
        p = float(probs[i])
        if p < MIN_CONFIDENCE:
            continue
        direction = "short" if ret2[idx] > 0 else "long"
        candidates.append((idx, direction, p))

    if not candidates:
        return []

    # Take top 3 most confident, equal weight
    candidates.sort(key=lambda x: -x[2])
    top = candidates[:3]
    w = 1.0 / len(top)

    orders = []
    for idx, direction, prob in top:
        op = float(opens[idx])

        if direction == "long":
            sl = op * (1 - SL_PCT)
            tp = op * (1 + TP_PCT)
        else:
            sl = op * (1 + SL_PCT)
            tp = op * (1 - TP_PCT)

        orders.append({
            "ticker": tickers[idx],
            "direction": direction,
            "weight": w,
            "stop_loss": sl,
            "take_profit": tp,
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% "
          f"dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% "
          f"trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
