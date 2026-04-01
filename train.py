"""
v31-ml-filter: v23 base (cross-sectional L/S MR) + ML quality filter.

The ML model is trained on v23's EXACT trade outcomes:
- Simulate v23's trades on training data (lb=5, n=2, 0.5% SL/TP)
- For each trade, compute features and record outcome (win/loss)
- Train a simple linear model to predict win probability
- At eval time: run v23, score each pick, skip low-confidence ones

Key: model is intentionally simple (logistic regression via single-layer NN)
to avoid overfitting on ~2000 training trades.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

SL_PCT = 0.005
TP_PCT = 0.005
N_LONG = 2
N_SHORT = 2
LOOKBACK = 5
MIN_PROB = 0.52  # only keep picks where model says >52% win chance


def compute_trade_features(ohlcv_np, tidx, bar_idx, stock_idx):
    """Simple features for a specific stock at a specific bar."""
    if bar_idx < 21:
        return None

    closes = ohlcv_np[:bar_idx, tidx, C]
    highs = ohlcv_np[:bar_idx, tidx, H]
    lows = ohlcv_np[:bar_idx, tidx, L]
    vols = ohlcv_np[:bar_idx, tidx, V]
    opens_h = ohlcv_np[:bar_idx, tidx, O]

    si = stock_idx
    feats = []

    # 1. Magnitude of 5-bar return (how extreme is this setup?)
    ret5 = (closes[-1, si] - closes[-6, si]) / max(abs(closes[-6, si]), 1e-8)
    feats.append(abs(ret5))

    # 2. Z-score from 10-bar SMA
    sma10 = np.mean(closes[-10:, si])
    std10 = np.std(closes[-10:, si])
    z = (closes[-1, si] - sma10) / max(std10, 1e-8)
    feats.append(abs(z))

    # 3. Recent volatility (5-bar)
    rets = np.diff(closes[-6:, si]) / np.maximum(np.abs(closes[-6:-1, si]), 1e-8)
    feats.append(np.std(rets))

    # 4. Volume ratio (current vs 10-bar avg)
    avg_vol = np.mean(vols[-10:, si])
    vol_ratio = vols[-1, si] / max(avg_vol, 1e-8)
    feats.append(min(vol_ratio, 10.0))

    # 5. Candle body direction alignment with MR
    # If we're fading a down move (going long), is the last candle showing recovery?
    last_body = (closes[-1, si] - opens_h[-1, si]) / max(abs(opens_h[-1, si]), 1e-8)
    feats.append(last_body * np.sign(-ret5))  # positive = aligned with MR

    # 6. Bar range relative to recent average
    bar_range = (highs[-1, si] - lows[-1, si]) / max(abs(closes[-2, si]), 1e-8)
    avg_range = np.mean((highs[-10:, si] - lows[-10:, si]) / np.maximum(np.abs(closes[-11:-1, si]), 1e-8))
    feats.append(bar_range / max(avg_range, 1e-8))

    # 7. Cross-sectional rank of this stock's return
    all_ret5 = (closes[-1] - closes[-6]) / np.maximum(np.abs(closes[-6]), 1e-8)
    valid_rets = all_ret5[~np.isnan(all_ret5)]
    rank = np.sum(valid_rets < ret5) / max(len(valid_rets) - 1, 1)
    feats.append(rank)

    # 8. Streak (consecutive bars in same direction)
    streak = 0
    for k in range(1, min(6, bar_idx)):
        if (closes[-k, si] - opens_h[-k, si]) * np.sign(-ret5) > 0:
            streak += 1
        else:
            break
    feats.append(streak / 5.0)

    arr = np.array(feats, dtype=np.float32)
    if np.isnan(arr).any():
        return None
    return arr


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]
    n_stocks = len(tidx)

    print(f"  Simulating v23 trades on {T} training bars...")

    # Simulate v23's exact trades and collect features + outcomes
    X_list = []
    y_list = []

    for t in range(21, T):
        opens = ohlcv[t, tidx, O]
        closes = ohlcv[:t, tidx, C]

        ret = (closes[-1] - closes[-6]) / np.maximum(np.abs(closes[-6]), 1e-8)

        valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < N_LONG + N_SHORT:
            continue

        valid_rets = ret[valid_idx]
        sorted_indices = np.argsort(valid_rets)

        long_picks = valid_idx[sorted_indices[:N_LONG]]
        short_picks = valid_idx[sorted_indices[-N_SHORT:]]

        for idx in long_picks:
            feats = compute_trade_features(ohlcv, tidx, t, idx)
            if feats is None:
                continue
            # Simulate outcome
            entry = ohlcv[t, tidx[idx], O]
            hi = ohlcv[t, tidx[idx], H]
            lo = ohlcv[t, tidx[idx], L]
            cl = ohlcv[t, tidx[idx], C]
            if np.isnan(entry) or np.isnan(hi) or np.isnan(lo) or np.isnan(cl):
                continue
            tp_hit = hi >= entry * (1 + TP_PCT)
            sl_hit = lo <= entry * (1 - SL_PCT)
            if tp_hit and sl_hit:
                outcome = 0.0  # pessimistic
            elif tp_hit:
                outcome = 1.0
            elif sl_hit:
                outcome = 0.0
            else:
                outcome = 1.0 if cl > entry else 0.0
            X_list.append(feats)
            y_list.append(outcome)

        for idx in short_picks:
            feats = compute_trade_features(ohlcv, tidx, t, idx)
            if feats is None:
                continue
            entry = ohlcv[t, tidx[idx], O]
            hi = ohlcv[t, tidx[idx], H]
            lo = ohlcv[t, tidx[idx], L]
            cl = ohlcv[t, tidx[idx], C]
            if np.isnan(entry) or np.isnan(hi) or np.isnan(lo) or np.isnan(cl):
                continue
            tp_hit = lo <= entry * (1 - TP_PCT)
            sl_hit = hi >= entry * (1 + SL_PCT)
            if tp_hit and sl_hit:
                outcome = 0.0
            elif tp_hit:
                outcome = 1.0
            elif sl_hit:
                outcome = 0.0
            else:
                outcome = 1.0 if cl < entry else 0.0
            X_list.append(feats)
            y_list.append(outcome)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    n_features = X.shape[1]

    print(f"  Trade samples: {len(X)}, features: {n_features}")
    print(f"  Base win rate: {y.mean():.3f}")

    # Normalize
    feat_mean = np.mean(X, axis=0)
    feat_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - feat_mean) / feat_std

    # Simple logistic regression (1-layer, no hidden)
    model = nn.Sequential(nn.Linear(n_features, 1), nn.Sigmoid())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCELoss()

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    for epoch in range(200):
        pred = model(X_t)
        loss = criterion(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_t).numpy().flatten()
        above = preds > MIN_PROB
        if above.sum() > 0:
            filtered_wr = y[above].mean()
            print(f"  Filtered WR ({MIN_PROB}+): {filtered_wr:.3f} ({above.sum()}/{len(preds)} trades)")
        else:
            print(f"  No trades above threshold")

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "n_stocks": n_stocks,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 21:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    model = strategy["model"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    ret = (closes[-1] - closes[-1 - LOOKBACK]) / np.maximum(np.abs(closes[-1 - LOOKBACK]), 1e-8)

    valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < N_LONG + N_SHORT:
        return []

    valid_rets = ret[valid_idx]
    sorted_indices = np.argsort(valid_rets)

    long_picks = valid_idx[sorted_indices[:N_LONG]]
    short_picks = valid_idx[sorted_indices[-N_SHORT:]]

    # Score each pick with ML model
    scored = []
    for idx in long_picks:
        feats = compute_trade_features(ohlcv_np, tidx_np, bar_idx, idx)
        if feats is None:
            continue
        feat_norm = (feats - strategy["feat_mean"]) / strategy["feat_std"]
        with torch.no_grad():
            prob = float(model(torch.from_numpy(feat_norm).unsqueeze(0)).item())
        if prob >= MIN_PROB:
            scored.append((idx, "long", prob))

    for idx in short_picks:
        feats = compute_trade_features(ohlcv_np, tidx_np, bar_idx, idx)
        if feats is None:
            continue
        feat_norm = (feats - strategy["feat_mean"]) / strategy["feat_std"]
        with torch.no_grad():
            prob = float(model(torch.from_numpy(feat_norm).unsqueeze(0)).item())
        if prob >= MIN_PROB:
            scored.append((idx, "short", prob))

    if not scored:
        return []

    w = 1.0 / len(scored)

    orders = []
    for idx, direction, prob in scored:
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
