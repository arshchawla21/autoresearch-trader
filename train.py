"""
v14-nn-selector: Neural network to select best mean-reversion candidate.

Hypothesis: a small NN trained on multiple features (multi-timeframe returns,
volume, ATR ratio, time-of-day) can learn which stocks' moves are most likely
to revert, improving stock selection over "pick the biggest mover."

Still uses the v9 stop framework. The NN replaces only the stock selection logic.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


class ReversionPredictor(nn.Module):
    """Predicts probability that a stock's move will revert in the next bar."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_features(ohlcv, tidx, bar_idx):
    """Compute features for all tradeable stocks at a given bar."""
    opens = ohlcv[bar_idx, tidx, O].numpy()
    n = len(tidx)

    features = np.zeros((n, 8), dtype=np.float32)

    if bar_idx < 21:
        return features, opens

    closes = ohlcv[:bar_idx, tidx, C].numpy()
    highs = ohlcv[:bar_idx, tidx, H].numpy()
    lows = ohlcv[:bar_idx, tidx, L].numpy()
    volumes = ohlcv[:bar_idx, tidx, V].numpy()

    # 1-bar return
    ret1 = (closes[-1] - closes[-2]) / np.maximum(np.abs(closes[-2]), 1e-8)
    # 2-bar return
    ret2 = (closes[-1] - closes[-3]) / np.maximum(np.abs(closes[-3]), 1e-8)
    # 5-bar return
    ret5 = (closes[-1] - closes[-6]) / np.maximum(np.abs(closes[-6]), 1e-8)
    # 20-bar return (trend)
    ret20 = (closes[-1] - closes[-21]) / np.maximum(np.abs(closes[-21]), 1e-8)

    # Volatility: recent ATR as pct
    recent_tr = (highs[-5:] - lows[-5:]) / np.maximum(closes[-5:], 1e-8)
    recent_atr = np.mean(recent_tr, axis=0)

    # Volume ratio: last bar vs 20-bar average
    avg_vol = np.mean(volumes[-20:], axis=0)
    vol_ratio = volumes[-1] / np.maximum(avg_vol, 1e-8)

    # Open-to-close of previous bar
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    oc_ret = (prev_close - prev_open) / np.maximum(np.abs(prev_open), 1e-8)

    # High-low range of previous bar as pct
    prev_range = (highs[-1] - lows[-1]) / np.maximum(closes[-1], 1e-8)

    features[:, 0] = ret1
    features[:, 1] = ret2
    features[:, 2] = ret5
    features[:, 3] = ret20
    features[:, 4] = recent_atr
    features[:, 5] = np.clip(vol_ratio, 0, 10)
    features[:, 6] = oc_ret
    features[:, 7] = prev_range

    return features, opens


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tidx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    T = ohlcv.shape[0]

    ohlcv_np = ohlcv.numpy()
    tidx_np = tidx.numpy()

    # Compute ATR for stop-loss calibration
    closes = ohlcv_np[:, tidx_np, C]
    highs_all = ohlcv_np[:, tidx_np, H]
    lows_all = ohlcv_np[:, tidx_np, L]
    tr = (highs_all - lows_all) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)

    # Build training dataset: for each bar, features -> next bar reversion
    X_list = []
    y_list = []

    for t in range(22, T - 1):
        feats, opens_t = compute_features(ohlcv, tidx, t)
        # Target: did the stock revert? (sign of next bar return opposite to current move)
        next_close = ohlcv_np[t, tidx_np, C]
        next_ret = (next_close - opens_t) / np.maximum(np.abs(opens_t), 1e-8)
        prev_close = ohlcv_np[t - 1, tidx_np, C]
        prev_open = ohlcv_np[t - 1, tidx_np, O]
        prev_move = (prev_close - prev_open) / np.maximum(np.abs(prev_open), 1e-8)

        # Reversion score: positive if next bar moved opposite to prev bar
        reversion = -prev_move * next_ret  # positive = reverted

        valid = ~np.isnan(feats).any(axis=1) & ~np.isnan(reversion) & (opens_t > 0)
        if valid.sum() > 0:
            X_list.append(feats[valid])
            y_list.append(reversion[valid])

    if not X_list:
        return {"tickers": tickers, "tradeable_idx": tidx, "avg_atr_pct": avg_atr_pct, "model": None}

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    # Normalize features
    feat_mean = np.mean(X, axis=0)
    feat_std = np.std(X, axis=0) + 1e-8

    X_norm = (X - feat_mean) / feat_std

    # Train
    model = ReversionPredictor(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y).float().unsqueeze(1)

    model.train()
    batch_size = 512
    n = len(X_t)

    for epoch in range(50):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            pred = model(X_t[batch_idx])
            loss = nn.MSELoss()(pred, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    print(f"  Trained NN on {n} samples, final loss={loss.item():.6f}")

    return {
        "tickers": tickers,
        "tradeable_idx": tidx,
        "avg_atr_pct": avg_atr_pct,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 22:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]
    model = strategy["model"]

    feats, opens = compute_features(ohlcv, tidx, bar_idx)
    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(feats).any(axis=1))[0]
    if len(valid) == 0:
        return []

    if model is None:
        # Fallback to plain mean-reversion
        prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
        prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
        mom = (prev_close - prev_open) / np.maximum(np.abs(prev_open), 1e-8)
        best = valid[np.argmax(np.abs(mom[valid]))]
        direction = "short" if mom[best] > 0 else "long"
    else:
        # Use NN to score candidates
        feat_norm = (feats - strategy["feat_mean"]) / strategy["feat_std"]
        with torch.no_grad():
            scores = model(torch.from_numpy(feat_norm[valid])).numpy().flatten()

        # Higher score = more likely to revert
        best_local = np.argmax(scores)
        best = valid[best_local]

        # Direction: fade the previous bar's move
        prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
        prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
        mom = (prev_close[best] - prev_open[best]) / max(abs(float(prev_open[best])), 1e-8)
        direction = "short" if mom > 0 else "long"

        if scores[best_local] < 0:
            return []  # NN says no good reversion opportunity

    ticker = tickers[best]
    op = float(opens[best])
    atr = float(avg_atr[best])

    sl_dist = max(atr * 0.7, 0.001) * op
    tp_dist = max(atr * 4.0, 0.005) * op

    if direction == "long":
        sl = op - sl_dist
        tp = op + tp_dist
    else:
        sl = op + sl_dist
        tp = op - tp_dist

    return [{
        "ticker": ticker,
        "direction": direction,
        "weight": 1.0,
        "stop_loss": sl,
        "take_profit": tp,
    }]


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print("\n---")
    print(f"sharpe_ratio:     {results.get('sharpe_ratio', 0):.4f}")
    print(f"total_return:     {results.get('total_return', 0)*100:.2f}%")
    print(f"max_drawdown:     {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"win_rate:         {results.get('win_rate', 0)*100:.1f}%")
    print(f"avg_trades/bar:   {results.get('avg_daily_trades', 0):.2f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
