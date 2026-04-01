"""
Autoresearch-trader training script. Single-GPU, single-file.

Pure NN with rich features and high-confidence thresholds.
Richer features: range position, volume signals, multi-timeframe momentum,
cross-sectional ranks. High selectivity (0.58/0.42) proven to work.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import numpy as np
import torch
import torch.nn as nn

from prepare import O, H, L, C, V, evaluate

t_start = time.time()
torch.manual_seed(42)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IntradayPredictor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_features(ohlcv, tradeable_idx, all_tickers, day_idx):
    N = len(tradeable_idx)
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]
    today_open = ohlcv[day_idx, tradeable_idx, O]

    # Multi-timeframe returns
    def ret(d):
        c = ohlcv[day_idx - d, tradeable_idx, C]
        return (prev_close - c) / c.clamp(min=1e-8)
    r1, r3, r5, r10, r20 = ret(1), ret(3), ret(5), ret(10), ret(20)

    # Gap
    gap = (today_open - prev_close) / prev_close.clamp(min=1e-8)

    # ATR
    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_p = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(torch.max(highs - lows, (highs - closes_p).abs()), (lows - closes_p).abs())
    atr_pct = (tr.mean(dim=0) / today_open.clamp(min=1e-8)).clamp(0.003, 0.15)

    # Realized vol
    closes_20 = ohlcv[day_idx - 20:day_idx, tradeable_idx, C]
    log_rets = torch.log(closes_20[1:] / closes_20[:-1].clamp(min=1e-8))
    vol_20 = log_rets.std(dim=0).clamp(min=1e-6)

    # Yesterday's range position: where did it close within its range?
    yest_h = ohlcv[day_idx - 1, tradeable_idx, H]
    yest_l = ohlcv[day_idx - 1, tradeable_idx, L]
    yest_range = (yest_h - yest_l).clamp(min=1e-8)
    range_pos = (prev_close - yest_l) / yest_range  # 0=low, 1=high

    # Yesterday's range relative to ATR (narrow or wide day?)
    range_vs_atr = yest_range / tr.mean(dim=0).clamp(min=1e-8)

    # Volume: yesterday vs 20-day average
    vol_yest = ohlcv[day_idx - 1, tradeable_idx, V]
    vol_20_avg = ohlcv[day_idx - 20:day_idx, tradeable_idx, V].mean(dim=0).clamp(min=1)
    vol_ratio = vol_yest / vol_20_avg

    # Market-level features
    vix_idx = next((j for j, t in enumerate(all_tickers) if t == "^VIX"), None)
    spy_idx = all_tickers.index("SPY")
    vix_val = float(ohlcv[day_idx - 1, vix_idx, C]) / 30.0 if vix_idx else 0.5
    spy_c = float(ohlcv[day_idx - 1, spy_idx, C])
    spy_20 = float(ohlcv[day_idx - 20, spy_idx, C])
    spy_5 = float(ohlcv[day_idx - 5, spy_idx, C])
    spy_mom20 = (spy_c - spy_20) / spy_20
    spy_mom5 = (spy_c - spy_5) / spy_5

    # Cross-sectional ranks
    rank_10d = r10.argsort().argsort().float()
    rank_10d = (rank_10d / max(N - 1, 1) * 2 - 1)
    rank_5d = r5.argsort().argsort().float()
    rank_5d = (rank_5d / max(N - 1, 1) * 2 - 1)

    features = torch.stack([
        r1, r3, r5, r10, r20,              # momentum
        gap,                                 # overnight gap
        atr_pct, vol_20,                     # volatility
        range_pos,                           # range position
        range_vs_atr,                        # range ratio
        vol_ratio,                           # volume signal
        torch.full((N,), vix_val),           # VIX
        torch.full((N,), spy_mom20),         # SPY 20d
        torch.full((N,), spy_mom5),          # SPY 5d
        rank_10d, rank_5d,                   # cross-sectional ranks
    ], dim=1)

    return features, atr_pct


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]
    D = ohlcv.shape[0]

    opens = ohlcv[:, tradeable_idx, O]
    closes = ohlcv[:, tradeable_idx, C]

    min_day = 21
    X_list, y_list = [], []
    for d in range(min_day, D):
        feats, _ = compute_features(ohlcv, tradeable_idx, all_tickers, d)
        intraday = (closes[d] - opens[d]) / opens[d].clamp(min=1e-8)
        X_list.append(feats)
        y_list.append(intraday)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - feat_mean) / feat_std
    y_binary = (y > 0).float()

    # Sample weights: more recent data weighted higher
    n_samples = X_norm.shape[0]
    n_tickers = len(tickers)
    n_days = n_samples // n_tickers
    day_weights = torch.linspace(0.5, 1.0, n_days).repeat_interleave(n_tickers)
    if len(day_weights) > n_samples:
        day_weights = day_weights[:n_samples]
    elif len(day_weights) < n_samples:
        day_weights = torch.cat([day_weights, torch.ones(n_samples - len(day_weights))])

    n_features = X_norm.shape[1]
    model = IntradayPredictor(n_features).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    X_train = X_norm.to(dev)
    y_train = y_binary.to(dev)
    w_train = day_weights.to(dev)

    batch_size = 2048
    n_epochs = min(50, max(5, 300 * 1000 // n_samples))

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:min(start + batch_size, n_samples)]
            pred = model(X_train[idx])
            loss = (criterion(pred, y_train[idx]) * w_train[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

    vix_idx = next((j for j, t in enumerate(all_tickers) if t == "^VIX"), None)

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "all_tickers": all_tickers,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "vix_idx": vix_idx,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    all_tickers = strategy["all_tickers"]
    model = strategy["model"]
    feat_mean = strategy["feat_mean"]
    feat_std = strategy["feat_std"]
    vix_idx = strategy["vix_idx"]

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]

    feats, atr_pct = compute_features(ohlcv, tradeable_idx, all_tickers, day_idx)
    feats_norm = (feats - feat_mean) / feat_std

    with torch.no_grad():
        probs = torch.sigmoid(model(feats_norm.to(dev))).cpu()

    vix = float(ohlcv[day_idx - 1, vix_idx, C]) if vix_idx else 20.0
    vol_scale = 0.5 if vix > 35 else 0.7 if vix > 28 else 1.0

    n_tickers = len(tickers)
    candidates = []

    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        p = float(probs[i])
        ap = float(atr_pct[i])
        confidence = abs(p - 0.5)

        if p > 0.58:
            candidates.append(("long", i, confidence, op, ap))
        elif p < 0.42:
            candidates.append(("short", i, confidence, op, ap))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:5]

    weight_each = vol_scale / len(candidates)
    stop_m = 1.5
    target_m = 2.0

    orders = []
    for (direction, idx, _, op, ap) in candidates:
        if direction == "long":
            orders.append({
                "ticker": tickers[idx],
                "direction": "long",
                "weight": weight_each,
                "stop_loss": op * (1 - ap * stop_m),
                "take_profit": op * (1 + ap * target_m),
            })
        else:
            orders.append({
                "ticker": tickers[idx],
                "direction": "short",
                "weight": weight_each,
                "stop_loss": op * (1 + ap * stop_m),
                "take_profit": op * (1 - ap * target_m),
            })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")

    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"sharpe_ratio:     {results['sharpe_ratio']:.4f}")
    print(f"total_return:     {results['total_return']*100:.2f}%")
    print(f"max_drawdown:     {results['max_drawdown']*100:.2f}%")
    print(f"win_rate:         {results['win_rate']*100:.1f}%")
    print(f"avg_daily_trades: {results['avg_daily_trades']:.1f}")
    print(f"num_slices:       {results['num_slices']}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
