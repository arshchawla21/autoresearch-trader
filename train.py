"""
Autoresearch-trader training script. Single-GPU, single-file.

v25: Pure direction play with wide stops.
Set TP/SL at 3x ATR — they almost never trigger. Every trade exits
at close. Win rate = directional accuracy (~53% base + NN edge).
Long only. Added day-of-week and market breadth features.

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
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_features(ohlcv, tradeable_idx, all_tickers, day_idx, dates=None):
    N = len(tradeable_idx)
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]
    prev_open = ohlcv[day_idx - 1, tradeable_idx, O]
    prev_high = ohlcv[day_idx - 1, tradeable_idx, H]
    prev_low = ohlcv[day_idx - 1, tradeable_idx, L]
    today_open = ohlcv[day_idx, tradeable_idx, O]

    def ret(d):
        c = ohlcv[day_idx - d, tradeable_idx, C]
        return (prev_close - c) / c.clamp(min=1e-8)

    r1, r3, r5, r10, r20 = ret(1), ret(3), ret(5), ret(10), ret(20)
    gap = (today_open - prev_close) / prev_close.clamp(min=1e-8)

    highs_14 = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows_14 = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_14p = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(torch.max(highs_14 - lows_14, (highs_14 - closes_14p).abs()), (lows_14 - closes_14p).abs())
    atr_pct = (tr.mean(dim=0) / today_open.clamp(min=1e-8)).clamp(0.003, 0.15)

    closes_20 = ohlcv[day_idx - 20:day_idx, tradeable_idx, C]
    log_rets = torch.log(closes_20[1:] / closes_20[:-1].clamp(min=1e-8))
    vol_20 = log_rets.std(dim=0).clamp(min=1e-6)

    vix_idx = next((j for j, t in enumerate(all_tickers) if t == "^VIX"), None)
    spy_idx = all_tickers.index("SPY")
    vix_val = float(ohlcv[day_idx - 1, vix_idx, C]) / 30.0 if vix_idx else 0.5
    spy_c = float(ohlcv[day_idx - 1, spy_idx, C])
    spy_20 = float(ohlcv[day_idx - 20, spy_idx, C])
    spy_mom = (spy_c - spy_20) / spy_20

    rank_10d = r10.argsort().argsort().float()
    rank_10d = (rank_10d / max(N - 1, 1) * 2 - 1)

    # Candlestick features
    prev_range = (prev_high - prev_low).clamp(min=1e-8)
    close_position = (prev_close - prev_low) / prev_range
    body_ratio = (prev_close - prev_open).abs() / prev_range

    vol_20d = ohlcv[day_idx - 20:day_idx, tradeable_idx, V]
    vol_avg = vol_20d.mean(dim=0).clamp(min=1.0)
    vol_yesterday = ohlcv[day_idx - 1, tradeable_idx, V]
    vol_ratio = (vol_yesterday / vol_avg).clamp(0.1, 5.0)

    gap_atr = gap / atr_pct.clamp(min=1e-6)

    # NEW: Market breadth — fraction of tradeable stocks that were up yesterday
    prev_day_close = ohlcv[day_idx - 1, tradeable_idx, C]
    prev_day_open = ohlcv[day_idx - 1, tradeable_idx, O]
    breadth = (prev_day_close > prev_day_open).float().mean().item()

    # NEW: Day of week (0=Mon, 4=Fri) encoded as sin/cos
    if dates is not None and day_idx < len(dates):
        dow = dates[day_idx].weekday()
    else:
        dow = 2  # default to Wednesday
    dow_sin = np.sin(2 * np.pi * dow / 5)
    dow_cos = np.cos(2 * np.pi * dow / 5)

    features = torch.stack([
        r1, r3, r5, r10, r20, gap, atr_pct, vol_20,
        torch.full((N,), vix_val),
        torch.full((N,), spy_mom),
        rank_10d,
        close_position, body_ratio, vol_ratio, gap_atr,
        torch.full((N,), breadth),
        torch.full((N,), dow_sin),
        torch.full((N,), dow_cos),
    ], dim=1)

    return features, atr_pct, vix_val * 30.0


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]
    dates = train_data["dates"]
    D = ohlcv.shape[0]

    opens = ohlcv[:, tradeable_idx, O]
    closes = ohlcv[:, tradeable_idx, C]

    min_day = 21
    X_list, y_list = [], []
    for d in range(min_day, D):
        feats, _, _ = compute_features(ohlcv, tradeable_idx, all_tickers, d, dates)
        intraday = (closes[d] - opens[d]) / opens[d].clamp(min=1e-8)
        labels = (intraday > 0).float()
        X_list.append(feats)
        y_list.append(labels)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - feat_mean) / feat_std

    n_features = X_norm.shape[1]
    model = IntradayPredictor(n_features).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    X_train = X_norm.to(dev)
    y_train = y.to(dev)

    batch_size = 2048
    n_samples = X_train.shape[0]
    n_epochs = min(50, max(5, 300 * 1000 // n_samples))

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:min(start + batch_size, n_samples)]
            loss = criterion(model(X_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "all_tickers": all_tickers,
        "dates": dates,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    all_tickers = strategy["all_tickers"]
    dates = strategy["dates"]
    model = strategy["model"]
    feat_mean = strategy["feat_mean"]
    feat_std = strategy["feat_std"]
    n_tickers = len(tickers)

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    feats, atr_pct, vix = compute_features(
        ohlcv, tradeable_idx, all_tickers, day_idx, dates
    )
    feats_norm = (feats - feat_mean) / feat_std

    with torch.no_grad():
        logits = model(feats_norm.to(dev)).cpu()
        probs = torch.sigmoid(logits)

    vol_scale = 0.5 if vix > 35 else 0.7 if vix > 28 else 1.0

    # Long only, moderate threshold
    long_thresh = 0.53

    candidates = []
    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        p = float(probs[i])
        ap = float(atr_pct[i])

        if p > long_thresh:
            candidates.append((i, p, op, ap))

    if not candidates:
        return []

    # Top 5 by confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:5]

    weight_each = vol_scale / len(candidates)

    # Very wide TP/SL — almost never trigger, pure close exit
    stop_m = 3.0
    target_m = 3.0

    orders = []
    for (idx, _, op, ap) in candidates:
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": weight_each,
            "stop_loss": op * (1 - ap * stop_m),
            "take_profit": op * (1 + ap * target_m),
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
