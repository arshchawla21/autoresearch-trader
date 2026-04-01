"""
Autoresearch-trader training script. Single-GPU, single-file.

v20: Ensemble of 3 classification NNs (different seeds) averaged.
Same architecture as v11 (the best so far). Ensemble should reduce
variance across slices without adding complexity.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import numpy as np
import torch
import torch.nn as nn

from prepare import O, H, L, C, evaluate

t_start = time.time()
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


def compute_features(ohlcv, tradeable_idx, all_tickers, day_idx):
    N = len(tradeable_idx)
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]
    today_open = ohlcv[day_idx, tradeable_idx, O]

    def ret(d):
        c = ohlcv[day_idx - d, tradeable_idx, C]
        return (prev_close - c) / c.clamp(min=1e-8)

    r1, r3, r5, r10, r20 = ret(1), ret(3), ret(5), ret(10), ret(20)
    gap = (today_open - prev_close) / prev_close.clamp(min=1e-8)

    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_p = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(torch.max(highs - lows, (highs - closes_p).abs()), (lows - closes_p).abs())
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

    features = torch.stack([
        r1, r3, r5, r10, r20, gap, atr_pct, vol_20,
        torch.full((N,), vix_val),
        torch.full((N,), spy_mom),
        rank_10d,
    ], dim=1)

    return features, atr_pct, spy_mom, vix_val * 30.0


def train_single_model(X_norm, y, seed):
    torch.manual_seed(seed)
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
    return model


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
        feats, _, _, _ = compute_features(ohlcv, tradeable_idx, all_tickers, d)
        intraday = (closes[d] - opens[d]) / opens[d].clamp(min=1e-8)
        labels = (intraday > 0).float()
        X_list.append(feats)
        y_list.append(labels)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - feat_mean) / feat_std

    # Train 3 models with different seeds
    models = []
    for seed in [42, 123, 7]:
        models.append(train_single_model(X_norm, y, seed))

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "all_tickers": all_tickers,
        "models": models,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    all_tickers = strategy["all_tickers"]
    models = strategy["models"]
    feat_mean = strategy["feat_mean"]
    feat_std = strategy["feat_std"]
    n_tickers = len(tickers)

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    feats, atr_pct, spy_mom, vix = compute_features(
        ohlcv, tradeable_idx, all_tickers, day_idx
    )
    feats_norm = (feats - feat_mean) / feat_std

    # Average predictions from all models
    with torch.no_grad():
        probs_sum = torch.zeros(n_tickers)
        for model in models:
            logits = model(feats_norm.to(dev)).cpu()
            probs_sum += torch.sigmoid(logits)
        probs = probs_sum / len(models)

    vol_scale = 0.5 if vix > 35 else 0.7 if vix > 28 else 1.0

    long_thresh = 0.58
    short_thresh = 0.42

    candidates = []
    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        p = float(probs[i])
        ap = float(atr_pct[i])

        if p > long_thresh:
            candidates.append(("long", i, p - long_thresh, op, ap))
        elif p < short_thresh:
            candidates.append(("short", i, short_thresh - p, op, ap))

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
