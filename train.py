"""
Autoresearch-trader training script. Single-GPU, single-file.

v39: Confidence-weighted L10/S10.
Based on v37 (Sharpe=0.73). Instead of equal weight, allocate more
capital to the NN's strongest signals. Positions weighted by |prob - 0.5|.
Also: wider targets (3x ATR) and regression head to predict magnitude.

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


class DualHead(nn.Module):
    """Classification head for direction + regression head for magnitude."""
    def __init__(self, n_features):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.direction_head = nn.Linear(64, 1)
        self.magnitude_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        direction = self.direction_head(h).squeeze(-1)
        magnitude = self.magnitude_head(h).squeeze(-1)
        return direction, magnitude


def compute_features(ohlcv, tradeable_idx, all_tickers, day_idx):
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
    spy_5 = float(ohlcv[day_idx - 5, spy_idx, C])
    spy_mom = (spy_c - spy_20) / spy_20
    spy_mom5 = (spy_c - spy_5) / spy_5

    rank_10d = r10.argsort().argsort().float()
    rank_10d = (rank_10d / max(N - 1, 1) * 2 - 1)
    rank_5d = r5.argsort().argsort().float()
    rank_5d = (rank_5d / max(N - 1, 1) * 2 - 1)

    prev_range = (prev_high - prev_low).clamp(min=1e-8)
    close_position = (prev_close - prev_low) / prev_range
    body_ratio = (prev_close - prev_open).abs() / prev_range

    vol_20d = ohlcv[day_idx - 20:day_idx, tradeable_idx, V]
    vol_avg = vol_20d.mean(dim=0).clamp(min=1.0)
    vol_yesterday = ohlcv[day_idx - 1, tradeable_idx, V]
    vol_ratio = (vol_yesterday / vol_avg).clamp(0.1, 5.0)

    gap_atr = gap / atr_pct.clamp(min=1e-6)

    prev_day_close = ohlcv[day_idx - 1, tradeable_idx, C]
    prev_day_open = ohlcv[day_idx - 1, tradeable_idx, O]
    breadth = (prev_day_close > prev_day_open).float().mean().item()

    features = torch.stack([
        r1, r3, r5, r10, r20, gap, atr_pct, vol_20,
        torch.full((N,), vix_val),
        torch.full((N,), spy_mom),
        torch.full((N,), spy_mom5),
        rank_10d, rank_5d,
        close_position, body_ratio, vol_ratio, gap_atr,
        torch.full((N,), breadth),
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
    X_list, y_dir_list, y_mag_list = [], [], []
    for d in range(min_day, D):
        feats, _ = compute_features(ohlcv, tradeable_idx, all_tickers, d)
        intraday = (closes[d] - opens[d]) / opens[d].clamp(min=1e-8)
        direction = (intraday > 0).float()
        magnitude = intraday.abs().clamp(0, 0.10)  # clip magnitude
        X_list.append(feats)
        y_dir_list.append(direction)
        y_mag_list.append(magnitude)

    X = torch.cat(X_list, dim=0)
    y_dir = torch.cat(y_dir_list, dim=0)
    y_mag = torch.cat(y_mag_list, dim=0)

    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - feat_mean) / feat_std

    mag_mean = y_mag.mean()
    mag_std = y_mag.std().clamp(min=1e-8)
    y_mag_scaled = (y_mag - mag_mean) / mag_std

    n_features = X_norm.shape[1]
    model = DualHead(n_features).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    X_train = X_norm.to(dev)
    y_dir_train = y_dir.to(dev)
    y_mag_train = y_mag_scaled.to(dev)

    batch_size = 2048
    n_samples = X_train.shape[0]
    n_epochs = min(60, max(10, 400 * 1000 // n_samples))

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:min(start + batch_size, n_samples)]
            dir_logits, mag_pred = model(X_train[idx])
            loss = bce(dir_logits, y_dir_train[idx]) + 0.5 * mse(mag_pred, y_mag_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "all_tickers": all_tickers,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "mag_mean": float(mag_mean),
        "mag_std": float(mag_std),
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    all_tickers = strategy["all_tickers"]
    model = strategy["model"]
    feat_mean = strategy["feat_mean"]
    feat_std = strategy["feat_std"]
    n_tickers = len(tickers)

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    feats, atr_pct = compute_features(
        ohlcv, tradeable_idx, all_tickers, day_idx
    )
    feats_norm = (feats - feat_mean) / feat_std

    with torch.no_grad():
        dir_logits, mag_pred = model(feats_norm.to(dev))
        dir_logits = dir_logits.cpu()
        probs = torch.sigmoid(dir_logits)

    scored = []
    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        p = float(probs[i])
        ap = float(atr_pct[i])
        confidence = abs(p - 0.5)  # how far from 0.5 = confidence
        scored.append((i, p, op, ap, confidence))

    if len(scored) < 10:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    n_long = 10
    n_short = 10
    longs = scored[:n_long]
    shorts = scored[-n_short:]

    # Confidence-weighted allocation
    long_confs = [s[4] for s in longs]
    short_confs = [s[4] for s in shorts]
    total_conf = sum(long_confs) + sum(short_confs) + 1e-8

    orders = []
    for (idx, _, op, ap, conf) in longs:
        w = conf / total_conf
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": w,
            "stop_loss": op * (1 - ap * 1.5),
            "take_profit": op * (1 + ap * 2.5),
        })

    for (idx, _, op, ap, conf) in shorts:
        w = conf / total_conf
        orders.append({
            "ticker": tickers[idx],
            "direction": "short",
            "weight": w,
            "stop_loss": op * (1 + ap * 1.5),
            "take_profit": op * (1 - ap * 2.5),
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
