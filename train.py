"""
Autoresearch-trader training script. Single-GPU, single-file.

Neural network approach: train a small MLP on historical features to predict
the sign of intraday (open-to-close) returns. Use model confidence to select
trades and determine long/short direction.

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
torch.manual_seed(42)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Neural network model
# ---------------------------------------------------------------------------

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
    """Compute feature vector for each tradeable stock on a given day.
    Returns (N_tradeable, n_features) tensor. All features use data up to day_idx-1."""
    N = len(tradeable_idx)

    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]
    today_open = ohlcv[day_idx, tradeable_idx, O]

    # Returns at various lookbacks
    def ret(d):
        c = ohlcv[day_idx - d, tradeable_idx, C]
        return (prev_close - c) / c.clamp(min=1e-8)

    r1 = ret(1)    # 1-day
    r3 = ret(3)    # 3-day
    r5 = ret(5)    # 5-day
    r10 = ret(10)  # 10-day
    r20 = ret(20)  # 20-day

    # Gap: open relative to prev close
    gap = (today_open - prev_close) / prev_close.clamp(min=1e-8)

    # ATR as pct of price
    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_prev = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs - lows, (highs - closes_prev).abs()),
        (lows - closes_prev).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.15)

    # Volatility: std of daily returns over 20 days
    closes_20 = ohlcv[day_idx - 20:day_idx, tradeable_idx, C]
    log_rets = torch.log(closes_20[1:] / closes_20[:-1].clamp(min=1e-8))
    vol_20 = log_rets.std(dim=0).clamp(min=1e-6)

    # VIX level (normalized)
    vix_idx = None
    for j, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_idx = j
            break
    vix_val = float(ohlcv[day_idx - 1, vix_idx, C]) / 30.0 if vix_idx is not None else 0.5
    vix_feat = torch.full((N,), vix_val)

    # SPY momentum (same for all stocks)
    spy_idx = all_tickers.index("SPY")
    spy_now = float(ohlcv[day_idx - 1, spy_idx, C])
    spy_20 = float(ohlcv[day_idx - 20, spy_idx, C])
    spy_mom = (spy_now - spy_20) / spy_20
    spy_feat = torch.full((N,), spy_mom)

    # Cross-sectional rank of 10d momentum (normalized to [-1, 1])
    rank_10d = r10.argsort().argsort().float()
    rank_10d = (rank_10d / (N - 1) * 2 - 1) if N > 1 else rank_10d

    features = torch.stack([
        r1, r3, r5, r10, r20,    # momentum at multiple timeframes
        gap,                       # overnight gap
        atr_pct,                   # volatility measure
        vol_20,                    # realized vol
        vix_feat,                  # market fear
        spy_feat,                  # market trend
        rank_10d,                  # cross-sectional rank
    ], dim=1)

    return features, atr_pct


# ---------------------------------------------------------------------------
# Strategy API: build_strategy + generate_orders
# ---------------------------------------------------------------------------

def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]
    D = ohlcv.shape[0]
    N = len(tickers)

    # Build training dataset: features -> sign of intraday return
    min_day = 21
    X_list = []
    y_list = []

    opens = ohlcv[:, tradeable_idx, O]
    closes = ohlcv[:, tradeable_idx, C]

    for d in range(min_day, D):
        feats, _ = compute_features(ohlcv, tradeable_idx, all_tickers, d)
        # Target: sign of (close - open) / open
        intraday = (closes[d] - opens[d]) / opens[d].clamp(min=1e-8)
        X_list.append(feats)
        y_list.append(intraday)

    X = torch.cat(X_list, dim=0)  # (samples, n_features)
    y = torch.cat(y_list, dim=0)  # (samples,)

    # Normalize features
    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - feat_mean) / feat_std

    # Target: binary classification (positive intraday return)
    y_binary = (y > 0).float()

    # Train neural network
    n_features = X_norm.shape[1]
    model = IntradayPredictor(n_features).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_train = X_norm.to(dev)
    y_train = y_binary.to(dev)

    batch_size = 2048
    n_samples = X_train.shape[0]
    n_epochs = min(50, max(5, 300 * 1000 // n_samples))  # adapt epochs to data size

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            pred = model(X_train[idx])
            loss = criterion(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # Find VIX index
    vix_idx = None
    for j, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_idx = j
            break

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

    # Compute features
    feats, atr_pct = compute_features(ohlcv, tradeable_idx, all_tickers, day_idx)
    feats_norm = (feats - feat_mean) / feat_std

    # Model prediction
    with torch.no_grad():
        logits = model(feats_norm.to(dev))
        probs = torch.sigmoid(logits).cpu()

    # VIX scaling
    vix = float(ohlcv[day_idx - 1, vix_idx, C]) if vix_idx is not None else 20.0
    if vix > 35:
        vol_scale = 0.5
    elif vix > 28:
        vol_scale = 0.7
    else:
        vol_scale = 1.0

    # Select high-confidence trades
    n_tickers = len(tickers)
    candidates = []

    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        p = float(probs[i])
        ap = float(atr_pct[i])

        confidence = abs(p - 0.5)
        if confidence < 0.08:  # skip low-confidence
            continue

        if p > 0.58:
            candidates.append({
                "idx": i,
                "direction": "long",
                "confidence": confidence,
                "open": op,
                "atr": ap,
            })
        elif p < 0.42:
            candidates.append({
                "idx": i,
                "direction": "short",
                "confidence": confidence,
                "open": op,
                "atr": ap,
            })

    if not candidates:
        return []

    # Sort by confidence, take top 5
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    candidates = candidates[:5]

    weight_each = vol_scale / len(candidates)
    stop_m = 1.5
    target_m = 2.0

    orders = []
    for c in candidates:
        op = c["open"]
        ap = c["atr"]
        if c["direction"] == "long":
            orders.append({
                "ticker": tickers[c["idx"]],
                "direction": "long",
                "weight": weight_each,
                "stop_loss": op * (1 - ap * stop_m),
                "take_profit": op * (1 + ap * target_m),
            })
        else:
            orders.append({
                "ticker": tickers[c["idx"]],
                "direction": "short",
                "weight": weight_each,
                "stop_loss": op * (1 + ap * stop_m),
                "take_profit": op * (1 - ap * target_m),
            })

    return orders


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

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
