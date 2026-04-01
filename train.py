"""
Autoresearch-trader training script. Single-GPU, single-file.

v55: Dual-model combo: Huber regression + BCE classification.
Train two independent models with different loss functions:
- Model A: Huber regression on clipped returns (captures magnitude)
- Model B: BCE classification on direction (captures probability)
At inference, combine: rank by 0.5*normalized_return_pred + 0.5*probability.
This fuses two complementary signals — magnitude and direction.

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


class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def train_model(X_norm, y, n_features, criterion, seed=42):
    torch.manual_seed(seed)
    model = Net(n_features).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_train = X_norm.to(dev)
    y_train = y.to(dev)

    batch_size = 2048
    n_samples = X_train.shape[0]
    n_epochs = min(60, max(10, 400 * 1000 // n_samples))

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:min(start + batch_size, n_samples)]
            pred = model(X_train[idx])
            loss = criterion(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def simulate_day(scores, today_open, today_high, today_low, today_close,
                 atr_pct, n_tickers, sl_mult, tp_mult):
    scored = []
    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0:
            continue
        scored.append((i, float(scores[i]), op, float(atr_pct[i])))

    if len(scored) < 10:
        return 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    longs = scored[:10]
    shorts = scored[-10:]
    w = 1.0 / 20

    daily_ret = 0.0
    for (idx, _, op, ap) in longs:
        sl = op * (1 - ap * sl_mult)
        tp = op * (1 + ap * tp_mult)
        hi, lo, cl = float(today_high[idx]), float(today_low[idx]), float(today_close[idx])
        if lo <= sl:
            daily_ret += w * (sl - op) / op
        elif hi >= tp:
            daily_ret += w * (tp - op) / op
        else:
            daily_ret += w * (cl - op) / op

    for (idx, _, op, ap) in shorts:
        sl = op * (1 + ap * sl_mult)
        tp = op * (1 - ap * tp_mult)
        hi, lo, cl = float(today_high[idx]), float(today_low[idx]), float(today_close[idx])
        if hi >= sl:
            daily_ret += w * (op - sl) / op
        elif lo <= tp:
            daily_ret += w * (op - tp) / op
        else:
            daily_ret += w * (op - cl) / op

    return daily_ret


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]
    D = ohlcv.shape[0]

    opens = ohlcv[:, tradeable_idx, O]
    closes = ohlcv[:, tradeable_idx, C]

    min_day = 21
    X_list, y_ret_list, y_dir_list = [], [], []
    for d in range(min_day, D):
        feats, _ = compute_features(ohlcv, tradeable_idx, all_tickers, d)
        intraday = (closes[d] - opens[d]) / opens[d].clamp(min=1e-8)
        X_list.append(feats)
        y_ret_list.append(intraday.clamp(-0.10, 0.10))
        y_dir_list.append((intraday > 0).float())

    X = torch.cat(X_list, dim=0)
    y_ret = torch.cat(y_ret_list, dim=0)
    y_dir = torch.cat(y_dir_list, dim=0)

    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - feat_mean) / feat_std

    y_ret_mean = y_ret.mean()
    y_ret_std = y_ret.std().clamp(min=1e-8)
    y_ret_norm = (y_ret - y_ret_mean) / y_ret_std

    n_features = X_norm.shape[1]

    # Train regression model (Huber)
    reg_model = train_model(X_norm, y_ret_norm, n_features,
                            nn.HuberLoss(delta=1.0), seed=42)

    # Train classification model (BCE)
    cls_model = train_model(X_norm, y_dir, n_features,
                            nn.BCEWithLogitsLoss(), seed=123)

    # Calibrate stops using combined score
    N = len(tradeable_idx)
    cal_start = max(min_day, D - 125)
    best_sharpe = -999
    best_sl, best_tp = 1.5, 2.5

    for sl_mult in [1.0, 1.5, 2.0, 2.5]:
        for tp_mult in [1.5, 2.0, 2.5, 3.0, 3.5]:
            daily_rets = []
            for d in range(cal_start, D):
                feats, atr = compute_features(ohlcv, tradeable_idx, all_tickers, d)
                feats_n = (feats - feat_mean) / feat_std
                with torch.no_grad():
                    reg_pred = reg_model(feats_n.to(dev)).cpu()
                    cls_logits = cls_model(feats_n.to(dev)).cpu()
                    cls_prob = torch.sigmoid(cls_logits)

                # Normalize reg predictions to [0,1] range for combination
                reg_norm = (reg_pred - reg_pred.min()) / (reg_pred.max() - reg_pred.min() + 1e-8)
                combined = 0.5 * reg_norm + 0.5 * cls_prob

                ret = simulate_day(
                    combined,
                    ohlcv[d, tradeable_idx, O],
                    ohlcv[d, tradeable_idx, H],
                    ohlcv[d, tradeable_idx, L],
                    ohlcv[d, tradeable_idx, C],
                    atr, N, sl_mult, tp_mult
                )
                daily_rets.append(ret)

            rets = torch.tensor(daily_rets)
            if rets.std() > 0:
                sharpe = rets.mean() / rets.std() * (252 ** 0.5)
            else:
                sharpe = 0.0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sl, best_tp = sl_mult, tp_mult

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "all_tickers": all_tickers,
        "reg_model": reg_model,
        "cls_model": cls_model,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "y_ret_mean": float(y_ret_mean),
        "y_ret_std": float(y_ret_std),
        "sl_mult": best_sl,
        "tp_mult": best_tp,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    all_tickers = strategy["all_tickers"]
    reg_model = strategy["reg_model"]
    cls_model = strategy["cls_model"]
    feat_mean = strategy["feat_mean"]
    feat_std = strategy["feat_std"]
    sl_mult = strategy["sl_mult"]
    tp_mult = strategy["tp_mult"]
    n_tickers = len(tickers)

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    feats, atr_pct = compute_features(
        ohlcv, tradeable_idx, all_tickers, day_idx
    )
    feats_norm = (feats - feat_mean) / feat_std

    with torch.no_grad():
        reg_pred = reg_model(feats_norm.to(dev)).cpu()
        cls_logits = cls_model(feats_norm.to(dev)).cpu()
        cls_prob = torch.sigmoid(cls_logits)

    # Normalize regression to [0,1] for combination with probability
    reg_norm = (reg_pred - reg_pred.min()) / (reg_pred.max() - reg_pred.min() + 1e-8)
    combined = 0.5 * reg_norm + 0.5 * cls_prob

    scored = []
    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        s = float(combined[i])
        ap = float(atr_pct[i])
        scored.append((i, s, op, ap))

    if len(scored) < 10:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    n_long = 10
    n_short = 10
    longs = scored[:n_long]
    shorts = scored[-n_short:]

    total = n_long + n_short
    w = 1.0 / total

    orders = []
    for (idx, _, op, ap) in longs:
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": w,
            "stop_loss": op * (1 - ap * sl_mult),
            "take_profit": op * (1 + ap * tp_mult),
        })

    for (idx, _, op, ap) in shorts:
        orders.append({
            "ticker": tickers[idx],
            "direction": "short",
            "weight": w,
            "stop_loss": op * (1 + ap * sl_mult),
            "take_profit": op * (1 - ap * tp_mult),
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
