"""
Autoresearch-trader training script. Single-GPU, single-file.

Regime-adaptive momentum: tilts long/short balance based on SPY trend.
Uptrend = long-only top momentum. Downtrend = short-only bottom momentum.
Sideways = balanced long/short. VIX scaling for extreme volatility.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import numpy as np
import torch

from prepare import O, H, L, C, evaluate

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Strategy API: build_strategy + generate_orders
# ---------------------------------------------------------------------------

def build_strategy(train_data):
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]

    # Find SPY and VIX indices
    spy_all_idx = all_tickers.index("SPY")
    vix_idx = None
    for j, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_idx = j
            break

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "spy_all_idx": spy_all_idx,
        "vix_idx": vix_idx,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    spy_all_idx = strategy["spy_all_idx"]
    vix_idx = strategy["vix_idx"]
    n_tickers = len(tickers)

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]

    # ATR over last 14 days
    highs_14 = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows_14 = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_14 = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs_14 - lows_14, (highs_14 - closes_14).abs()),
        (lows_14 - closes_14).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.15)

    # Medium-term momentum (10d + 20d)
    close_10 = ohlcv[day_idx - 10, tradeable_idx, C]
    close_20 = ohlcv[day_idx - 20, tradeable_idx, C]
    mom_10d = (prev_close - close_10) / close_10.clamp(min=1e-8)
    mom_20d = (prev_close - close_20) / close_20.clamp(min=1e-8)
    trend = 0.6 * mom_10d + 0.4 * mom_20d

    # Short-term pullback: 3-day return (negative = pullback for longs)
    close_3 = ohlcv[day_idx - 3, tradeable_idx, C]
    ret_3d = (prev_close - close_3) / close_3.clamp(min=1e-8)

    # SPY regime
    spy_close_now = float(ohlcv[day_idx - 1, spy_all_idx, C])
    spy_close_20 = float(ohlcv[day_idx - 20, spy_all_idx, C])
    spy_mom = (spy_close_now - spy_close_20) / spy_close_20

    # VIX regime
    vix = float(ohlcv[day_idx - 1, vix_idx, C]) if vix_idx is not None else 20.0
    if vix > 35:
        vol_scale = 0.5
    elif vix > 28:
        vol_scale = 0.7
    else:
        vol_scale = 1.0

    # Build candidate lists with pullback filter
    long_candidates = []
    short_candidates = []

    for i in range(n_tickers):
        op = float(today_open[i])
        if op <= 0 or np.isnan(op):
            continue
        t = float(trend[i])
        r3 = float(ret_3d[i])
        ap = float(atr_pct[i])

        # Long: positive trend + recent pullback (buy the dip)
        if t > 0.01 and r3 < 0.0:
            score = t * (1 + abs(r3) * 5)  # bigger dip in uptrend = better
            long_candidates.append((i, score, op, ap))

        # Short: negative trend + recent bounce (sell the rally)
        if t < -0.01 and r3 > 0.0:
            score = abs(t) * (1 + abs(r3) * 5)
            short_candidates.append((i, score, op, ap))

    long_candidates.sort(key=lambda x: x[1], reverse=True)
    short_candidates.sort(key=lambda x: x[1], reverse=True)

    orders = []
    stop_m = 1.5
    target_m = 2.5

    if spy_mom > 0.02:
        # Uptrend: long-only from pullback candidates
        picks = long_candidates[:4]
        if not picks:
            # Fallback: top momentum even without pullback
            ranked = trend.argsort(descending=True)
            for j in range(min(4, n_tickers)):
                idx = int(ranked[j])
                op = float(today_open[idx])
                ap = float(atr_pct[idx])
                if op > 0 and not np.isnan(op):
                    picks.append((idx, 0, op, ap))
        w = vol_scale / max(len(picks), 1)
        for (idx, _, op, ap) in picks:
            orders.append({
                "ticker": tickers[idx],
                "direction": "long",
                "weight": w,
                "stop_loss": op * (1 - ap * stop_m),
                "take_profit": op * (1 + ap * target_m),
            })

    elif spy_mom < -0.02:
        # Downtrend: short-only from rally candidates
        picks = short_candidates[:4]
        if not picks:
            ranked = trend.argsort()
            for j in range(min(4, n_tickers)):
                idx = int(ranked[j])
                op = float(today_open[idx])
                ap = float(atr_pct[idx])
                if op > 0 and not np.isnan(op):
                    picks.append((idx, 0, op, ap))
        w = vol_scale / max(len(picks), 1)
        for (idx, _, op, ap) in picks:
            orders.append({
                "ticker": tickers[idx],
                "direction": "short",
                "weight": w,
                "stop_loss": op * (1 + ap * stop_m),
                "take_profit": op * (1 - ap * target_m),
            })

    else:
        # Sideways: balanced from pullback candidates
        longs = long_candidates[:2]
        shorts = short_candidates[:2]
        n_total = len(longs) + len(shorts)
        if n_total == 0:
            return []
        w = vol_scale / n_total
        for (idx, _, op, ap) in longs:
            orders.append({
                "ticker": tickers[idx],
                "direction": "long",
                "weight": w,
                "stop_loss": op * (1 - ap * stop_m),
                "take_profit": op * (1 + ap * target_m),
            })
        for (idx, _, op, ap) in shorts:
            orders.append({
                "ticker": tickers[idx],
                "direction": "short",
                "weight": w,
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
