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

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]

    # ATR over last 14 days
    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_atr = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs - lows, (highs - closes_atr).abs()),
        (lows - closes_atr).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.15)

    # Cross-sectional momentum signal (10d + 20d blend)
    close_10 = ohlcv[day_idx - 10, tradeable_idx, C]
    close_20 = ohlcv[day_idx - 20, tradeable_idx, C]
    mom_10d = (prev_close - close_10) / close_10.clamp(min=1e-8)
    mom_20d = (prev_close - close_20) / close_20.clamp(min=1e-8)
    signal = 0.6 * mom_10d + 0.4 * mom_20d

    # Rank stocks
    ranks = signal.argsort(descending=True)

    # SPY regime: 20-day return of SPY
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

    # Regime-adaptive position allocation
    n_positions = 4
    orders = []

    if spy_mom > 0.02:
        # Uptrend: long-only top momentum
        top_n = ranks[:n_positions]
        weight_each = vol_scale / n_positions
        for i in range(n_positions):
            idx = int(top_n[i])
            op = float(today_open[idx])
            if op <= 0 or np.isnan(op):
                continue
            ap = float(atr_pct[idx])
            orders.append({
                "ticker": tickers[idx],
                "direction": "long",
                "weight": weight_each,
                "stop_loss": op * (1 - ap * 2.0),
                "take_profit": op * (1 + ap * 2.0),
            })
    elif spy_mom < -0.02:
        # Downtrend: short-only bottom momentum
        bot_n = ranks[-n_positions:]
        weight_each = vol_scale / n_positions
        for i in range(n_positions):
            idx = int(bot_n[i])
            op = float(today_open[idx])
            if op <= 0 or np.isnan(op):
                continue
            ap = float(atr_pct[idx])
            orders.append({
                "ticker": tickers[idx],
                "direction": "short",
                "weight": weight_each,
                "stop_loss": op * (1 + ap * 2.0),
                "take_profit": op * (1 - ap * 2.0),
            })
    else:
        # Sideways: balanced long/short (market-neutral)
        n_each = n_positions // 2
        top_n = ranks[:n_each]
        bot_n = ranks[-n_each:]
        weight_each = vol_scale / (2 * n_each)

        for i in range(n_each):
            idx = int(top_n[i])
            op = float(today_open[idx])
            if op <= 0 or np.isnan(op):
                continue
            ap = float(atr_pct[idx])
            orders.append({
                "ticker": tickers[idx],
                "direction": "long",
                "weight": weight_each,
                "stop_loss": op * (1 - ap * 2.0),
                "take_profit": op * (1 + ap * 2.0),
            })

        for i in range(n_each):
            idx = int(bot_n[i])
            op = float(today_open[idx])
            if op <= 0 or np.isnan(op):
                continue
            ap = float(atr_pct[idx])
            orders.append({
                "ticker": tickers[idx],
                "direction": "short",
                "weight": weight_each,
                "stop_loss": op * (1 + ap * 2.0),
                "take_profit": op * (1 - ap * 2.0),
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
