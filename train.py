"""
Autoresearch-trader training script. Single-GPU, single-file.

Momentum strategy: go long on stocks with strong short-term momentum,
using ATR-based stops and asymmetric risk/reward.

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
    D = ohlcv.shape[0]

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]

    lookback = 10
    if day_idx < lookback + 1:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]

    # Compute ATR over last 14 days
    atr_len = min(14, day_idx - 1)
    highs = ohlcv[day_idx - atr_len:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - atr_len:day_idx, tradeable_idx, L]
    closes_atr = ohlcv[day_idx - atr_len - 1:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs - lows, (highs - closes_atr).abs()),
        (lows - closes_atr).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.10)

    # Short-term momentum: 5-day return
    closes = ohlcv[day_idx - lookback:day_idx, tradeable_idx, C]
    ret_5d = (closes[-1] / closes[-5].clamp(min=1e-8)) - 1.0
    ret_10d = (closes[-1] / closes[0].clamp(min=1e-8)) - 1.0

    # Gap: open vs yesterday's close
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]
    gap = (today_open / prev_close.clamp(min=1e-8)) - 1.0

    # Combined momentum signal
    signal = 0.5 * ret_5d + 0.3 * ret_10d + 0.2 * gap

    n_tickers = len(tickers)
    candidates = []

    for i in range(n_tickers):
        open_price = float(today_open[i])
        if open_price <= 0 or np.isnan(open_price):
            continue

        s = float(signal[i])
        atr_p = float(atr_pct[i])

        # Long momentum trades only — go with the trend
        if s > 0.005:
            stop_pct = atr_p * 0.8
            target_pct = atr_p * 2.0
            candidates.append({
                "ticker": tickers[i],
                "direction": "long",
                "signal_strength": s,
                "stop_loss": open_price * (1 - stop_pct),
                "take_profit": open_price * (1 + target_pct),
            })

    if not candidates:
        return []

    # Sort by signal strength, take top 3
    candidates.sort(key=lambda x: x["signal_strength"], reverse=True)
    candidates = candidates[:3]

    weight_each = 1.0 / len(candidates)

    orders = []
    for c in candidates:
        orders.append({
            "ticker": c["ticker"],
            "direction": c["direction"],
            "weight": weight_each,
            "stop_loss": c["stop_loss"],
            "take_profit": c["take_profit"],
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
