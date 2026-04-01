"""
Autoresearch-trader training script. Single-GPU, single-file.

Gap-fade strategy: buy stocks that gap down at open, sell stocks that gap up.
Intraday mean-reversion on overnight gaps is a well-documented anomaly.

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

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]

    if day_idx < 15:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]

    # Compute ATR over last 14 days
    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_atr = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs - lows, (highs - closes_atr).abs()),
        (lows - closes_atr).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.10)

    # Gap = (open - prev_close) / prev_close
    gap = (today_open - prev_close) / prev_close.clamp(min=1e-8)

    n_tickers = len(tickers)
    candidates = []

    for i in range(n_tickers):
        open_price = float(today_open[i])
        if open_price <= 0 or np.isnan(open_price):
            continue

        g = float(gap[i])
        atr_p = float(atr_pct[i])

        # Gap down > 0.3% -> buy (expect intraday reversion)
        if g < -0.003:
            stop_pct = atr_p * 1.0
            target_pct = abs(g) * 0.5  # target partial gap fill
            target_pct = max(target_pct, atr_p * 0.5)
            candidates.append({
                "ticker": tickers[i],
                "direction": "long",
                "signal_strength": abs(g),
                "stop_loss": open_price * (1 - stop_pct),
                "take_profit": open_price * (1 + target_pct),
            })
        # Gap up > 0.3% -> short (expect intraday reversion)
        elif g > 0.003:
            stop_pct = atr_p * 1.0
            target_pct = abs(g) * 0.5
            target_pct = max(target_pct, atr_p * 0.5)
            candidates.append({
                "ticker": tickers[i],
                "direction": "short",
                "signal_strength": abs(g),
                "stop_loss": open_price * (1 + stop_pct),
                "take_profit": open_price * (1 - target_pct),
            })

    if not candidates:
        return []

    # Sort by gap size (larger gaps = stronger signal), take top 5
    candidates.sort(key=lambda x: x["signal_strength"], reverse=True)
    candidates = candidates[:5]

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
