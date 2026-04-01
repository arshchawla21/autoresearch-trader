"""
Autoresearch-trader training script. Single-GPU, single-file.

Hold-to-close: long SPY every day with very wide stops (never triggered).
Captures the daily equity premium. Minimal strategy, maximum simplicity.

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
    tickers = train_data["tradeable_tickers"]
    tradeable_idx = train_data["tradeable_indices"]
    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "spy_idx": tickers.index("SPY"),
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    spy_i = strategy["spy_idx"]

    if day_idx < 2:
        return []

    open_price = float(ohlcv[day_idx, tradeable_idx[spy_i], O])
    if open_price <= 0 or np.isnan(open_price):
        return []

    # Set stops extremely wide — effectively hold to close
    return [{
        "ticker": "SPY",
        "direction": "long",
        "weight": 1.0,
        "stop_loss": open_price * 0.90,    # 10% below — never hit intraday
        "take_profit": open_price * 1.10,   # 10% above — never hit intraday
    }]


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
