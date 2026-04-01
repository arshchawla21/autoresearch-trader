"""
v32b-0.8pct: v23 base with 0.8% SL/TP (highest return configuration).

Sweet spot between tight (0.5%) and wide (1%): rarely triggers within 1 bar
but the MR direction call gives consistent open-to-close edge.
"""

import time
import numpy as np
import torch
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

SL_PCT = 0.008   # 0.8% stop loss (sweet spot: highest return)
TP_PCT = 0.008   # 0.8% take profit
N_LONG = 2
N_SHORT = 2
LOOKBACK = 5


def build_strategy(train_data):
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < LOOKBACK + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    ret = (closes[-1] - closes[-1 - LOOKBACK]) / np.maximum(np.abs(closes[-1 - LOOKBACK]), 1e-8)

    valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(ret)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < N_LONG + N_SHORT:
        return []

    valid_rets = ret[valid_idx]
    sorted_indices = np.argsort(valid_rets)

    long_picks = valid_idx[sorted_indices[:N_LONG]]
    short_picks = valid_idx[sorted_indices[-N_SHORT:]]

    w = 1.0 / (N_LONG + N_SHORT)

    orders = []
    for idx in long_picks:
        op = float(opens[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "long",
            "weight": w,
            "stop_loss": op * (1 - SL_PCT),
            "take_profit": op * (1 + TP_PCT),
        })

    for idx in short_picks:
        op = float(opens[idx])
        orders.append({
            "ticker": tickers[idx],
            "direction": "short",
            "weight": w,
            "stop_loss": op * (1 + SL_PCT),
            "take_profit": op * (1 - TP_PCT),
        })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% "
          f"dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% "
          f"trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
