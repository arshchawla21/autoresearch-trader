"""
v30-gap-fade: Fade the open-to-previous-close gap.

Different signal from v23: instead of fading the 5-bar return (close-to-close),
fade the GAP between current bar's open and previous bar's close.
Large gaps tend to fill within the bar.
"""

import time
import numpy as np
import torch
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

SL_PCT = 0.005
TP_PCT = 0.005
N_LONG = 2
N_SHORT = 2


def build_strategy(train_data):
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 2:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    prev_close = ohlcv_np[bar_idx - 1, tidx_np, C]

    # Gap: how much the stock jumped from last close to this open
    gap = (opens - prev_close) / np.maximum(np.abs(prev_close), 1e-8)

    valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(gap)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < N_LONG + N_SHORT:
        return []

    # Fade the gap: short stocks that gapped up, long stocks that gapped down
    valid_gaps = gap[valid_idx]
    sorted_indices = np.argsort(valid_gaps)

    long_picks = valid_idx[sorted_indices[:N_LONG]]    # most negative gaps
    short_picks = valid_idx[sorted_indices[-N_SHORT:]]  # most positive gaps

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
