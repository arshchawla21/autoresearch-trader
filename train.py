"""
v26-multi-lookback: Cross-sectional L/S MR with blended multi-lookback signal.

Instead of a single lookback, blend 1-bar and 5-bar returns for ranking.
Short-term (1-bar) captures immediate overreaction.
Medium-term (5-bar) captures sustained drift that's likely to revert.
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
    if bar_idx < 6:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    # Blend 1-bar and 5-bar returns (equal weight)
    ret1 = (closes[-1] - closes[-2]) / np.maximum(np.abs(closes[-2]), 1e-8)
    ret5 = (closes[-1] - closes[-6]) / np.maximum(np.abs(closes[-6]), 1e-8)

    # Normalize each to z-scores before blending (different scales)
    def zscore(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / max(s, 1e-8)

    signal = zscore(ret1) + zscore(ret5)

    valid_mask = (opens > 0) & ~np.isnan(opens) & ~np.isnan(signal)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < N_LONG + N_SHORT:
        return []

    valid_signal = signal[valid_idx]
    sorted_indices = np.argsort(valid_signal)

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
