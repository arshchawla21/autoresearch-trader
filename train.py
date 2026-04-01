"""
v29-dual-tf: Dual-timeframe cross-sectional L/S MR.

Run two independent strategies simultaneously:
- lb=5, top/bottom 2 stocks (half weight)
- lb=10, top/bottom 2 stocks (half weight, different picks preferred)

Diversification across timeframes should reduce variance.
"""

import time
import numpy as np
import torch
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

SL_PCT = 0.005
TP_PCT = 0.005


def build_strategy(train_data):
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 11:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    valid_mask = (opens > 0) & ~np.isnan(opens)

    orders = []
    used_tickers = set()

    for lb, n_pos in [(5, 2), (10, 2)]:
        ret = (closes[-1] - closes[-1 - lb]) / np.maximum(np.abs(closes[-1 - lb]), 1e-8)
        combined_mask = valid_mask & ~np.isnan(ret)
        valid_idx = np.where(combined_mask)[0]

        if len(valid_idx) < 2 * n_pos:
            continue

        valid_rets = ret[valid_idx]
        sorted_indices = np.argsort(valid_rets)

        long_picks = valid_idx[sorted_indices[:n_pos]]
        short_picks = valid_idx[sorted_indices[-n_pos:]]

        # Half weight for each timeframe
        w = 0.5 / (2 * n_pos)

        for idx in long_picks:
            ticker = tickers[idx]
            if ticker in used_tickers:
                continue
            used_tickers.add(ticker)
            op = float(opens[idx])
            orders.append({
                "ticker": ticker,
                "direction": "long",
                "weight": w,
                "stop_loss": op * (1 - SL_PCT),
                "take_profit": op * (1 + TP_PCT),
            })

        for idx in short_picks:
            ticker = tickers[idx]
            if ticker in used_tickers:
                continue
            used_tickers.add(ticker)
            op = float(opens[idx])
            orders.append({
                "ticker": ticker,
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
