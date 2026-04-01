"""
v9-tighter-sl: Even tighter SL (0.7x ATR), wider TP (4.0x ATR).

Hypothesis: pushing the asymmetry further improves risk-reward.
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    closes = ohlcv[:, tidx, C]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)
    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < 2:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    opens = ohlcv[bar_idx, tidx, O].numpy()
    # 1-bar: use open-to-close of previous bar as momentum signal
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (prev_close - prev_open) / np.maximum(prev_open, 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) == 0:
        return []

    # Pick the single most extreme mover
    abs_mom = np.abs(momentum[valid])
    best = valid[np.argmax(abs_mom)]

    ticker = tickers[best]
    op = float(opens[best])
    mom = float(momentum[best])
    atr = float(avg_atr[best])

    if abs(mom) < 0.0005:
        return []

    direction = "short" if mom > 0 else "long"

    sl_dist = max(atr * 0.7, 0.001) * op
    tp_dist = max(atr * 4.0, 0.005) * op

    if direction == "long":
        sl = op - sl_dist
        tp = op + tp_dist
    else:
        sl = op + sl_dist
        tp = op - tp_dist

    return [{
        "ticker": ticker,
        "direction": direction,
        "weight": 1.0,
        "stop_loss": sl,
        "take_profit": tp,
    }]


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print("\n---")
    print(f"sharpe_ratio:     {results.get('sharpe_ratio', 0):.4f}")
    print(f"total_return:     {results.get('total_return', 0)*100:.2f}%")
    print(f"max_drawdown:     {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"win_rate:         {results.get('win_rate', 0)*100:.1f}%")
    print(f"avg_trades/bar:   {results.get('avg_daily_trades', 0):.2f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
