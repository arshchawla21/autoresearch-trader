"""
v16-multi-bar-signal: Combine 1-bar and 2-bar momentum signals.

Hypothesis: averaging the 1-bar and 2-bar mean-reversion signals is more
robust — it smooths noise while still being responsive. The stock with
the strongest combined signal gets traded.

Uses training-period ATR with v9 stops (0.7x/4.0x).
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
    if bar_idx < 3:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    opens = ohlcv[bar_idx, tidx, O].numpy()

    # 1-bar signal: open-to-close of previous bar
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    sig1 = (prev_close - prev_open) / np.maximum(prev_open, 1e-8)

    # 2-bar signal: close 2 bars ago to close 1 bar ago
    close_2ago = ohlcv[bar_idx - 2, tidx, C].numpy()
    sig2 = (prev_close - close_2ago) / np.maximum(close_2ago, 1e-8)

    # Combined signal (average)
    combined = (sig1 + sig2) / 2.0

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(combined))[0]
    if len(valid) == 0:
        return []

    best = valid[np.argmax(np.abs(combined[valid]))]
    mom = float(combined[best])
    if abs(mom) < 0.0005:
        return []

    ticker = tickers[best]
    op = float(opens[best])
    atr = float(avg_atr[best])

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
