"""
v15-adaptive-atr: Use recent (5-bar) ATR instead of training average.

Hypothesis: recent volatility better reflects current market conditions
for stop calibration. If vol just spiked, wider stops; if calm, tighter.

Same stock selection as v9 (biggest 1-bar mover), same SL/TP multipliers.
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


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

    opens = ohlcv[bar_idx, tidx, O].numpy()

    # 1-bar signal
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (prev_close - prev_open) / np.maximum(prev_open, 1e-8)

    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) == 0:
        return []

    best = valid[np.argmax(np.abs(momentum[valid]))]
    mom = float(momentum[best])
    if abs(mom) < 0.0005:
        return []

    ticker = tickers[best]
    op = float(opens[best])

    # Adaptive ATR: last 5 bars
    recent_highs = ohlcv[bar_idx - 5:bar_idx, tidx, H].numpy()[:, best]
    recent_lows = ohlcv[bar_idx - 5:bar_idx, tidx, L].numpy()[:, best]
    recent_closes = ohlcv[bar_idx - 5:bar_idx, tidx, C].numpy()[:, best]
    recent_tr = (recent_highs - recent_lows) / np.maximum(recent_closes, 1e-8)
    atr = float(np.mean(recent_tr))

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
