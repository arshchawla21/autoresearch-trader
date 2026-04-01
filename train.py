"""
v2-momentum: Short-term momentum strategy with ATR-based stops.

Hypothesis: stocks trending in the last few bars will continue for the next bar.
Use 4-bar momentum to pick direction, ATR for stop/TP calibration.
Trade the top 3 momentum stocks each bar, equal weight.
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


def build_strategy(train_data):
    """Compute ATR stats from training data for stop calibration."""
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()

    # Compute average ATR per stock (as fraction of price)
    closes = ohlcv[:, tidx, C]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]

    # True range as pct of close
    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)  # per stock

    return {
        "tickers": train_data["tradeable_tickers"],
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
    }


def generate_orders(strategy, data, bar_idx):
    """Trade top momentum stocks with ATR-based stops."""
    lookback = 4
    if bar_idx < lookback + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    # Current opens
    opens = ohlcv[bar_idx, tidx, O].numpy()

    # Momentum: return over last `lookback` bars
    past_close = ohlcv[bar_idx - lookback, tidx, C].numpy()
    current_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    momentum = (current_close - past_close) / np.maximum(past_close, 1e-8)

    # Filter valid stocks
    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(momentum))[0]
    if len(valid) == 0:
        return []

    # Sort by absolute momentum, pick top 3
    abs_mom = np.abs(momentum[valid])
    top_k = min(3, len(valid))
    top_indices = valid[np.argsort(-abs_mom)[:top_k]]

    orders = []
    weight = 1.0 / top_k

    for idx in top_indices:
        ticker = tickers[idx]
        op = float(opens[idx])
        mom = float(momentum[idx])
        atr = float(avg_atr[idx])

        if abs(mom) < 0.001:  # skip if no meaningful momentum
            continue

        direction = "long" if mom > 0 else "short"

        # ATR-based stops: 1.5x ATR stop, 2x ATR take-profit
        sl_dist = max(atr * 1.5, 0.002) * op
        tp_dist = max(atr * 2.0, 0.003) * op

        if direction == "long":
            sl = op - sl_dist
            tp = op + tp_dist
        else:
            sl = op + sl_dist
            tp = op - tp_dist

        orders.append({
            "ticker": ticker,
            "direction": direction,
            "weight": weight,
            "stop_loss": sl,
            "take_profit": tp,
        })

    return orders


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
