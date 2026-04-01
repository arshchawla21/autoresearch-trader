"""
v5-rsi-meanrev: RSI-based mean-reversion. Trade only at extremes.

Hypothesis: RSI identifies overbought/oversold better than raw momentum.
Only trade when RSI < 30 (buy) or RSI > 70 (sell). Wider TP for bigger wins.
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()


def compute_rsi(closes, period=14):
    """Compute RSI for each stock. closes shape: (T, N)"""
    T, N = closes.shape
    rsi = np.full((T, N), 50.0)  # neutral default

    for t in range(period + 1, T):
        window = closes[t - period:t + 1]
        deltas = np.diff(window, axis=0)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)

        avg_gain = np.mean(gains, axis=0)
        avg_loss = np.mean(losses, axis=0)

        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi[t] = 100 - 100 / (1 + rs)

    return rsi


def build_strategy(train_data):
    """Compute ATR and precompute RSI on training data."""
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
    """Trade based on RSI extremes."""
    rsi_period = 14
    if bar_idx < rsi_period + 2:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    # Compute RSI on recent history
    closes = ohlcv[bar_idx - rsi_period - 1:bar_idx, tidx, C].numpy()
    deltas = np.diff(closes, axis=0)
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = np.mean(gains, axis=0)
    avg_loss = np.mean(losses, axis=0)
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - 100 / (1 + rs)

    opens = ohlcv[bar_idx, tidx, O].numpy()
    valid = np.where((opens > 0) & ~np.isnan(opens))[0]
    if len(valid) == 0:
        return []

    orders = []
    candidates = []

    for idx in valid:
        r = float(rsi[idx])
        if r < 30:
            candidates.append((idx, "long", 30 - r))  # more extreme = stronger signal
        elif r > 70:
            candidates.append((idx, "short", r - 70))

    if not candidates:
        return []

    # Sort by signal strength, take top 3
    candidates.sort(key=lambda x: -x[2])
    top = candidates[:3]
    weight = 1.0 / len(top)

    for idx, direction, strength in top:
        ticker = tickers[idx]
        op = float(opens[idx])
        atr = float(avg_atr[idx])

        # Wider TP: 2.5x ATR, SL: 1.5x ATR
        sl_dist = max(atr * 1.5, 0.002) * op
        tp_dist = max(atr * 2.5, 0.004) * op

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
