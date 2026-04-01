"""
v13-trend-aligned-fade: Only fade short-term moves when they're against the trend.

Hypothesis: a pullback within an uptrend (short-term dip in longer-term rally)
reverts more reliably than fading a move in the trend's direction. This uses
a 20-bar trend filter with 1-bar mean-reversion signal.

Economic rationale: trend pullbacks have institutional buyers/sellers as support.
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
    trend_lookback = 20
    if bar_idx < trend_lookback + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]

    opens = ohlcv[bar_idx, tidx, O].numpy()

    # Short-term signal: 1-bar open-to-close
    prev_open = ohlcv[bar_idx - 1, tidx, O].numpy()
    prev_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    short_mom = (prev_close - prev_open) / np.maximum(prev_open, 1e-8)

    # Longer-term trend: 20-bar return
    trend_start = ohlcv[bar_idx - trend_lookback, tidx, C].numpy()
    trend_end = ohlcv[bar_idx - 1, tidx, C].numpy()
    trend = (trend_end - trend_start) / np.maximum(trend_start, 1e-8)

    valid = np.where(
        (opens > 0) & ~np.isnan(opens) & ~np.isnan(short_mom) & ~np.isnan(trend)
    )[0]
    if len(valid) == 0:
        return []

    # Score: prefer stocks where short-term move is AGAINST the trend
    # i.e., trend is up but short-term dipped (or vice versa)
    scores = np.zeros(len(tickers))
    for i in valid:
        sm = short_mom[i]
        tr = trend[i]
        # Against trend: signs differ
        if sm * tr < 0:
            # Signal strength: magnitude of short-term move
            scores[i] = abs(sm)
        # else: short-term is with trend — skip (score stays 0)

    best = np.argmax(scores)
    if scores[best] < 0.0003:
        # No good against-trend signal; fall back to plain mean-reversion
        abs_mom = np.abs(short_mom[valid])
        best = valid[np.argmax(abs_mom)]
        if abs_mom[np.argmax(abs_mom)] < 0.0005:
            return []

    ticker = tickers[best]
    op = float(opens[best])
    mom = float(short_mom[best])
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
