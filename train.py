#!/usr/bin/env python3
"""
train.py — v3-bbands-adx-mr
===========================
Hypothesis: USD/JPY mean-reverts when the market is range-bound (low ADX)
and trends when ADX is high. Gate a Bollinger-band reversion with a low-ADX
regime filter to restrict trades to exactly the regime where reversion has
positive expectancy. Pure price + structure, no ML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 12.0
SL_PIPS = 10.0

BB_LOOKBACK = 20
BB_STD = 2.0
ADX_LOOKBACK = 14
ADX_MAX = 22.0  # only trade if ADX below this → range regime
WINDOW = 80


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> float:
    if len(high) < 2 * n + 1:
        return float("nan")
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ]
    )
    # Wilder smoothing via simple EMA approximation (rolling mean is fine for ADX proxy)
    def _sma(x: np.ndarray, w: int) -> np.ndarray:
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[w:] - c[:-w]) / w

    if len(tr) < n:
        return float("nan")
    atr = _sma(tr, n)
    pdi = 100.0 * _sma(plus_dm, n) / (atr + 1e-12)
    mdi = 100.0 * _sma(minus_dm, n) / (atr + 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    if len(dx) < n:
        return float("nan")
    adx = _sma(dx, n)
    return float(adx[-1])


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < WINDOW:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    tail = pair.iloc[-WINDOW:]
    closes = tail["close"].values.astype(float)
    highs = tail["high"].values.astype(float)
    lows = tail["low"].values.astype(float)

    adx_val = _adx(highs, lows, closes, ADX_LOOKBACK)
    if np.isnan(adx_val) or adx_val >= ADX_MAX:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    bb_slice = closes[-BB_LOOKBACK:]
    mu = float(bb_slice.mean())
    sd = float(bb_slice.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    upper = mu + BB_STD * sd
    lower = mu - BB_STD * sd
    last = float(closes[-1])

    if last < lower:
        direction = 1
    elif last > upper:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
