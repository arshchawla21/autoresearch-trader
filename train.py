#!/usr/bin/env python3
"""
train.py — Strategy v7: EMA Trend-Follow
==========================================
Hypothesis: if mean reversion is the right frame for USD/JPY 15m, a pure
trend-follow strategy should fail symmetrically. Test: fast EMA(8) crosses
above slow EMA(21) → long, reverse → short. This is a classic trend
signal. If it loses, it reinforces that the pair's 15m structure favours
fading extremes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

FAST = 8
SLOW = 21


def _ema(x: np.ndarray, n: int) -> float:
    alpha = 2.0 / (n + 1)
    e = float(x[0])
    for v in x[1:]:
        e = alpha * float(v) + (1 - alpha) * e
    return e


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < SLOW + 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].dropna().values
    if len(closes) < SLOW + 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    fast_now = _ema(closes[-FAST:], FAST)
    slow_now = _ema(closes[-SLOW:], SLOW)
    fast_prev = _ema(closes[-FAST - 1:-1], FAST)
    slow_prev = _ema(closes[-SLOW - 1:-1], SLOW)

    # Current state + cross detection
    if fast_now > slow_now and fast_prev <= slow_prev:
        direction = 1
    elif fast_now < slow_now and fast_prev >= slow_prev:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
