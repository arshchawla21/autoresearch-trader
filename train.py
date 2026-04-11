#!/usr/bin/env python3
"""
train.py — v19-wick-rejection
=============================
Hypothesis: long-wick candles mark price rejection — a long lower wick
means buyers defended that level; a long upper wick means sellers did.
Fade the wick: long on bullish lower-wick rejection, short on bearish
upper-wick rejection. Pure single-bar microstructure, no learning.
Required: meaningful bar range (>5 pips), wick > 60% of total range.
"""

from __future__ import annotations

import pandas as pd

TP_PIPS = 12.0
SL_PIPS = 10.0

WICK_FRAC = 0.60
MIN_RANGE_PIPS = 5.0


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    bar = pair.iloc[-1]
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    rng = h - l
    if rng < MIN_RANGE_PIPS * 0.01:  # 0.05 yen = 5 pips
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    body_top = max(o, c)
    body_bot = min(o, c)
    upper_wick = h - body_top
    lower_wick = body_bot - l

    if lower_wick / rng >= WICK_FRAC:
        direction = 1
    elif upper_wick / rng >= WICK_FRAC:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
