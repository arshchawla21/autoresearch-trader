#!/usr/bin/env python3
"""
train.py — v13-strong-trend-dip
===============================
Hypothesis: v11 (42.3% win) worked because JPY's own 24h trend is
persistent; v12 (40% win) showed DXY trend isn't. Keep the self-trend
gate but add a trend-strength filter — weak trends don't persist, so
skip them. Require |24h trend| > 0.3% before fading short-term z-score
extremes in the trend direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
TREND_LB = 96
TREND_MIN = 0.003  # 0.3% minimum trend magnitude


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < TREND_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].values.astype(float)
    last = float(closes[-1])

    window = closes[-Z_LB:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    z = (last - mu) / sd

    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    trend = last / prev - 1.0

    direction = 0
    if trend > TREND_MIN and z < -Z_ENTRY:
        direction = 1
    elif trend < -TREND_MIN and z > Z_ENTRY:
        direction = -1
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
