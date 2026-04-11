#!/usr/bin/env python3
"""
train.py — v11-trend-aligned-mr
===============================
Hypothesis: classic "buy dips in an uptrend, sell rips in a downtrend".
Gate a 20-bar z-score MR signal with the sign of the 24h (96-bar) USD/JPY
return. Only take long MR when the daily trend is up, only take short MR
when the daily trend is down. TP/SL 15/10 asymmetric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
TREND_LB = 96  # 24h


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < TREND_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].values.astype(float)
    last = float(closes[-1])

    # 20-bar z
    window = closes[-Z_LB:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    z = (last - mu) / sd

    # 24h trend
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    trend = last / prev - 1.0

    # Fade extreme z in direction of trend
    direction = 0
    if trend > 0 and z < -Z_ENTRY:
        direction = 1      # buy the dip in uptrend
    elif trend < 0 and z > Z_ENTRY:
        direction = -1     # sell the rip in downtrend

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
