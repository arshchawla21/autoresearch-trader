#!/usr/bin/env python3
"""
train.py — v8-zscore-mr-control
===============================
Hypothesis: pure 20-bar price z-score mean reversion on USD/JPY is the
simplest MR statement possible. apr12 v4 on 30-day fake data scored +4.8;
v7–v1 on real data all show momentum fades. This is the control test for
whether classic level-z MR survives 170 days + 0.8p spread with a TP:SL
of 15/10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.5


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < Z_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    closes = pair["close"].values[-Z_LB:].astype(float)
    mu = float(closes.mean())
    sd = float(closes.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    z = (float(pair["close"].iloc[-1]) - mu) / sd
    if z > Z_ENTRY:
        direction = -1
    elif z < -Z_ENTRY:
        direction = 1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
