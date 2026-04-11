#!/usr/bin/env python3
"""
train.py — v6-shock-ride
========================
Hypothesis (flip of v5): 2-bar shocks don't revert, they continue. v5 fading
>2σ shocks produced a 39% win rate — the mirror image is the edge. Ride
the shock: when the 2-bar z of log-returns exceeds 2σ, trade *with* the
direction, TP=15 / SL=10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

SHOCK_LB = 2          # 30-minute shock window
VOL_LB = 60           # ~15h rolling stdev
Z_ENTRY = 2.0


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < VOL_LB + SHOCK_LB + 2:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].values[-(VOL_LB + SHOCK_LB + 5):].astype(float)
    # 2-bar log returns
    ret2 = np.log(closes[SHOCK_LB:] / closes[:-SHOCK_LB])
    if len(ret2) < VOL_LB:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    recent = ret2[-VOL_LB:]
    sd = float(recent.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    last = float(ret2[-1])
    z = last / sd

    if z > Z_ENTRY:
        direction = 1
    elif z < -Z_ENTRY:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
