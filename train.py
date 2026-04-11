#!/usr/bin/env python3
"""
train.py — v12-dxy-gated-dip
============================
Hypothesis: v11 confirmed dip-buying-in-trend has real edge (42.3% win,
best yet). The trend gate there was JPY's own 24h return — circular with
the signal noise we're fading. Decouple by using DXY's 24h direction as
an independent macro trend filter: fade JPY z only on the side aligned
with the DXY macro move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
TREND_LB = 96  # 24h DXY lookback


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    dxy = prices.get("DX-Y.NYB")
    if pair is None or dxy is None or len(pair) < Z_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].values.astype(float)
    last = float(closes[-1])

    window = closes[-Z_LB:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    z = (last - mu) / sd

    # DXY 24h trend
    dxy_idx = pair.index[-1]
    dxy_prev_idx = pair.index[-1 - TREND_LB] if len(pair) > TREND_LB else None
    if dxy_prev_idx is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    dxy_close = dxy["close"]
    try:
        dxy_now = float(dxy_close.asof(dxy_idx))
        dxy_prev = float(dxy_close.asof(dxy_prev_idx))
    except Exception:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if np.isnan(dxy_now) or np.isnan(dxy_prev) or dxy_prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    dxy_trend = dxy_now / dxy_prev - 1.0

    direction = 0
    if dxy_trend > 0 and z < -Z_ENTRY:
        direction = 1  # DXY up → buy JPY dip
    elif dxy_trend < 0 and z > Z_ENTRY:
        direction = -1  # DXY down → sell JPY rip

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
