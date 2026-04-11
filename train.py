#!/usr/bin/env python3
"""
train.py — v9-london-orb
========================
Hypothesis: USD/JPY has no short-horizon MR edge (v8 confirmed). Try
structural session-based alpha: trade breakouts of the 4-hour pre-London
range (03:00–06:45 UTC) during the London-open window (07:00–09:00 UTC).
Classic ORB — institutional participation change at session handoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

# Pre-London range build window (UTC hours)
RANGE_START_H = 3
RANGE_END_H = 7      # range = bars with hour in [3, 7)
# Trade window
ENTRY_START_H = 7
ENTRY_END_H = 10     # enter only while hour in [7, 10)


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < 40:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    ts = pair.index[-1]
    utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts
    hour = utc.hour
    if hour < ENTRY_START_H or hour >= ENTRY_END_H:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    # Look at today's pre-London window
    today = utc.normalize()
    # Get bars from today with hour in [3, 7)
    idx = pair.index
    if idx.tzinfo is None:
        today_local = today.tz_localize(None)
    else:
        today_local = today
    # Slice last ~96 bars (one day) for perf
    recent = pair.iloc[-96:]
    ridx = recent.index
    ridx_utc = ridx.tz_convert("UTC") if ridx.tzinfo is not None else ridx

    mask = (
        (ridx_utc.normalize() == today)
        & (ridx_utc.hour >= RANGE_START_H)
        & (ridx_utc.hour < RANGE_END_H)
    )
    window = recent[mask]
    if len(window) < 12:  # need most of the 16 bars
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    hi = float(window["high"].max())
    lo = float(window["low"].min())
    last = float(recent["close"].iloc[-1])

    if last > hi:
        direction = 1
    elif last < lo:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
