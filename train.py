#!/usr/bin/env python3
"""
train.py — v29-coint-spread-momo
================================
Hypothesis: v28 showed JPY-DXY log spread z-score fires 9.2/day but the
mean-reversion direction is wrong (44.4% win at 49.1% breakeven). The
signal is real but the pair residual TRENDS rather than reverts —
USD-idiosyncratic flow pushes the spread further once it extends. Ride
the spread breakout instead: long JPY when spread is already high, short
when low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 12.0
SL_PIPS = 10.0

SPREAD_LB = 60
Z_ENTRY = 1.6
MIN_HIST = 120

_CACHE: dict = {}


def _spread_series(pair: pd.DataFrame, dxy: pd.DataFrame) -> pd.Series | None:
    if pair is None or dxy is None:
        return None
    # Align DXY to pair bars via asof; use only tail for speed.
    n_tail = MIN_HIST + SPREAD_LB + 20
    p_tail = pair.iloc[-n_tail:]
    # For each pair timestamp, fetch latest DXY close at or before it.
    d_close = dxy["close"].dropna()
    if len(d_close) < SPREAD_LB + 10:
        return None
    try:
        d_aligned = d_close.reindex(p_tail.index, method="ffill")
    except Exception:
        return None
    if d_aligned.isna().all():
        return None
    p_close = p_tail["close"].astype(float)
    valid = (~d_aligned.isna()) & (p_close > 0) & (d_aligned > 0)
    if valid.sum() < SPREAD_LB + 5:
        return None
    log_pair = np.log(p_close[valid])
    log_dxy = np.log(d_aligned[valid].astype(float))
    return log_pair - log_dxy


def _spread_signal(spread: pd.Series) -> tuple[int, float]:
    if spread is None or len(spread) < SPREAD_LB + 1:
        return 0, 0.0
    win = spread.iloc[-SPREAD_LB:].values
    mu = float(win.mean())
    sd = float(win.std(ddof=1))
    if sd <= 0:
        return 0, 0.0
    last = float(spread.iloc[-1])
    z = (last - mu) / sd
    if z > Z_ENTRY:
        return 1, z  # spread trending up → ride JPY long
    if z < -Z_ENTRY:
        return -1, z  # spread trending down → ride JPY short
    return 0, z


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    dxy = prices.get("DX-Y.NYB")
    if pair is None or dxy is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if len(pair) < MIN_HIST:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    spread = _spread_series(pair, dxy)
    if spread is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    direction, _z = _spread_signal(spread)
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
