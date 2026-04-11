#!/usr/bin/env python3
"""
train.py — v25-trend-xasset-only
================================
Hypothesis: the pullback oscillator (z/RSI/WR) over-restricts entries.
Drop it entirely — just require 24h trend direction and a 4-bar cross-
asset confirmation. Should fire much more frequently; test whether the
quality filter alone (cross-asset anomaly + trend alignment) holds up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

TREND_LB = 96
LB_SHORT = 4
MIN_MOVE = 0.0005


def _ret(s: pd.Series, ts_now, ts_prev) -> float:
    try:
        a = float(s.asof(ts_now))
        b = float(s.asof(ts_prev))
    except Exception:
        return float("nan")
    if np.isnan(a) or np.isnan(b) or b <= 0:
        return float("nan")
    return float(np.log(a / b))


def _crossasset_confirms(pair: pd.DataFrame, prices: dict, want_long: bool) -> bool:
    if len(pair) < LB_SHORT + 1:
        return False
    ts_now = pair.index[-1]
    ts_prev = pair.index[-1 - LB_SHORT]
    p_now = float(pair["close"].iloc[-1])
    p_prev = float(pair["close"].iloc[-1 - LB_SHORT])
    if p_prev <= 0:
        return False
    jr = float(np.log(p_now / p_prev))

    gold = prices.get("GC=F")
    dxy = prices.get("DX-Y.NYB")
    nk = prices.get("^N225")

    gr = _ret(gold["close"], ts_now, ts_prev) if gold is not None else float("nan")
    dr = _ret(dxy["close"], ts_now, ts_prev) if dxy is not None else float("nan")
    nr = _ret(nk["close"], ts_now, ts_prev) if nk is not None else float("nan")

    if want_long:
        if not np.isnan(gr) and jr < -MIN_MOVE and gr < -MIN_MOVE:
            return True
        if not np.isnan(dr) and jr < -MIN_MOVE and dr > MIN_MOVE:
            return True
        if not np.isnan(nr) and jr < -MIN_MOVE and nr > MIN_MOVE:
            return True
        return False
    else:
        if not np.isnan(gr) and jr > MIN_MOVE and gr > MIN_MOVE:
            return True
        if not np.isnan(dr) and jr > MIN_MOVE and dr < -MIN_MOVE:
            return True
        if not np.isnan(nr) and jr > MIN_MOVE and nr < -MIN_MOVE:
            return True
        return False


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < TREND_LB + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    closes = pair["close"].values.astype(float)
    last = float(closes[-1])
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    trend = last / prev - 1.0

    if trend > 0 and _crossasset_confirms(pair, prices, want_long=True):
        direction = 1
    elif trend < 0 and _crossasset_confirms(pair, prices, want_long=False):
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
