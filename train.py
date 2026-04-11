#!/usr/bin/env python3
"""
train.py — v22-v21-plus-tnx
===========================
Hypothesis: v21 was the champion (+0.50%, 43.83% win). Add a fourth
cross-asset confirmation path: TNX (US 10Y yield futures proxy) — yields
and USD/JPY should be positively correlated (higher yields → stronger
USD → JPY up). JPY-TNX anti-correlation means one lagged the other; fade
JPY's move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
TREND_LB = 96
LB_SHORT = 4
MIN_MOVE = 0.0005


def _v11_signal(pair: pd.DataFrame) -> int:
    if len(pair) < TREND_LB + 1:
        return 0
    closes = pair["close"].values.astype(float)
    last = float(closes[-1])
    win = closes[-Z_LB:]
    sd = float(win.std(ddof=1))
    if sd <= 0:
        return 0
    z = (last - float(win.mean())) / sd
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return 0
    trend = last / prev - 1.0
    if trend > 0 and z < -Z_ENTRY:
        return 1
    if trend < 0 and z > Z_ENTRY:
        return -1
    return 0


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
    tnx = prices.get("^TNX")

    gr = _ret(gold["close"], ts_now, ts_prev) if gold is not None else float("nan")
    dr = _ret(dxy["close"], ts_now, ts_prev) if dxy is not None else float("nan")
    nr = _ret(nk["close"], ts_now, ts_prev) if nk is not None else float("nan")
    tr = _ret(tnx["close"], ts_now, ts_prev) if tnx is not None else float("nan")

    if want_long:
        # Long ⇒ JPY should revert up. Any one confirmation is enough.
        if not np.isnan(gr) and jr < -MIN_MOVE and gr < -MIN_MOVE:
            return True
        if not np.isnan(dr) and jr < -MIN_MOVE and dr > MIN_MOVE:
            return True
        if not np.isnan(nr) and jr < -MIN_MOVE and nr > MIN_MOVE:
            return True
        if not np.isnan(tr) and jr < -MIN_MOVE and tr > MIN_MOVE:
            return True
        return False
    else:
        # Short ⇒ JPY should revert down.
        if not np.isnan(gr) and jr > MIN_MOVE and gr > MIN_MOVE:
            return True
        if not np.isnan(dr) and jr > MIN_MOVE and dr < -MIN_MOVE:
            return True
        if not np.isnan(nr) and jr > MIN_MOVE and nr < -MIN_MOVE:
            return True
        if not np.isnan(tr) and jr > MIN_MOVE and tr < -MIN_MOVE:
            return True
        return False


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    s11 = _v11_signal(pair)
    if s11 == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if _crossasset_confirms(pair, prices, want_long=(s11 == 1)):
        direction = s11
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
