#!/usr/bin/env python3
"""
train.py — v26-multi-horizon-trend
==================================
Hypothesis: v24 champion used a single 96-bar trend gate. Replace with
an OR of three trend horizons (48/96/192 bars). A bar is eligible for
long if ANY of them is positive (and no horizon is strongly against).
More bars pass the gate → more pullback+cross-asset opportunities,
same quality since the filters (z/RSI/WR + cross-asset) still apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

Z_LB = 20
Z_ENTRY = 1.2
RSI_LB = 14
RSI_LOW = 32.0
RSI_HIGH = 68.0
WR_LB = 14
WR_LOW = -85.0
WR_HIGH = -15.0
TREND_LBS = (48, 96, 192)
LB_SHORT = 4
MIN_MOVE = 0.0005


def _rsi(closes: np.ndarray, n: int) -> float:
    if len(closes) < n + 1:
        return float("nan")
    diffs = np.diff(closes[-(n + 1):])
    ups = np.maximum(diffs, 0.0)
    downs = np.maximum(-diffs, 0.0)
    avg_up = ups.mean()
    avg_down = downs.mean()
    if avg_down <= 0 and avg_up <= 0:
        return 50.0
    if avg_down <= 0:
        return 100.0
    rs = avg_up / avg_down
    return float(100.0 - 100.0 / (1.0 + rs))


def _williams_r(pair: pd.DataFrame, n: int) -> float:
    if len(pair) < n:
        return float("nan")
    tail = pair.iloc[-n:]
    hi = float(tail["high"].max())
    lo = float(tail["low"].min())
    c = float(tail["close"].iloc[-1])
    if hi - lo <= 0:
        return float("nan")
    return -100.0 * (hi - c) / (hi - lo)


def _trend_direction(closes: np.ndarray) -> int:
    """Return +1 if any horizon up and none strongly down; -1 mirror; else 0."""
    rets = []
    for lb in TREND_LBS:
        if len(closes) < lb + 1:
            continue
        prev = float(closes[-1 - lb])
        if prev <= 0:
            continue
        rets.append(float(closes[-1]) / prev - 1.0)
    if not rets:
        return 0
    any_up = any(r > 0 for r in rets)
    any_down = any(r < 0 for r in rets)
    majority_up = sum(1 for r in rets if r > 0)
    majority_down = sum(1 for r in rets if r < 0)
    if majority_up >= 2 and not (majority_down >= 2):
        return 1
    if majority_down >= 2 and not (majority_up >= 2):
        return -1
    return 0


def _pullback_signal(pair: pd.DataFrame) -> int:
    closes = pair["close"].values.astype(float)
    if len(closes) < max(Z_LB, RSI_LB + 1, WR_LB, max(TREND_LBS)) + 1:
        return 0

    trend = _trend_direction(closes)
    if trend == 0:
        return 0

    last = float(closes[-1])
    win = closes[-Z_LB:]
    sd = float(win.std(ddof=1))
    z = (last - float(win.mean())) / sd if sd > 0 else 0.0

    rsi = _rsi(closes, RSI_LB)
    wr = _williams_r(pair, WR_LB)

    long_pullback = (
        (z < -Z_ENTRY)
        or (not np.isnan(rsi) and rsi < RSI_LOW)
        or (not np.isnan(wr) and wr < WR_LOW)
    )
    short_pullback = (
        (z > Z_ENTRY)
        or (not np.isnan(rsi) and rsi > RSI_HIGH)
        or (not np.isnan(wr) and wr > WR_HIGH)
    )

    if trend > 0 and long_pullback:
        return 1
    if trend < 0 and short_pullback:
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
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    s = _pullback_signal(pair)
    if s == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if _crossasset_confirms(pair, prices, want_long=(s == 1)):
        direction = s
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
