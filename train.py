#!/usr/bin/env python3
"""
train.py — v57-v47-exhaustion
=============================
Hypothesis: v47 persistence requires pullback valid on 2 consecutive bars.
But it doesn't require the pullback to be *ending*. Add exhaustion: the
z-score on the current bar must be LESS extreme than on the prior bar
(i.e., the pullback has bottomed and is starting to reverse). This should
lift win rate by catching the turn rather than entering mid-dip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PIP = 0.01
TP_BASE = 15.0
SL_BASE = 10.0
TP_MIN, TP_MAX = 8.0, 25.0
SL_MIN, SL_MAX = 6.0, 18.0
ATR_LB = 20

Z_LB = 20
Z_ENTRY = 1.2
RSI_LB = 14
RSI_LOW = 32.0
RSI_HIGH = 68.0
WR_LB = 14
WR_LOW = -85.0
WR_HIGH = -15.0
TREND_LB = 96
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


def _z_at(closes: np.ndarray, idx: int, n: int) -> float:
    if idx == -1:
        win = closes[-n:]
    else:
        win = closes[idx - n + 1 : idx + 1]
    if len(win) < n:
        return float("nan")
    sd = float(win.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float((closes[idx] - float(win.mean())) / sd)


def _pullback_at(pair: pd.DataFrame, offset: int) -> tuple[bool, bool, float, float]:
    """Return (long_pullback, short_pullback, trend, z) at bar -offset."""
    closes = pair["close"].values.astype(float)
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    idx = -offset
    if len(closes) < max(Z_LB, RSI_LB + 1, WR_LB, TREND_LB) + offset + 1:
        return False, False, 0.0, float("nan")
    last = float(closes[idx])
    prev = float(closes[idx - TREND_LB])
    if prev <= 0:
        return False, False, 0.0, float("nan")
    trend = last / prev - 1.0

    zv = _z_at(closes, idx, Z_LB)

    rsi_slice = closes[: idx + 1] if idx != -1 else closes
    rsi_val = _rsi(rsi_slice, RSI_LB)

    if idx == -1:
        hi = float(highs[-WR_LB:].max())
        lo = float(lows[-WR_LB:].min())
        c = float(closes[-1])
    else:
        hi = float(highs[idx - WR_LB + 1 : idx + 1].max())
        lo = float(lows[idx - WR_LB + 1 : idx + 1].min())
        c = float(closes[idx])
    wr_val = -100.0 * (hi - c) / (hi - lo) if hi - lo > 0 else float("nan")

    long_pullback = (
        (zv < -Z_ENTRY)
        or (not np.isnan(rsi_val) and rsi_val < RSI_LOW)
        or (not np.isnan(wr_val) and wr_val < WR_LOW)
    )
    short_pullback = (
        (zv > Z_ENTRY)
        or (not np.isnan(rsi_val) and rsi_val > RSI_HIGH)
        or (not np.isnan(wr_val) and wr_val > WR_HIGH)
    )
    return long_pullback, short_pullback, trend, zv


def _pullback_signal(pair: pd.DataFrame) -> int:
    lp1, sp1, tr1, z1 = _pullback_at(pair, 1)
    lp2, sp2, _, z2 = _pullback_at(pair, 2)
    if np.isnan(z1) or np.isnan(z2):
        return 0
    # Exhaustion: current bar less extreme than prior bar (pullback reversing).
    if tr1 > 0 and lp1 and lp2 and z1 > z2:
        return 1
    if tr1 < 0 and sp1 and sp2 and z1 < z2:
        return -1
    return 0


def _atr_pips(pair: pd.DataFrame, n: int) -> float:
    if len(pair) < n + 1:
        return TP_BASE
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    closes = pair["close"].values.astype(float)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    if len(tr) < n:
        return TP_BASE
    return float(tr[-n:].mean()) / PIP


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

    def _r(other):
        if other is None:
            return float("nan")
        try:
            a = float(other["close"].asof(ts_now))
            b = float(other["close"].asof(ts_prev))
        except Exception:
            return float("nan")
        if np.isnan(a) or np.isnan(b) or b <= 0:
            return float("nan")
        return float(np.log(a / b))

    gr = _r(prices.get("GC=F"))
    dr = _r(prices.get("DX-Y.NYB"))
    nr = _r(prices.get("^N225"))

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
        return {"direction": 0, "tp_pips": TP_BASE, "sl_pips": SL_BASE}

    atr = _atr_pips(pair, ATR_LB)
    tp = max(TP_MIN, min(TP_MAX, 1.5 * atr))
    sl = max(SL_MIN, min(SL_MAX, 1.0 * atr))

    s = _pullback_signal(pair)
    if s == 0:
        return {"direction": 0, "tp_pips": tp, "sl_pips": sl}
    if _crossasset_confirms(pair, prices, want_long=(s == 1)):
        direction = s
    else:
        direction = 0
    return {"direction": direction, "tp_pips": tp, "sl_pips": sl}
