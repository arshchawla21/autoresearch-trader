#!/usr/bin/env python3
"""
train.py — v34-session-asymmetric
=================================
Hypothesis: In Tokyo hours USD/JPY leads its cross-asset peers (BoJ/domestic
flow driven) so fading against the move makes no sense — ride the 1-hour
momentum instead. In London hours DXY leads and USD/JPY lags, so v24's
pullback + cross-asset confirm structure is the correct model. Stack two
different sub-strategies per session. Outside those windows, flat.
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
TREND_LB = 96
LB_SHORT = 4
MIN_MOVE = 0.0005
TOKYO_MOMO_THRESH = 0.0010  # 10 bps 1h move to trigger

LONDON_HOURS = set(range(7, 16))
TOKYO_HOURS = set(range(0, 6))


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


def _v24_london(pair: pd.DataFrame, prices: dict) -> int:
    closes = pair["close"].values.astype(float)
    if len(closes) < max(Z_LB, RSI_LB + 1, WR_LB, TREND_LB) + 1:
        return 0
    last = float(closes[-1])
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return 0
    trend = last / prev - 1.0

    win = closes[-Z_LB:]
    sd = float(win.std(ddof=1))
    z = (last - float(win.mean())) / sd if sd > 0 else 0.0
    rsi = _rsi(closes, RSI_LB)
    wr = _williams_r(pair, WR_LB)

    long_pull = (
        (z < -Z_ENTRY)
        or (not np.isnan(rsi) and rsi < RSI_LOW)
        or (not np.isnan(wr) and wr < WR_LOW)
    )
    short_pull = (
        (z > Z_ENTRY)
        or (not np.isnan(rsi) and rsi > RSI_HIGH)
        or (not np.isnan(wr) and wr > WR_HIGH)
    )
    s = 0
    if trend > 0 and long_pull:
        s = 1
    elif trend < 0 and short_pull:
        s = -1
    if s == 0:
        return 0

    # cross-asset confirm
    ts_now = pair.index[-1]
    ts_prev = pair.index[-1 - LB_SHORT]
    jr = float(np.log(last / float(closes[-1 - LB_SHORT])))

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

    if s == 1:
        ok = (
            (not np.isnan(gr) and jr < -MIN_MOVE and gr < -MIN_MOVE)
            or (not np.isnan(dr) and jr < -MIN_MOVE and dr > MIN_MOVE)
            or (not np.isnan(nr) and jr < -MIN_MOVE and nr > MIN_MOVE)
        )
    else:
        ok = (
            (not np.isnan(gr) and jr > MIN_MOVE and gr > MIN_MOVE)
            or (not np.isnan(dr) and jr > MIN_MOVE and dr < -MIN_MOVE)
            or (not np.isnan(nr) and jr > MIN_MOVE and nr < -MIN_MOVE)
        )
    return s if ok else 0


def _tokyo_momo(pair: pd.DataFrame) -> int:
    """Ride 1h (4-bar) JPY momentum in Tokyo — JPY leads here."""
    closes = pair["close"].values.astype(float)
    if len(closes) < LB_SHORT + 1:
        return 0
    last = float(closes[-1])
    prev = float(closes[-1 - LB_SHORT])
    if prev <= 0:
        return 0
    r = np.log(last / prev)
    if r > TOKYO_MOMO_THRESH:
        return 1
    if r < -TOKYO_MOMO_THRESH:
        return -1
    return 0


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    h = pair.index[-1].hour

    if h in LONDON_HOURS:
        d = _v24_london(pair, prices)
        return {"direction": d, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if h in TOKYO_HOURS:
        d = _tokyo_momo(pair)
        return {"direction": d, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
