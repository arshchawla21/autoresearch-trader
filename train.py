#!/usr/bin/env python3
"""
train.py — v42-v36-gold-corr-regime
===================================
Hypothesis: v24's cross-asset confirm relies on gold as one of three
signals. That only works when JPY-gold are in their usual safe-haven
co-move regime. Compute rolling 100-bar correlation of JPY vs gold
returns. Only enable v36 trades when corr > 0 (classical risk-off
regime). When correlation is negative, the assets have decoupled and
the xasset confirm is unreliable → skip. Novel regime gate.
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
VOL_LB = 20
CORR_LB = 100
CORR_MIN = 0.0


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


def _pullback_signal(pair: pd.DataFrame) -> int:
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


def _vol_ok(pair: pd.DataFrame) -> bool:
    if "volume" not in pair.columns:
        return True
    v = pair["volume"].values.astype(float)
    if len(v) < VOL_LB + 1:
        return False
    last = v[-1]
    med = float(np.median(v[-VOL_LB:]))
    return last > med


def _corr_ok(pair: pd.DataFrame, prices: dict) -> bool:
    gold = prices.get("GC=F")
    if gold is None or len(pair) < CORR_LB + 2:
        return True
    try:
        g_aligned = gold["close"].reindex(pair.index[-CORR_LB - 1:], method="ffill").astype(float)
    except Exception:
        return True
    p_tail = pair["close"].iloc[-CORR_LB - 1:].astype(float)
    p_ret = np.diff(np.log(p_tail.values))
    g_vals = g_aligned.values
    if np.any(g_vals <= 0) or np.any(np.isnan(g_vals)):
        return True
    g_ret = np.diff(np.log(g_vals))
    if len(p_ret) != len(g_ret) or len(p_ret) < 20:
        return True
    sd_p = p_ret.std()
    sd_g = g_ret.std()
    if sd_p <= 0 or sd_g <= 0:
        return True
    corr = float(np.mean((p_ret - p_ret.mean()) * (g_ret - g_ret.mean())) / (sd_p * sd_g))
    return corr > CORR_MIN


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
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    s = _pullback_signal(pair)
    if s == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if not _vol_ok(pair):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if not _corr_ok(pair, prices):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if _crossasset_confirms(pair, prices, want_long=(s == 1)):
        direction = s
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
