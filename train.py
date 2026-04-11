#!/usr/bin/env python3
"""
train.py — v65-v61-osc-vote
===========================
Hypothesis: v61 uses OR across z/RSI/WR (any single oscillator extreme
fires). Require instead 2-of-3 oscillators to agree — stricter
consensus filters out single-indicator noise and should lift win rate.
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
ATR_MED_LB = 200
ATR_SPIKE_LB = 500
ATR_SPIKE_Q = 0.90

Z_LB = 20
Z_ENTRY_BASE = 1.2
Z_ENTRY_MIN = 0.9
Z_ENTRY_MAX = 1.8
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


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    return np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )


def _atr_series(pair: pd.DataFrame, n: int) -> np.ndarray:
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    closes = pair["close"].values.astype(float)
    tr = _true_range(highs, lows, closes)
    if len(tr) < n:
        return np.array([])
    # rolling mean
    csum = np.cumsum(np.insert(tr, 0, 0.0))
    atr = (csum[n:] - csum[:-n]) / n
    return atr


def _adaptive_z(pair: pd.DataFrame) -> float:
    atr_ser = _atr_series(pair, ATR_LB)
    if len(atr_ser) < ATR_MED_LB:
        return Z_ENTRY_BASE
    cur = float(atr_ser[-1])
    med = float(np.median(atr_ser[-ATR_MED_LB:]))
    if med <= 0:
        return Z_ENTRY_BASE
    ratio = cur / med
    # ratio 1.0 -> base; higher vol -> stricter
    z = Z_ENTRY_BASE * ratio
    return max(Z_ENTRY_MIN, min(Z_ENTRY_MAX, z))


def _pullback_at(pair: pd.DataFrame, offset: int, z_entry: float) -> tuple[bool, bool, float]:
    closes = pair["close"].values.astype(float)
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    idx = -offset
    if len(closes) < max(Z_LB, RSI_LB + 1, WR_LB, TREND_LB) + offset + 1:
        return False, False, 0.0
    last = float(closes[idx])
    prev = float(closes[idx - TREND_LB])
    if prev <= 0:
        return False, False, 0.0
    trend = last / prev - 1.0

    win = closes[idx - Z_LB + 1 : idx + 1] if idx != -1 else closes[-Z_LB:]
    sd = float(win.std(ddof=1))
    zv = (last - float(win.mean())) / sd if sd > 0 else 0.0

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

    z_long = zv < -z_entry
    z_short = zv > z_entry
    rsi_long = (not np.isnan(rsi_val)) and rsi_val < RSI_LOW
    rsi_short = (not np.isnan(rsi_val)) and rsi_val > RSI_HIGH
    wr_long = (not np.isnan(wr_val)) and wr_val < WR_LOW
    wr_short = (not np.isnan(wr_val)) and wr_val > WR_HIGH
    long_pullback = (int(z_long) + int(rsi_long) + int(wr_long)) >= 2
    short_pullback = (int(z_short) + int(rsi_short) + int(wr_short)) >= 2
    return long_pullback, short_pullback, trend


def _pullback_signal(pair: pd.DataFrame, z_entry: float) -> int:
    lp1, sp1, tr1 = _pullback_at(pair, 1, z_entry)
    lp2, sp2, _ = _pullback_at(pair, 2, z_entry)
    if tr1 > 0 and lp1 and lp2:
        return 1
    if tr1 < 0 and sp1 and sp2:
        return -1
    return 0


def _atr_pips(pair: pd.DataFrame, n: int) -> float:
    if len(pair) < n + 1:
        return TP_BASE
    highs = pair["high"].values.astype(float)
    lows = pair["low"].values.astype(float)
    closes = pair["close"].values.astype(float)
    tr = _true_range(highs, lows, closes)
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

    # Vol spike skip: stand aside in top decile of 500-bar ATR distribution
    atr_hist = _atr_series(pair, ATR_LB)
    if len(atr_hist) >= ATR_SPIKE_LB:
        cur_atr = float(atr_hist[-1])
        thresh = float(np.quantile(atr_hist[-ATR_SPIKE_LB:], ATR_SPIKE_Q))
        if cur_atr > thresh:
            return {"direction": 0, "tp_pips": tp, "sl_pips": sl}

    z_entry = _adaptive_z(pair)
    s = _pullback_signal(pair, z_entry)
    if s == 0:
        return {"direction": 0, "tp_pips": tp, "sl_pips": sl}
    if _crossasset_confirms(pair, prices, want_long=(s == 1)):
        direction = s
    else:
        direction = 0
    return {"direction": direction, "tp_pips": tp, "sl_pips": sl}
