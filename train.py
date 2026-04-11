#!/usr/bin/env python3
"""
train.py — v50-v47-dead-market
==============================
Hypothesis: v47 fires across all regimes. In very quiet markets
(lowest 10% ATR) the oscillator signals are dominated by noise and
there is no real move to catch. Skip when 20-bar ATR is below the 10th
percentile of its own 200-bar distribution. Structural regime filter.
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
VOL_LB = 20


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


def _pullback_at(pair: pd.DataFrame, offset: int) -> tuple[bool, bool, float]:
    """Return (long_pullback, short_pullback, trend) at bar -offset (1 = current)."""
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

    # williams %R over bars [idx-WR_LB+1, idx]
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
    return long_pullback, short_pullback, trend


def _pullback_signal(pair: pd.DataFrame) -> int:
    lp1, sp1, tr1 = _pullback_at(pair, 1)
    lp2, sp2, _ = _pullback_at(pair, 2)
    if tr1 > 0 and lp1 and lp2:
        return 1
    if tr1 < 0 and sp1 and sp2:
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


def _atr_pips(pair: pd.DataFrame, n: int) -> float:
    if len(pair) < n + 1:
        return TP_BASE  # fallback
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

    # dead-market filter: skip when ATR is in bottom 10% of 200-bar window
    if len(pair) >= 220:
        highs = pair["high"].values.astype(float)
        lows = pair["low"].values.astype(float)
        closes_all = pair["close"].values.astype(float)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes_all[:-1]),
                np.abs(lows[1:] - closes_all[:-1]),
            ),
        )
        # per-bar ATR20 over the last 200 bars
        atr_series = np.array([tr[-(200 + 20 - i):-(20 - i)].mean() if (20 - i) > 0
                               else tr[-200:].mean() for i in range(200)])
        # simpler: take last 200 rolling ATR20 values
        atr_hist = np.convolve(tr, np.ones(ATR_LB) / ATR_LB, mode="valid")
        atr_window = atr_hist[-200:] if len(atr_hist) >= 200 else atr_hist
        cur_atr = atr_hist[-1]
        lo_q = float(np.quantile(atr_window, 0.10))
        if cur_atr < lo_q:
            return {"direction": 0, "tp_pips": tp, "sl_pips": sl}

    s = _pullback_signal(pair)
    if s == 0:
        return {"direction": 0, "tp_pips": tp, "sl_pips": sl}
    if _crossasset_confirms(pair, prices, want_long=(s == 1)):
        direction = s
    else:
        direction = 0
    return {"direction": direction, "tp_pips": tp, "sl_pips": sl}
