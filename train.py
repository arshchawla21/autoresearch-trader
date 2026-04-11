#!/usr/bin/env python3
"""
train.py — v30-macd-divergence
==============================
Hypothesis: MACD histogram divergence is a classic reversal pattern that
hasn't been tested on this dataset. When price makes a lower low over
N bars but MACD histogram makes a HIGHER low, momentum is waning and a
reversal is likely. Mirror for bearish. Gate by 96-bar JPY trend so we
only fade against minor counter-trend moves inside a dominant direction.
Completely new signal family vs v24 oscillator threshold or v28 residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

FAST = 12
SLOW = 26
SIG = 9
DIV_LB = 20  # window to find swing low/high
TREND_LB = 96


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _macd_hist_tail(closes: np.ndarray, n_tail: int) -> np.ndarray | None:
    need = SLOW + SIG + n_tail + 5
    if len(closes) < need:
        return None
    seg = closes[-need:]
    ema_f = _ema(seg, FAST)
    ema_s = _ema(seg, SLOW)
    macd = ema_f - ema_s
    sig = _ema(macd, SIG)
    hist = macd - sig
    return hist[-n_tail:]


def _macd_divergence(pair: pd.DataFrame) -> int:
    closes = pair["close"].values.astype(float)
    if len(closes) < TREND_LB + 1:
        return 0
    hist = _macd_hist_tail(closes, DIV_LB)
    if hist is None:
        return 0
    price_tail = closes[-DIV_LB:]

    # Split window in half; compare recent half vs prior half
    half = DIV_LB // 2
    p_prev = price_tail[:half]
    p_now = price_tail[half:]
    h_prev = hist[:half]
    h_now = hist[half:]

    prev_low = float(p_prev.min())
    now_low = float(p_now.min())
    prev_high = float(p_prev.max())
    now_high = float(p_now.max())

    h_prev_low = float(h_prev.min())
    h_now_low = float(h_now.min())
    h_prev_high = float(h_prev.max())
    h_now_high = float(h_now.max())

    # Last bar must be at/near the recent extreme (entering on the swing)
    last_p = float(price_tail[-1])
    last_h = float(hist[-1])

    bullish_div = (
        now_low < prev_low
        and h_now_low > h_prev_low
        and last_p <= now_low * 1.0002
        and last_h > h_now_low
    )
    bearish_div = (
        now_high > prev_high
        and h_now_high < h_prev_high
        and last_p >= now_high * 0.9998
        and last_h < h_now_high
    )

    if bullish_div:
        return 1
    if bearish_div:
        return -1
    return 0


def _trend_gate(pair: pd.DataFrame, direction: int) -> bool:
    closes = pair["close"].values.astype(float)
    if len(closes) < TREND_LB + 1:
        return False
    last = float(closes[-1])
    prev = float(closes[-1 - TREND_LB])
    if prev <= 0:
        return False
    tr = last / prev - 1.0
    if direction == 1:
        return tr > 0
    if direction == -1:
        return tr < 0
    return False


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    d = _macd_divergence(pair)
    if d == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    if not _trend_gate(pair, d):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    return {"direction": d, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
