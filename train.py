#!/usr/bin/env python3
"""
train.py — Strategy v10: ATR-Adaptive Z-Score Mean Reversion
==============================================================
Hypothesis: v4 (pure z-score MR) uses fixed 15/10 pip brackets regardless
of regime. In low-vol conditions 15 pips is too far and the trade expires
at SL from random walk; in high-vol conditions 10 pips is inside noise
and we stop out prematurely. Size TP/SL to recent realised volatility
(ATR) so the bracket represents a constant *vol multiple* rather than a
constant pip distance. Direction logic unchanged from v4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_TP_PIPS = 15.0   # used as fallback
BASE_SL_PIPS = 10.0

Z_LOOKBACK = 20
Z_ENTRY = 1.5
ATR_LOOKBACK = 14
TP_ATR_MULT = 1.3     # TP is 1.3 × ATR
SL_ATR_MULT = 0.9     # SL is 0.9 × ATR (so TP:SL ~= 1.45)
PIP_SIZE = 0.01       # JPY pair


def _atr_pips(df: pd.DataFrame, n: int) -> float:
    if len(df) < n + 1:
        return 0.0
    highs = df["high"].values[-n - 1:]
    lows = df["low"].values[-n - 1:]
    closes = df["close"].values[-n - 1:]
    prev_close = closes[:-1]
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - prev_close),
        np.abs(lows[1:] - prev_close),
    ])
    return float(np.mean(tr) / PIP_SIZE)


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < max(Z_LOOKBACK, ATR_LOOKBACK) + 2:
        return {"direction": 0, "tp_pips": BASE_TP_PIPS, "sl_pips": BASE_SL_PIPS}

    closes = pair["close"].dropna()
    window = closes.iloc[-Z_LOOKBACK:]
    mu = float(window.mean())
    sd = float(window.std(ddof=1))
    if sd <= 0:
        return {"direction": 0, "tp_pips": BASE_TP_PIPS, "sl_pips": BASE_SL_PIPS}
    z = (float(closes.iloc[-1]) - mu) / sd

    atr_pips = _atr_pips(pair, ATR_LOOKBACK)
    if atr_pips < 3.0:
        # degenerate / weekend hole — fall back
        tp = BASE_TP_PIPS
        sl = BASE_SL_PIPS
    else:
        tp = max(5.0, min(40.0, atr_pips * TP_ATR_MULT))
        sl = max(4.0, min(30.0, atr_pips * SL_ATR_MULT))

    if z > Z_ENTRY:
        direction = -1
    elif z < -Z_ENTRY:
        direction = 1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": tp, "sl_pips": sl}
