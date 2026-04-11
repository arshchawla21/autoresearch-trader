#!/usr/bin/env python3
"""
train.py — Strategy v8: USD/JPY vs DXY Divergence Mean Reversion
=================================================================
Hypothesis: USD/JPY and DXY should co-move intraday (both are USD-up
trades). When USD/JPY z-score diverges sharply from DXY z-score over the
same window, that divergence tends to close. If USD/JPY is stretched
high vs DXY, short the pair expecting convergence. Vice versa. This is
structurally different from pure pair-mean-reversion — it requires BOTH
a USD/JPY extreme AND a cross-asset mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TP_PIPS = 15.0
SL_PIPS = 10.0

LOOKBACK = 20
DIV_THRESHOLD = 1.0   # minimum z-score gap between pair and dxy


def _z(series: pd.Series, n: int) -> float | None:
    if len(series) < n:
        return None
    w = series.iloc[-n:]
    mu = float(w.mean())
    sd = float(w.std(ddof=1))
    if sd <= 0:
        return None
    return (float(series.iloc[-1]) - mu) / sd


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    pair = prices.get("JPY=X")
    dxy = prices.get("DX-Y.NYB")
    if pair is None or dxy is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    p_close = pair["close"].dropna()
    d_close = dxy["close"].dropna()
    if len(p_close) < LOOKBACK + 1 or len(d_close) < LOOKBACK + 1:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    pz = _z(p_close, LOOKBACK)
    dz = _z(d_close, LOOKBACK)
    if pz is None or dz is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    gap = pz - dz  # positive → pair stretched high relative to DXY

    if gap > DIV_THRESHOLD and pz > 0.5:
        direction = -1    # pair rich vs DXY → fade
    elif gap < -DIV_THRESHOLD and pz < -0.5:
        direction = 1     # pair cheap vs DXY → buy
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
