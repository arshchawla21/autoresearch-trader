#!/usr/bin/env python3
"""
train.py — v18-hour-seasonality
===============================
Hypothesis: USD/JPY has systematic intraday bias tied to FX session flows
(Tokyo fix, London open, NY fix, etc.). Learn the mean 4-bar forward
log-return per UTC hour from the 90-day warmup — only 24 parameters, low
overfit risk — and at eval take a position in the direction of that hour's
bias whenever |bias| is strong enough. Module-level cache for the 24 biases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_PIPS = 15.0
SL_PIPS = 10.0

FWD_H = 4            # 1h forward label
BIAS_MIN = 2e-5      # 2bp minimum |bias|
_HOUR_DIR: np.ndarray | None = None  # shape (24,) of {-1,0,1}


def _fit_hour_bias(prices: dict[str, pd.DataFrame]) -> None:
    global _HOUR_DIR
    pair = prices["JPY=X"]["close"].astype(float)
    if len(pair) < 500:
        return
    fwd = np.log(pair.shift(-FWD_H) / pair).dropna()
    idx = fwd.index
    hours = (
        idx.tz_convert("UTC").hour.to_numpy()
        if idx.tzinfo is not None
        else idx.hour.to_numpy()
    )
    dirs = np.zeros(24, dtype=np.int8)
    vals = fwd.values
    for h in range(24):
        mask = hours == h
        if mask.sum() < 50:
            continue
        mean_ret = float(vals[mask].mean())
        if mean_ret > BIAS_MIN:
            dirs[h] = 1
        elif mean_ret < -BIAS_MIN:
            dirs[h] = -1
    _HOUR_DIR = dirs


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    global _HOUR_DIR
    if _HOUR_DIR is None:
        _fit_hour_bias(prices)
    if _HOUR_DIR is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    pair = prices.get("JPY=X")
    if pair is None or len(pair) == 0:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    ts = pair.index[-1]
    hour = ts.tz_convert("UTC").hour if ts.tzinfo is not None else ts.hour
    direction = int(_HOUR_DIR[hour])
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
