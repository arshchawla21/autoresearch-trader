#!/usr/bin/env python3
"""
train.py — v10-histgbm-longh
============================
Hypothesis: short-horizon features don't generalize (v2). Try HistGBM on
longer-horizon cross-asset features (4h / 24h / 3d returns on JPY, DXY,
TNX, gold, Nikkei, SPY + VIX level + hour of day) predicting the sign of
the next 24h USD/JPY return. The signal should persist longer so 15/10
pip brackets can harvest it multiple times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TP_PIPS = 15.0
SL_PIPS = 10.0

FWD_H = 96           # 24h forward horizon for label
LONG_THRESH = 0.53
SHORT_THRESH = 0.47

FEAT_COLS = [
    "jpy_r16", "jpy_r96", "jpy_r288",
    "dxy_r16", "dxy_r96",
    "tnx_r96", "gold_r96", "n225_r96", "spy_r96",
    "vix", "hour",
]

_FITTED: tuple | None = None


def _align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return s.reindex(idx).ffill()


def _full_features(prices: dict[str, pd.DataFrame]):
    pair = prices["JPY=X"]["close"].astype(float)
    idx = pair.index
    dxy = _align(prices["DX-Y.NYB"]["close"].astype(float), idx)
    tnx = _align(prices["^TNX"]["close"].astype(float), idx)
    gold = _align(prices["GC=F"]["close"].astype(float), idx)
    n225 = _align(prices["^N225"]["close"].astype(float), idx)
    spy = _align(prices["SPY"]["close"].astype(float), idx)
    vix = _align(prices["^VIX"]["close"].astype(float), idx)

    def lr(s: pd.Series, n: int) -> pd.Series:
        return np.log(s / s.shift(n))

    if idx.tzinfo is not None:
        hours = idx.tz_convert("UTC").hour.to_numpy()
    else:
        hours = idx.hour.to_numpy()

    df = pd.DataFrame(
        {
            "jpy_r16": lr(pair, 16),
            "jpy_r96": lr(pair, 96),
            "jpy_r288": lr(pair, 288),
            "dxy_r16": lr(dxy, 16),
            "dxy_r96": lr(dxy, 96),
            "tnx_r96": lr(tnx, 96),
            "gold_r96": lr(gold, 96),
            "n225_r96": lr(n225, 96),
            "spy_r96": lr(spy, 96),
            "vix": vix,
            "hour": hours.astype(float),
        },
        index=idx,
    )
    return df, pair


def _fit(prices: dict[str, pd.DataFrame]) -> None:
    global _FITTED
    feat, pair = _full_features(prices)
    fwd = np.log(pair.shift(-FWD_H) / pair)
    df = pd.concat([feat, fwd.rename("fwd")], axis=1).dropna()
    if len(df) < 500:
        return
    X = df[FEAT_COLS].values
    y = (df["fwd"] > 0).astype(int).values
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)
    _FITTED = (model,)


def _last_feat_row(prices: dict[str, pd.DataFrame]) -> np.ndarray | None:
    pair_s = prices["JPY=X"]["close"].astype(float)
    if len(pair_s) < 300:
        return None
    dxy_s = prices["DX-Y.NYB"]["close"].astype(float)
    tnx_s = prices["^TNX"]["close"].astype(float)
    gold_s = prices["GC=F"]["close"].astype(float)
    n225_s = prices["^N225"]["close"].astype(float)
    spy_s = prices["SPY"]["close"].astype(float)
    vix_s = prices["^VIX"]["close"].astype(float)

    def lr(s: pd.Series, n: int) -> float:
        if len(s) < n + 1:
            return float("nan")
        prev = float(s.iloc[-1 - n])
        if prev <= 0:
            return float("nan")
        return float(np.log(float(s.iloc[-1]) / prev))

    idx_last = pair_s.index[-1]
    hour = (
        idx_last.tz_convert("UTC").hour if idx_last.tzinfo is not None else idx_last.hour
    )
    return np.array(
        [
            lr(pair_s, 16),
            lr(pair_s, 96),
            lr(pair_s, 288),
            lr(dxy_s, 16),
            lr(dxy_s, 96),
            lr(tnx_s, 96),
            lr(gold_s, 96),
            lr(n225_s, 96),
            lr(spy_s, 96),
            float(vix_s.iloc[-1]) if len(vix_s) else float("nan"),
            float(hour),
        ]
    )


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    global _FITTED
    if _FITTED is None:
        _fit(prices)
    if _FITTED is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    model = _FITTED[0]

    x = _last_feat_row(prices)
    if x is None or np.any(np.isnan(x)):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    if prob > LONG_THRESH:
        direction = 1
    elif prob < SHORT_THRESH:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
