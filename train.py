#!/usr/bin/env python3
"""
train.py — v2-logit-xasset
==========================
Hypothesis: a logistic regression on cross-asset returns (USD/JPY, DXY
proxy, TNX, gold, Nikkei) + synthetic VIX + USD/JPY z-score, fit once
on 90d warmup, can predict the sign of the next 4-bar USD/JPY return
with enough edge to beat 0.8p spread. Model cached at module level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TP_PIPS = 12.0
SL_PIPS = 8.0
LONG_THRESH = 0.54
SHORT_THRESH = 0.46

FEAT_COLS = [
    "jpy_r4", "jpy_r16",
    "dxy_r4", "dxy_r16",
    "tnx_r16", "gold_r16", "n225_r16",
    "vix",
    "jpy_z20",
]

_FITTED: tuple | None = None


def _align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return s.reindex(idx).ffill()


def _full_features(prices: dict[str, pd.DataFrame]):
    pair = prices["JPY=X"]["close"].astype(float)
    dxy = _align(prices["DX-Y.NYB"]["close"].astype(float), pair.index)
    tnx = _align(prices["^TNX"]["close"].astype(float), pair.index)
    gold = _align(prices["GC=F"]["close"].astype(float), pair.index)
    n225 = _align(prices["^N225"]["close"].astype(float), pair.index)
    vix = _align(prices["^VIX"]["close"].astype(float), pair.index)

    def lr(s: pd.Series, n: int) -> pd.Series:
        return np.log(s / s.shift(n))

    feat = pd.DataFrame(
        {
            "jpy_r4": lr(pair, 4),
            "jpy_r16": lr(pair, 16),
            "dxy_r4": lr(dxy, 4),
            "dxy_r16": lr(dxy, 16),
            "tnx_r16": lr(tnx, 16),
            "gold_r16": lr(gold, 16),
            "n225_r16": lr(n225, 16),
            "vix": vix,
            "jpy_z20": (pair - pair.rolling(20).mean())
            / (pair.rolling(20).std(ddof=1) + 1e-12),
        },
        index=pair.index,
    )
    return feat, pair


def _fit(prices: dict[str, pd.DataFrame]) -> None:
    global _FITTED
    feat, pair = _full_features(prices)
    fwd = np.log(pair.shift(-4) / pair)
    df = pd.concat([feat, fwd.rename("fwd")], axis=1).dropna()
    df = df.iloc[:-4]  # drop last few rows with NaN forward
    if len(df) < 500:
        return
    X = df[FEAT_COLS].values
    y = (df["fwd"] > 0).astype(int).values
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0) + 1e-12
    Xs = (X - mu) / sd
    model = LogisticRegression(C=0.5, max_iter=500, solver="lbfgs")
    model.fit(Xs, y)
    _FITTED = (model, mu, sd)


def _last_feat_vec(prices: dict[str, pd.DataFrame]) -> np.ndarray | None:
    pair = prices["JPY=X"]["close"].astype(float)
    if len(pair) < 25:
        return None
    dxy_s = prices["DX-Y.NYB"]["close"].astype(float)
    tnx_s = prices["^TNX"]["close"].astype(float)
    gold_s = prices["GC=F"]["close"].astype(float)
    n225_s = prices["^N225"]["close"].astype(float)
    vix_s = prices["^VIX"]["close"].astype(float)

    def lr(s: pd.Series, n: int) -> float:
        if len(s) < n + 1:
            return float("nan")
        prev = float(s.iloc[-1 - n])
        if prev <= 0:
            return float("nan")
        return float(np.log(float(s.iloc[-1]) / prev))

    p20 = pair.iloc[-20:].values
    mu20 = float(p20.mean())
    sd20 = float(p20.std(ddof=1)) + 1e-12
    jpy_z20 = (float(pair.iloc[-1]) - mu20) / sd20

    return np.array([
        lr(pair, 4),
        lr(pair, 16),
        lr(dxy_s, 4),
        lr(dxy_s, 16),
        lr(tnx_s, 16),
        lr(gold_s, 16),
        lr(n225_s, 16),
        float(vix_s.iloc[-1]) if len(vix_s) else float("nan"),
        jpy_z20,
    ])


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    global _FITTED
    if _FITTED is None:
        _fit(prices)
    if _FITTED is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    model, mu, sd = _FITTED

    x = _last_feat_vec(prices)
    if x is None or np.any(np.isnan(x)):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    xs = ((x - mu) / sd).reshape(1, -1)
    prob_up = float(model.predict_proba(xs)[0, 1])

    if prob_up > LONG_THRESH:
        direction = 1
    elif prob_up < SHORT_THRESH:
        direction = -1
    else:
        direction = 0
    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
