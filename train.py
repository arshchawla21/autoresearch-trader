#!/usr/bin/env python3
"""
train.py — Strategy v9: Logistic Regression Direction Classifier
==================================================================
Hypothesis: a simple linear model trained on the warmup window should be
able to learn a weak but consistent MR-biased direction signal from
engineered features (pair z-score, recent returns, DXY diff, TNX diff,
VIX level). If it learns *anything* beyond random it should beat the
momentum baselines. Trade when |predicted prob - 0.5| exceeds a
threshold, so we only act on high-confidence signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


TP_PIPS = 15.0
SL_PIPS = 10.0

FEATURE_WINDOWS = (3, 6, 12, 20)
PROB_THRESHOLD = 0.55

_MODEL: LogisticRegression | None = None
_SCALER: StandardScaler | None = None
_TRAINED_ON: int = -1


def _build_features(pair: pd.Series, dxy: pd.Series, tnx: pd.Series, vix: pd.Series) -> np.ndarray | None:
    n = len(pair)
    if n < max(FEATURE_WINDOWS) + 2:
        return None
    feats: list[float] = []
    # Pair return features
    for w in FEATURE_WINDOWS:
        if pair.iloc[-w - 1] > 0:
            feats.append((float(pair.iloc[-1]) - float(pair.iloc[-w - 1])) / float(pair.iloc[-w - 1]))
        else:
            feats.append(0.0)
    # Pair z-score (20 bars)
    w20 = pair.iloc[-20:]
    mu, sd = float(w20.mean()), float(w20.std(ddof=1))
    feats.append((float(pair.iloc[-1]) - mu) / sd if sd > 0 else 0.0)
    # DXY recent return (6 bars)
    if len(dxy) >= 7 and dxy.iloc[-7] > 0:
        feats.append((float(dxy.iloc[-1]) - float(dxy.iloc[-7])) / float(dxy.iloc[-7]))
    else:
        feats.append(0.0)
    # TNX recent return
    if len(tnx) >= 7 and tnx.iloc[-7] > 0:
        feats.append((float(tnx.iloc[-1]) - float(tnx.iloc[-7])) / float(tnx.iloc[-7]))
    else:
        feats.append(0.0)
    # VIX level (z over recent window)
    if len(vix) >= 48:
        vw = vix.iloc[-48:]
        vmu, vsd = float(vw.mean()), float(vw.std(ddof=1))
        feats.append((float(vix.iloc[-1]) - vmu) / vsd if vsd > 0 else 0.0)
    else:
        feats.append(0.0)
    return np.array(feats, dtype=float)


def _build_training_set(prices: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray] | None:
    pair = prices["JPY=X"]["close"].dropna()
    dxy = prices.get("DX-Y.NYB")
    tnx = prices.get("^TNX")
    vix = prices.get("^VIX")
    if dxy is None or tnx is None or vix is None:
        return None
    dxy_c = dxy["close"].reindex(pair.index, method="ffill")
    tnx_c = tnx["close"].reindex(pair.index, method="ffill")
    vix_c = vix["close"].reindex(pair.index, method="ffill")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    # Walk through warmup bars and build (features at bar i, sign of return i→i+1)
    start_idx = max(FEATURE_WINDOWS) + 5
    for i in range(start_idx, len(pair) - 1):
        f = _build_features(
            pair.iloc[: i + 1],
            dxy_c.iloc[: i + 1],
            tnx_c.iloc[: i + 1],
            vix_c.iloc[: i + 1],
        )
        if f is None:
            continue
        next_ret = float(pair.iloc[i + 1]) - float(pair.iloc[i])
        if abs(next_ret) < 1e-9:
            continue
        X_list.append(f)
        y_list.append(1 if next_ret > 0 else 0)
    if len(X_list) < 200:
        return None
    return np.vstack(X_list), np.array(y_list, dtype=int)


def _ensure_model(prices: dict[str, pd.DataFrame]) -> bool:
    global _MODEL, _SCALER, _TRAINED_ON
    pair = prices.get("JPY=X")
    if pair is None:
        return False
    n = len(pair)
    if _MODEL is not None:
        return True
    # Only train once we have enough warmup data
    if n < 300:
        return False
    built = _build_training_set(prices)
    if built is None:
        return False
    X, y = built
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(max_iter=500, C=0.5)
    model.fit(Xs, y)
    _SCALER = scaler
    _MODEL = model
    _TRAINED_ON = n
    return True


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    if not _ensure_model(prices):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    pair = prices["JPY=X"]["close"].dropna()
    dxy = prices["DX-Y.NYB"]["close"]
    tnx = prices["^TNX"]["close"]
    vix = prices["^VIX"]["close"]

    dxy_c = dxy.reindex(pair.index, method="ffill")
    tnx_c = tnx.reindex(pair.index, method="ffill")
    vix_c = vix.reindex(pair.index, method="ffill")

    feat = _build_features(pair, dxy_c, tnx_c, vix_c)
    if feat is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    assert _MODEL is not None and _SCALER is not None
    Xs = _SCALER.transform(feat.reshape(1, -1))
    prob_up = float(_MODEL.predict_proba(Xs)[0, 1])

    if prob_up > PROB_THRESHOLD:
        direction = 1
    elif prob_up < 1.0 - PROB_THRESHOLD:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
