#!/usr/bin/env python3
"""
train.py — Strategy v11: Gradient Boosting Direction Classifier
=================================================================
Hypothesis: the logistic model (v9) was too linear to capture the real
conditional structure of "when does USD/JPY mean-revert vs trend". A
non-linear boosted tree model on the same feature set should separate
regimes and produce more trades with higher confidence. Same walk-forward
training on the warmup window, then predict on eval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


TP_PIPS = 15.0
SL_PIPS = 10.0

FEATURE_WINDOWS = (3, 6, 12, 20, 48)
PROB_THRESHOLD = 0.53

_MODEL: HistGradientBoostingClassifier | None = None


def _build_features(pair: pd.Series, dxy: pd.Series, tnx: pd.Series, vix: pd.Series, gold: pd.Series) -> np.ndarray | None:
    if len(pair) < max(FEATURE_WINDOWS) + 2:
        return None
    feats: list[float] = []
    p_last = float(pair.iloc[-1])
    # Pair return features
    for w in FEATURE_WINDOWS:
        past = float(pair.iloc[-w - 1])
        feats.append((p_last - past) / past if past > 0 else 0.0)
    # Pair z-scores at two horizons
    for w in (20, 48):
        window = pair.iloc[-w:]
        mu, sd = float(window.mean()), float(window.std(ddof=1))
        feats.append((p_last - mu) / sd if sd > 0 else 0.0)
    # Realised vol (std of 20 returns)
    ret20 = pair.iloc[-21:].pct_change().dropna().values
    feats.append(float(np.std(ret20, ddof=1)) if len(ret20) > 1 else 0.0)
    # Cross-asset 6-bar returns
    for s in (dxy, tnx, vix, gold):
        if len(s) >= 7 and float(s.iloc[-7]) > 0:
            feats.append((float(s.iloc[-1]) - float(s.iloc[-7])) / float(s.iloc[-7]))
        else:
            feats.append(0.0)
    # VIX level z
    if len(vix) >= 48:
        vw = vix.iloc[-48:]
        vmu, vsd = float(vw.mean()), float(vw.std(ddof=1))
        feats.append((float(vix.iloc[-1]) - vmu) / vsd if vsd > 0 else 0.0)
    else:
        feats.append(0.0)
    return np.array(feats, dtype=float)


def _aligned_series(prices: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series] | None:
    pair = prices["JPY=X"]["close"].dropna()
    dxy = prices.get("DX-Y.NYB")
    tnx = prices.get("^TNX")
    vix = prices.get("^VIX")
    gold = prices.get("GC=F")
    if dxy is None or tnx is None or vix is None or gold is None:
        return None
    dxy_c = dxy["close"].reindex(pair.index, method="ffill")
    tnx_c = tnx["close"].reindex(pair.index, method="ffill")
    vix_c = vix["close"].reindex(pair.index, method="ffill")
    gold_c = gold["close"].reindex(pair.index, method="ffill")
    return pair, dxy_c, tnx_c, vix_c, gold_c


def _build_training_set(prices: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray] | None:
    aligned = _aligned_series(prices)
    if aligned is None:
        return None
    pair, dxy, tnx, vix, gold = aligned
    X: list[np.ndarray] = []
    y: list[int] = []
    start = max(FEATURE_WINDOWS) + 5
    for i in range(start, len(pair) - 1):
        f = _build_features(pair.iloc[: i + 1], dxy.iloc[: i + 1], tnx.iloc[: i + 1], vix.iloc[: i + 1], gold.iloc[: i + 1])
        if f is None:
            continue
        delta = float(pair.iloc[i + 1]) - float(pair.iloc[i])
        if abs(delta) < 1e-9:
            continue
        X.append(f)
        y.append(1 if delta > 0 else 0)
    if len(X) < 300:
        return None
    return np.vstack(X), np.array(y, dtype=int)


def _ensure_model(prices: dict[str, pd.DataFrame]) -> bool:
    global _MODEL
    if _MODEL is not None:
        return True
    pair = prices.get("JPY=X")
    if pair is None or len(pair) < 400:
        return False
    built = _build_training_set(prices)
    if built is None:
        return False
    X, y = built
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=30,
        l2_regularization=1.0,
    )
    model.fit(X, y)
    _MODEL = model
    return True


def trade(prices: dict[str, pd.DataFrame]) -> dict:
    if not _ensure_model(prices):
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    aligned = _aligned_series(prices)
    if aligned is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
    pair, dxy, tnx, vix, gold = aligned
    feat = _build_features(pair, dxy, tnx, vix, gold)
    if feat is None:
        return {"direction": 0, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}

    assert _MODEL is not None
    prob_up = float(_MODEL.predict_proba(feat.reshape(1, -1))[0, 1])

    if prob_up > PROB_THRESHOLD:
        direction = 1
    elif prob_up < 1.0 - PROB_THRESHOLD:
        direction = -1
    else:
        direction = 0

    return {"direction": direction, "tp_pips": TP_PIPS, "sl_pips": SL_PIPS}
