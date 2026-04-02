#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 8b: ML-Based Prediction with Fixed Feature Engineering

Uses a simple linear model with momentum features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Global cache
_model_cache = None
_trained = False


def _compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute fixed feature vector for a symbol."""
    if len(df) < 21:
        return None

    closes = df["close"].values

    # Fixed 10 features
    features = []

    # 1. 1-bar return
    features.append((closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0)

    # 2. 5-bar return
    features.append((closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0)

    # 3. 10-bar return
    features.append(
        (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0.0
    )

    # 4. 20-bar return
    features.append(
        (closes[-1] - closes[-21]) / closes[-21] if closes[-21] > 0 else 0.0
    )

    # 5. Volatility (std of 10-bar returns)
    if len(closes) >= 11:
        rets = np.diff(closes[-11:]) / closes[-11:-1]
        features.append(float(np.std(rets)) if len(rets) > 1 else 0.0)
    else:
        features.append(0.0)

    # 6. Distance from 5-bar SMA
    sma5 = np.mean(closes[-5:])
    features.append((closes[-1] - sma5) / sma5 if sma5 > 0 else 0.0)

    # 7. Distance from 10-bar SMA
    sma10 = np.mean(closes[-10:])
    features.append((closes[-1] - sma10) / sma10 if sma10 > 0 else 0.0)

    # 8. Distance from 20-bar SMA
    sma20 = np.mean(closes[-20:])
    features.append((closes[-1] - sma20) / sma20 if sma20 > 0 else 0.0)

    # 9. Price position in 20-bar range (0-1)
    recent = closes[-20:]
    min_p, max_p = np.min(recent), np.max(recent)
    if max_p > min_p:
        features.append((closes[-1] - min_p) / (max_p - min_p))
    else:
        features.append(0.5)

    # 10. Acceleration (change in 5-bar return)
    ret5_now = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
    ret5_prev = (closes[-6] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0.0
    features.append(ret5_now - ret5_prev)

    return np.array(features, dtype=float)


def _train_model(prices: dict[str, pd.DataFrame], symbols: list[str]):
    """Train model using simple linear regression."""
    global _model_cache, _trained

    X_list = []
    y_list = []

    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 50:
            continue

        df = prices[sym]
        closes = df["close"].values

        # Generate training samples
        for i in range(25, len(closes) - 5):
            sub_df = df.iloc[: i + 1]
            feats = _compute_features(sub_df)

            if feats is None:
                continue

            # Target: 3-bar future return
            if closes[i] > 0:
                target = (closes[i + 3] - closes[i]) / closes[i]
            else:
                target = 0.0

            X_list.append(feats)
            y_list.append(target)

    if len(X_list) < 50:
        # Not enough data, use heuristic
        _model_cache = None
        _trained = True
        return

    X = np.array(X_list)
    y = np.array(y_list)

    # Simple linear regression: w = (X'X)^-1 X'y
    # Add small regularization
    XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
    Xty = X.T @ y

    try:
        w = np.linalg.solve(XtX, Xty)
        _model_cache = w
    except:
        _model_cache = None

    _trained = True


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Linear model prediction strategy.
    """
    global _model_cache, _trained

    # Train on first call
    if not _trained:
        _train_model(prices, symbols)

    # Get VIX
    vix_level = 20.0
    if "^VIX" in prices and len(prices["^VIX"]) > 0:
        vix_level = float(prices["^VIX"]["close"].iloc[-1])

    weights = []

    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 25:
            weights.append(0.0)
            continue

        df = prices[sym]
        feats = _compute_features(df)

        if feats is None:
            weights.append(0.0)
            continue

        # Predict
        if _model_cache is None:
            # Fallback: use 5-bar momentum
            closes = df["close"].values
            pred = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
        else:
            pred = float(feats @ _model_cache)

        # Position
        pos = np.sign(pred) * min(abs(pred) * 10, 0.15)
        weights.append(pos)

    # Normalize
    total_lev = sum(abs(w) for w in weights)
    if total_lev > 1.0:
        weights = [w / total_lev for w in weights]

    # VIX scaling
    vix_scale = 1.0
    if vix_level > 30:
        vix_scale = 0.4
    elif vix_level > 25:
        vix_scale = 0.6
    elif vix_level > 20:
        vix_scale = 0.8

    return [w * vix_scale for w in weights]
