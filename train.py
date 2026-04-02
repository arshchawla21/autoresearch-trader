#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 6: Adaptive Momentum with ATR-Based Position Sizing

Hypothesis: Volatility-adjusted position sizing using ATR should
improve risk-adjusted returns. We use EMA crossover signals scaled
by inverse volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_ema(prices: np.ndarray, period: int) -> float:
    """Compute EMA for the given price series."""
    if len(prices) < period:
        return float(prices[-1]) if len(prices) > 0 else 0.0

    alpha = 2.0 / (period + 1)
    ema = float(prices[0])
    for price in prices[1:]:
        ema = alpha * float(price) + (1 - alpha) * ema
    return ema


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range."""
    if len(df) < period + 1:
        return 0.0

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    tr_values = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr_values.append(max(tr1, tr2, tr3))

    if len(tr_values) >= period:
        recent_tr = tr_values[-period:]
        return float(np.mean(recent_tr))
    return 0.0


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Adaptive momentum with ATR position sizing.

    1. EMA 12/26 crossover for trend direction
    2. Position size = signal strength / ATR (volatility adjustment)
    3. VIX-based overall exposure cap
    """
    # Get VIX for overall exposure
    vix_level = 20.0
    if "^VIX" in prices and len(prices["^VIX"]) > 0:
        vix_level = float(prices["^VIX"]["close"].iloc[-1])

    raw_weights = []

    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 30:
            raw_weights.append(0.0)
            continue

        df = prices[sym]
        closes = df["close"].values

        # EMAs
        ema12 = _compute_ema(closes, 12)
        ema26 = _compute_ema(closes, 26)

        # Trend signal
        if ema26 > 0:
            trend = (ema12 - ema26) / ema26
        else:
            trend = 0.0

        # ATR for volatility scaling
        atr = _compute_atr(df, period=14)
        current_price = float(closes[-1])

        if atr > 0 and current_price > 0:
            # Inverse vol scaling: lower position in high vol
            vol_factor = 0.02 / (atr / current_price)  # normalize ATR as % of price
            vol_factor = min(vol_factor, 2.0)  # cap scaling
        else:
            vol_factor = 1.0

        # Position = trend * vol_factor
        pos = np.sign(trend) * min(abs(trend) * 10 * vol_factor, 0.2)
        raw_weights.append(pos)

    # Normalize
    total_lev = sum(abs(w) for w in raw_weights)
    if total_lev > 1.0:
        raw_weights = [w / total_lev for w in raw_weights]

    # VIX exposure scaling
    vix_scale = 1.0
    if vix_level > 30:
        vix_scale = 0.4
    elif vix_level > 25:
        vix_scale = 0.6
    elif vix_level > 20:
        vix_scale = 0.8

    return [w * vix_scale for w in raw_weights]
