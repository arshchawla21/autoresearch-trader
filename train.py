#!/usr/bin/env python3
"""
train.py — Strategy Implementation
====================================
THIS IS THE ONLY FILE THE AI AGENT MODIFIES.

Strategy 4: Ensemble of VIX Regime Momentum + Z-Score Mean Reversion

Hypothesis: The two successful individual strategies capture different
market phenomena (trend following vs mean reversion). Combining them
should provide diversification benefits and potentially higher Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_vix_momentum_weights(
    prices: dict[str, pd.DataFrame], symbols: list[str]
) -> dict[str, float]:
    """
    Strategy A: VIX regime-based momentum/mean-reversion.
    High VIX -> mean reversion, Low VIX -> momentum.
    """
    vix_sym = "^VIX"
    if vix_sym not in prices or len(prices[vix_sym]) < 5:
        return {sym: 0.0 for sym in symbols}

    vix_df = prices[vix_sym]
    current_vix = vix_df["close"].iloc[-1]

    VIX_THRESHOLD = 20.0
    high_vol_regime = current_vix > VIX_THRESHOLD

    # Compute 5-bar returns
    returns = {}
    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 6:
            returns[sym] = 0.0
            continue
        df = prices[sym]
        recent_close = df["close"].iloc[-1]
        past_close = df["close"].iloc[-6]
        ret = (recent_close - past_close) / past_close if past_close > 0 else 0.0
        returns[sym] = ret

    sorted_syms = sorted(returns.keys(), key=lambda s: returns[s])
    weights = {sym: 0.0 for sym in symbols}

    if high_vol_regime:
        # Mean-reversion: long worst, short best
        for sym in sorted_syms[:3]:
            weights[sym] = 1.0 / 6.0
        for sym in sorted_syms[-3:]:
            weights[sym] = -1.0 / 6.0
    else:
        # Momentum: long best, short worst
        for sym in sorted_syms[-3:]:
            weights[sym] = 1.0 / 6.0
        for sym in sorted_syms[:3]:
            weights[sym] = -1.0 / 6.0

    return weights


def _get_zscore_weights(
    prices: dict[str, pd.DataFrame], symbols: list[str]
) -> dict[str, float]:
    """
    Strategy B: Z-score mean reversion.
    Trade deviations from 20-bar SMA.
    """
    weights = {sym: 0.0 for sym in symbols}

    for sym in symbols:
        if sym not in prices or len(prices[sym]) < 21:
            continue
        df = prices[sym]
        closes = df["close"].values

        recent = closes[-20:]
        current = closes[-1]

        sma = float(np.mean(recent))
        std = float(np.std(recent))

        if std > 0:
            zscore = (current - sma) / std
            if abs(zscore) >= 1.0:
                weights[sym] = -np.sign(zscore) * min(abs(zscore) * 0.1, 0.15)

    return weights


def trade(
    prices: dict[str, pd.DataFrame],
    current_idx: int,
    symbols: list[str],
) -> list[float]:
    """
    Ensemble of two complementary strategies.

    Weights: 60% VIX regime + 40% z-score
    """
    # Get weights from each strategy
    vix_weights = _get_vix_momentum_weights(prices, symbols)
    zscore_weights = _get_zscore_weights(prices, symbols)

    # Ensemble weighting
    VIX_ALLOC = 0.6
    ZSCORE_ALLOC = 0.4

    final_weights = []
    for sym in symbols:
        combined = VIX_ALLOC * vix_weights.get(
            sym, 0.0
        ) + ZSCORE_ALLOC * zscore_weights.get(sym, 0.0)
        final_weights.append(combined)

    # Normalize to ensure leverage <= 1.0
    total_lev = sum(abs(w) for w in final_weights)
    if total_lev > 1.0:
        final_weights = [w / total_lev for w in final_weights]

    return final_weights
