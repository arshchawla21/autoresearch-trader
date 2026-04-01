"""
Autoresearch-trader training script. Single-GPU, single-file.

Baseline: Simple mean-reversion day-trading strategy.
For each day, identifies stocks that moved > 1 std from recent mean,
takes contrarian positions with 1% stop-loss and 1.5% take-profit.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import numpy as np
import torch

from prepare import O, H, L, C, evaluate

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Strategy API: build_strategy + generate_orders
# ---------------------------------------------------------------------------

def build_strategy(train_data):
    """
    Analyze training data and build strategy parameters.

    Args:
        train_data: dict with ohlcv (D_train, N_all, 5), dates, tickers, etc.

    Returns:
        strategy dict (or any object) passed to generate_orders.
    """
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    tickers = train_data["tradeable_tickers"]
    D = ohlcv.shape[0]

    # Compute daily close-to-close returns for tradeable assets
    close = ohlcv[:, tradeable_idx, C]  # (D, N_tradeable)
    log_close = torch.log(close.clamp(min=1e-8))
    daily_ret = torch.zeros(D, len(tickers))
    daily_ret[1:] = log_close[1:] - log_close[:-1]

    # Rolling 20-day mean and std of returns per ticker
    lookback = 20
    mean_ret = torch.zeros(len(tickers))
    std_ret = torch.zeros(len(tickers))

    if D > lookback:
        recent = daily_ret[-lookback:]
        mean_ret = recent.mean(dim=0)
        std_ret = recent.std(dim=0).clamp(min=1e-6)

    # Average true range (ATR) for stop/limit calibration
    highs = ohlcv[:, tradeable_idx, H]
    lows = ohlcv[:, tradeable_idx, L]
    closes = ohlcv[:, tradeable_idx, C]

    tr = torch.zeros(D, len(tickers))
    tr[1:] = torch.max(
        torch.max(highs[1:] - lows[1:], (highs[1:] - closes[:-1]).abs()),
        (lows[1:] - closes[:-1]).abs()
    )
    atr_14 = tr[-14:].mean(dim=0) if D >= 14 else tr.mean(dim=0)
    atr_pct = (atr_14 / closes[-1].clamp(min=1e-8)).clamp(min=0.002, max=0.10)

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "atr_pct": atr_pct,  # average true range as % of price
        "lookback": lookback,
    }


def generate_orders(strategy, data, day_idx):
    """
    Generate day-trading orders for a single day.

    Args:
        strategy: returned by build_strategy
        data: full data dict from load_data()
              - data["ohlcv"][:day_idx] is known history
              - data["ohlcv"][day_idx, :, O] is today's open (available at market open)
              - DO NOT access today's H, L, C (that is future data)
        day_idx: current day index in the full dataset

    Returns:
        list of order dicts:
        [
            {
                "ticker": str,
                "direction": "long" or "short",
                "weight": float in (0, 1],
                "stop_loss": float (price level),
                "take_profit": float (price level),
            },
            ...
        ]
    """
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    atr_pct = strategy["atr_pct"]
    lookback = strategy["lookback"]

    if day_idx < lookback + 1:
        return []

    # Today's open price
    today_open = ohlcv[day_idx, tradeable_idx, O]

    # Recent returns for signal computation
    closes = ohlcv[day_idx - lookback:day_idx, tradeable_idx, C]
    log_closes = torch.log(closes.clamp(min=1e-8))
    recent_ret = log_closes[-1] - log_closes[0]  # lookback-period return
    recent_std = (log_closes[1:] - log_closes[:-1]).std(dim=0).clamp(min=1e-6)

    # Z-score of recent return
    z_scores = recent_ret / (recent_std * np.sqrt(lookback))

    orders = []
    n_tickers = len(tickers)
    candidates = []

    for i in range(n_tickers):
        open_price = float(today_open[i])
        if open_price <= 0 or np.isnan(open_price):
            continue

        z = float(z_scores[i])
        atr = float(atr_pct[i])

        # Mean reversion: fade extreme moves
        if z < -1.5:  # oversold -> go long
            stop_pct = max(atr * 1.0, 0.005)   # 1x ATR or 0.5% minimum
            target_pct = max(atr * 1.5, 0.008)  # 1.5x ATR or 0.8% minimum
            candidates.append({
                "ticker": tickers[i],
                "direction": "long",
                "signal_strength": abs(z),
                "stop_loss": open_price * (1 - stop_pct),
                "take_profit": open_price * (1 + target_pct),
            })
        elif z > 1.5:  # overbought -> go short
            stop_pct = max(atr * 1.0, 0.005)
            target_pct = max(atr * 1.5, 0.008)
            candidates.append({
                "ticker": tickers[i],
                "direction": "short",
                "signal_strength": abs(z),
                "stop_loss": open_price * (1 + stop_pct),
                "take_profit": open_price * (1 - target_pct),
            })

    if not candidates:
        return []

    # Sort by signal strength, take top 5
    candidates.sort(key=lambda x: x["signal_strength"], reverse=True)
    candidates = candidates[:5]

    # Equal-weight allocation
    weight_each = 1.0 / len(candidates)

    for c in candidates:
        orders.append({
            "ticker": c["ticker"],
            "direction": c["direction"],
            "weight": weight_each,
            "stop_loss": c["stop_loss"],
            "take_profit": c["take_profit"],
        })

    return orders


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")

    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"sharpe_ratio:     {results['sharpe_ratio']:.4f}")
    print(f"total_return:     {results['total_return']*100:.2f}%")
    print(f"max_drawdown:     {results['max_drawdown']*100:.2f}%")
    print(f"win_rate:         {results['win_rate']*100:.1f}%")
    print(f"avg_daily_trades: {results['avg_daily_trades']:.1f}")
    print(f"num_slices:       {results['num_slices']}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
