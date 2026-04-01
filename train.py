"""
Autoresearch-trader training script. Single-GPU, single-file.

Selective high-volatility gap-fade with VIX regime filter.
Only trades on days with large gaps and elevated VIX, targeting gap fill.
Key: trade infrequently to avoid 10bps cost bleeding.

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
    ohlcv = train_data["ohlcv"]
    tradeable_idx = train_data["tradeable_indices"]
    macro_idx = train_data["macro_indices"]
    tickers = train_data["tradeable_tickers"]
    all_tickers = train_data["all_tickers"]

    # Find VIX index in all_tickers
    vix_idx = None
    for j, t in enumerate(all_tickers):
        if t == "^VIX":
            vix_idx = j
            break

    return {
        "tickers": tickers,
        "tradeable_idx": tradeable_idx,
        "vix_idx": vix_idx,
    }


def generate_orders(strategy, data, day_idx):
    ohlcv = data["ohlcv"]
    tickers = strategy["tickers"]
    tradeable_idx = strategy["tradeable_idx"]
    vix_idx = strategy["vix_idx"]

    if day_idx < 21:
        return []

    today_open = ohlcv[day_idx, tradeable_idx, O]
    prev_close = ohlcv[day_idx - 1, tradeable_idx, C]

    # VIX level (use yesterday's close as proxy for current regime)
    vix_level = float(ohlcv[day_idx - 1, vix_idx, C]) if vix_idx is not None else 20.0

    # Compute ATR over last 14 days
    highs = ohlcv[day_idx - 14:day_idx, tradeable_idx, H]
    lows = ohlcv[day_idx - 14:day_idx, tradeable_idx, L]
    closes_atr = ohlcv[day_idx - 15:day_idx - 1, tradeable_idx, C]
    tr = torch.max(
        torch.max(highs - lows, (highs - closes_atr).abs()),
        (lows - closes_atr).abs()
    )
    atr = tr.mean(dim=0)
    atr_pct = (atr / today_open.clamp(min=1e-8)).clamp(min=0.003, max=0.15)

    # Gap = (open - prev_close) / prev_close
    gap = (today_open - prev_close) / prev_close.clamp(min=1e-8)

    # 20-day momentum for trend context
    close_20 = ohlcv[day_idx - 20, tradeable_idx, C]
    mom_20d = (prev_close - close_20) / close_20.clamp(min=1e-8)

    n_tickers = len(tickers)
    candidates = []

    # Only trade when VIX > 18 (elevated fear = bigger intraday moves)
    min_gap = 0.005 if vix_level > 22 else 0.008 if vix_level > 18 else 0.012

    for i in range(n_tickers):
        open_price = float(today_open[i])
        if open_price <= 0 or np.isnan(open_price):
            continue

        g = float(gap[i])
        atr_p = float(atr_pct[i])
        mom = float(mom_20d[i])

        # Large gap down -> long (gap fade)
        # Stronger signal if stock has positive 20d momentum (gap against trend)
        if g < -min_gap:
            # Gap against uptrend is strongest signal
            signal = abs(g)
            if mom > 0:
                signal *= 1.5  # gap down in uptrend = stronger reversion

            stop_pct = atr_p * 1.5    # wider stop to avoid getting shaken out
            target_pct = abs(g) * 0.6  # target 60% gap fill
            target_pct = max(target_pct, atr_p * 0.8)
            target_pct = min(target_pct, atr_p * 3.0)

            candidates.append({
                "ticker": tickers[i],
                "direction": "long",
                "signal_strength": signal,
                "stop_loss": open_price * (1 - stop_pct),
                "take_profit": open_price * (1 + target_pct),
            })
        # Large gap up -> short (gap fade)
        elif g > min_gap:
            signal = abs(g)
            if mom < 0:
                signal *= 1.5  # gap up in downtrend = stronger reversion

            stop_pct = atr_p * 1.5
            target_pct = abs(g) * 0.6
            target_pct = max(target_pct, atr_p * 0.8)
            target_pct = min(target_pct, atr_p * 3.0)

            candidates.append({
                "ticker": tickers[i],
                "direction": "short",
                "signal_strength": signal,
                "stop_loss": open_price * (1 + stop_pct),
                "take_profit": open_price * (1 - target_pct),
            })

    if not candidates:
        return []

    # Take top 3 by signal strength
    candidates.sort(key=lambda x: x["signal_strength"], reverse=True)
    candidates = candidates[:3]

    weight_each = min(1.0 / len(candidates), 0.5)

    orders = []
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
