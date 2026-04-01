"""
v12-sector-divergence: Trade stock-vs-sector divergence.

Hypothesis: when a stock moves differently from its sector ETF over the last
2 bars, it tends to revert toward the sector. This is pairs-trading logic
with economic rationale — sector correlation is mean-reverting.

Still uses v9 stops (0.7x/4.0x ATR).
"""

import os
import time
import numpy as np

from prepare import O, H, L, C, V, evaluate

t_start = time.time()

# Stock -> sector ETF mapping
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK", "NVDA": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF",
    "JNJ": "XLV", "UNH": "XLV",
    "PG": "XLP",
}


def build_strategy(train_data):
    ohlcv = train_data["ohlcv"].numpy()
    tidx = train_data["tradeable_indices"].numpy()
    tickers = train_data["tradeable_tickers"]

    closes = ohlcv[:, tidx, C]
    highs = ohlcv[:, tidx, H]
    lows = ohlcv[:, tidx, L]
    tr = (highs - lows) / np.maximum(closes, 1e-8)
    avg_atr_pct = np.nanmean(tr, axis=0)

    # Build ticker -> index in tradeable array
    ticker_to_tidx = {t: i for i, t in enumerate(tickers)}

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "avg_atr_pct": avg_atr_pct,
        "ticker_to_tidx": ticker_to_tidx,
    }


def generate_orders(strategy, data, bar_idx):
    lookback = 2
    if bar_idx < lookback + 1:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    ohlcv = data["ohlcv"]
    avg_atr = strategy["avg_atr_pct"]
    t2i = strategy["ticker_to_tidx"]

    opens = ohlcv[bar_idx, tidx, O].numpy()
    past_close = ohlcv[bar_idx - lookback, tidx, C].numpy()
    curr_close = ohlcv[bar_idx - 1, tidx, C].numpy()
    returns = (curr_close - past_close) / np.maximum(past_close, 1e-8)

    # Compute divergence: stock return minus sector ETF return
    best_score = 0
    best_idx = None
    best_direction = None

    for stock, sector_etf in SECTOR_MAP.items():
        if stock not in t2i or sector_etf not in t2i:
            continue
        si = t2i[stock]
        ei = t2i[sector_etf]

        if np.isnan(returns[si]) or np.isnan(returns[ei]):
            continue
        if opens[si] <= 0 or np.isnan(opens[si]):
            continue

        divergence = returns[si] - returns[ei]
        score = abs(divergence)

        if score > best_score:
            best_score = score
            best_idx = si
            # Fade the divergence: if stock outperformed sector, short it
            best_direction = "short" if divergence > 0 else "long"

    if best_idx is None or best_score < 0.001:
        return []

    # Also check the plain mean-reversion signal as fallback
    valid = np.where((opens > 0) & ~np.isnan(opens) & ~np.isnan(returns))[0]
    abs_ret = np.abs(returns[valid])
    plain_best = valid[np.argmax(abs_ret)]
    plain_score = float(abs_ret[np.argmax(abs_ret)])

    # Use whichever signal is stronger
    if plain_score > best_score * 1.5:
        best_idx = plain_best
        mom = float(returns[plain_best])
        best_direction = "short" if mom > 0 else "long"

    ticker = tickers[best_idx]
    op = float(opens[best_idx])
    atr = float(avg_atr[best_idx])

    sl_dist = max(atr * 0.7, 0.001) * op
    tp_dist = max(atr * 4.0, 0.005) * op

    if best_direction == "long":
        sl = op - sl_dist
        tp = op + tp_dist
    else:
        sl = op + sl_dist
        tp = op - tp_dist

    return [{
        "ticker": ticker,
        "direction": best_direction,
        "weight": 1.0,
        "stop_loss": sl,
        "take_profit": tp,
    }]


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print("\n---")
    print(f"sharpe_ratio:     {results.get('sharpe_ratio', 0):.4f}")
    print(f"total_return:     {results.get('total_return', 0)*100:.2f}%")
    print(f"max_drawdown:     {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"win_rate:         {results.get('win_rate', 0)*100:.1f}%")
    print(f"avg_trades/bar:   {results.get('avg_daily_trades', 0):.2f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
