"""
v33-pairs: Statistical pairs trading.

Define natural pairs of correlated stocks. When the spread deviates from
its short-term mean, go long the underperformer and short the outperformer.
"""

import time
import numpy as np
import torch
from prepare import O, H, L, C, V, evaluate

t_start = time.time()

SL_PCT = 0.005
TP_PCT = 0.005

# Natural pairs (highly correlated stocks)
PAIRS = [
    ("AAPL", "MSFT"),
    ("JPM", "BAC"),
    ("V", "MA"),
    ("GOOGL", "META"),
    ("SPY", "QQQ"),
    ("XLF", "JPM"),
    ("XLK", "AAPL"),
    ("AMZN", "GOOGL"),
    ("UNH", "JNJ"),
    ("HD", "PG"),
]

LOOKBACK = 5  # spread lookback for z-score
SPREAD_WINDOW = 20  # rolling window for spread stats


def build_strategy(train_data):
    tickers = train_data["tradeable_tickers"]
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # Filter pairs where both stocks are tradeable
    valid_pairs = []
    for a, b in PAIRS:
        if a in ticker_to_idx and b in ticker_to_idx:
            valid_pairs.append((ticker_to_idx[a], ticker_to_idx[b], a, b))

    return {
        "tickers": tickers,
        "tradeable_idx": train_data["tradeable_indices"],
        "pairs": valid_pairs,
    }


def generate_orders(strategy, data, bar_idx):
    if bar_idx < SPREAD_WINDOW + 2:
        return []

    tickers = strategy["tickers"]
    tidx = strategy["tradeable_idx"]
    pairs = strategy["pairs"]
    ohlcv = data["ohlcv"]

    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tidx_np = tidx.numpy() if isinstance(tidx, torch.Tensor) else tidx

    opens = ohlcv_np[bar_idx, tidx_np, O]
    closes = ohlcv_np[:bar_idx, tidx_np, C]

    # Compute spread z-score for each pair
    candidates = []
    for idx_a, idx_b, ticker_a, ticker_b in pairs:
        ca = closes[-SPREAD_WINDOW:, idx_a]
        cb = closes[-SPREAD_WINDOW:, idx_b]

        if np.isnan(ca).any() or np.isnan(cb).any():
            continue
        if opens[idx_a] <= 0 or opens[idx_b] <= 0:
            continue
        if np.isnan(opens[idx_a]) or np.isnan(opens[idx_b]):
            continue

        # Log-spread
        spread = np.log(ca) - np.log(cb)
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        if spread_std < 1e-8:
            continue

        current_spread = np.log(closes[-1, idx_a]) - np.log(closes[-1, idx_b])
        z = (current_spread - spread_mean) / spread_std

        if abs(z) > 0.5:  # meaningful deviation
            candidates.append((idx_a, idx_b, ticker_a, ticker_b, z, abs(z)))

    if not candidates:
        return []

    # Sort by abs(z), take top pairs
    candidates.sort(key=lambda x: -x[5])
    top = candidates[:3]

    w = 1.0 / (len(top) * 2)  # 2 positions per pair

    orders = []
    for idx_a, idx_b, ticker_a, ticker_b, z, _ in top:
        op_a = float(opens[idx_a])
        op_b = float(opens[idx_b])

        if z > 0:
            # A outperformed B → short A, long B
            orders.append({
                "ticker": ticker_a, "direction": "short", "weight": w,
                "stop_loss": op_a * (1 + SL_PCT), "take_profit": op_a * (1 - TP_PCT),
            })
            orders.append({
                "ticker": ticker_b, "direction": "long", "weight": w,
                "stop_loss": op_b * (1 - SL_PCT), "take_profit": op_b * (1 + TP_PCT),
            })
        else:
            # B outperformed A → short B, long A
            orders.append({
                "ticker": ticker_a, "direction": "long", "weight": w,
                "stop_loss": op_a * (1 - SL_PCT), "take_profit": op_a * (1 + TP_PCT),
            })
            orders.append({
                "ticker": ticker_b, "direction": "short", "weight": w,
                "stop_loss": op_b * (1 + SL_PCT), "take_profit": op_b * (1 - TP_PCT),
            })

    return orders


if __name__ == "__main__":
    results = evaluate(build_strategy, generate_orders, device="cpu")
    t_end = time.time()
    print(f"\n---\nsharpe={results.get('sharpe_ratio',0):.4f} ret={results.get('total_return',0)*100:.2f}% "
          f"dd={results.get('max_drawdown',0)*100:.2f}% wr={results.get('win_rate',0)*100:.1f}% "
          f"trades/bar={results.get('avg_daily_trades',0):.2f} time={t_end-t_start:.1f}s")
