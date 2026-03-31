"""
Data preparation and evaluation harness for autoresearch-trader.
Downloads raw OHLCV market data and provides the fixed backtesting metric.

Usage:
    python prepare.py                  # full prep (download + process)
    python prepare.py --tickers SPY    # single ticker for quick testing

Data is stored in ~/.cache/autoresearch-trader/.

DO NOT MODIFY THIS FILE during experimentation.
The only file you edit is train.py.
"""

import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
TRAIN_END = "2025-06-30"   # last date in training set (inclusive)
TEST_START = "2025-07-01"  # first date in test set
DATA_START = "2021-01-01"  # start of data download (~5 years)
TRANSACTION_COST_BPS = 10  # 10 basis points per unit turnover
TARGET_LEVERAGE = 1.0      # gross leverage for position normalization

# OHLCV channel indices (convenience constants for train.py)
O, H, L, C, V = 0, 1, 2, 3, 4

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trader")
DATA_DIR = os.path.join(CACHE_DIR, "data")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")

# Tradeable universe: liquid ETFs + major stocks
TRADEABLE_TICKERS = [
    # Broad market ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB",
    # Top stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "JNJ", "UNH", "PG", "HD", "MA", "BAC",
]

# Reference-only tickers (not tradeable, available as signals)
MACRO_TICKERS = [
    "^VIX",    # Volatility index (fear gauge)
    "^TNX",    # 10-year Treasury yield
    "GLD",     # Gold ETF
    "UUP",     # US Dollar index ETF
    "TLT",     # Long-term Treasury ETF
    "HYG",     # High-yield corporate bond ETF
]

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(tradeable=None):
    """Download raw OHLCV data from Yahoo Finance. Returns DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_ohlcv.parquet")

    if os.path.exists(cache_path):
        print(f"Data: cached at {cache_path}")
        return pd.read_parquet(cache_path)

    trade_tickers = tradeable or TRADEABLE_TICKERS
    all_tickers = list(dict.fromkeys(trade_tickers + MACRO_TICKERS))

    print(f"Downloading {len(all_tickers)} tickers from {DATA_START}...")
    t0 = time.time()

    raw = yf.download(
        all_tickers, start=DATA_START,
        end=datetime.now().strftime("%Y-%m-%d"),
        auto_adjust=True, threads=True,
    )
    if raw.empty:
        print("ERROR: download failed."); sys.exit(1)

    frames = []
    for ticker in all_tickers:
        try:
            df_t = (raw.xs(ticker, axis=1, level=1).copy()
                    if isinstance(raw.columns, pd.MultiIndex) else raw.copy())
            df_t["Ticker"] = ticker
            df_t.index.name = "Date"
            frames.append(df_t.reset_index())
        except Exception as e:
            print(f"  Warning: {ticker}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    print(f"Data: {df['Ticker'].nunique()} tickers, {time.time()-t0:.1f}s, saved to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Processing: raw OHLCV → aligned tensors
# ---------------------------------------------------------------------------

def process_data(df_raw, tradeable=None):
    """
    Align raw OHLCV into tensors. Saves to PROCESSED_DIR.

    Outputs (all torch tensors / pickle):
        ohlcv.pt              — (D, N_all, 5) float32 [Open,High,Low,Close,Volume]
        forward_returns.pt    — (D, N_tradeable) float32, simple close-to-close returns
        tradeable_indices.pt  — LongTensor, tradeable ticker positions in ohlcv
        macro_indices.pt      — LongTensor, macro ticker positions in ohlcv
        train_end_idx.pt      — scalar int
        test_start_idx.pt     — scalar int
        dates.pkl             — list of datetime.date
        all_tickers.pkl       — ordered list of all ticker names
        tradeable_tickers.pkl — ordered list of tradeable ticker names
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(os.path.join(PROCESSED_DIR, "ohlcv.pt")):
        print(f"Data: already processed at {PROCESSED_DIR}")
        return

    trade_list = tradeable or TRADEABLE_TICKERS
    available = set(df_raw["Ticker"].unique())
    trade_list = [t for t in trade_list if t in available]
    macro_list = [t for t in MACRO_TICKERS if t in available]
    all_list = trade_list + [t for t in macro_list if t not in trade_list]

    print(f"Processing: {len(trade_list)} tradeable + {len(macro_list)} macro tickers")

    # Build per-ticker DataFrames, find common dates
    ticker_data = {}
    common_dates = None
    for ticker in all_list:
        sub = df_raw[df_raw["Ticker"] == ticker].sort_values("Date").set_index("Date")
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(sub) < 100:
            print(f"  Skipping {ticker}: only {len(sub)} rows"); continue
        ticker_data[ticker] = sub
        idx = sub.index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)

    common_dates = common_dates.sort_values()
    trade_list = [t for t in trade_list if t in ticker_data]
    macro_list = [t for t in macro_list if t in ticker_data]
    all_list = trade_list + [t for t in macro_list if t not in trade_list]

    D, N = len(common_dates), len(all_list)
    N_trade = len(trade_list)
    print(f"  {D} trading days, {N} tickers ({N_trade} tradeable)")
    print(f"  {common_dates[0].date()} → {common_dates[-1].date()}")

    # Build OHLCV tensor
    ohlcv = torch.zeros(D, N, 5, dtype=torch.float32)
    for j, ticker in enumerate(all_list):
        sub = ticker_data[ticker].loc[common_dates]
        for k, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
            ohlcv[:, j, k] = torch.from_numpy(sub[col].values.astype(np.float32))

    # Forward simple returns for tradeable assets
    tradeable_idx = torch.tensor([all_list.index(t) for t in trade_list], dtype=torch.long)
    close_trade = ohlcv[:, tradeable_idx, C]
    fwd = torch.zeros(D, N_trade)
    fwd[:-1] = close_trade[1:] / close_trade[:-1].clamp(min=1e-8) - 1.0
    fwd.clamp_(-0.5, 0.5)

    macro_idx = torch.tensor([all_list.index(t) for t in macro_list], dtype=torch.long)

    # Date split
    dates_list = [d.date() if hasattr(d, 'date') else d for d in common_dates]
    train_end_dt = pd.Timestamp(TRAIN_END).date()
    test_start_dt = pd.Timestamp(TEST_START).date()
    train_end_i = max(i for i, d in enumerate(dates_list) if d <= train_end_dt)
    test_start_i = min(i for i, d in enumerate(dates_list) if d >= test_start_dt)
    print(f"  Train: 0–{train_end_i}  |  Test: {test_start_i}–{D-2}")

    # Save
    torch.save(ohlcv, os.path.join(PROCESSED_DIR, "ohlcv.pt"))
    torch.save(fwd, os.path.join(PROCESSED_DIR, "forward_returns.pt"))
    torch.save(tradeable_idx, os.path.join(PROCESSED_DIR, "tradeable_indices.pt"))
    torch.save(macro_idx, os.path.join(PROCESSED_DIR, "macro_indices.pt"))
    torch.save(torch.tensor(train_end_i), os.path.join(PROCESSED_DIR, "train_end_idx.pt"))
    torch.save(torch.tensor(test_start_i), os.path.join(PROCESSED_DIR, "test_start_idx.pt"))
    with open(os.path.join(PROCESSED_DIR, "dates.pkl"), "wb") as f:
        pickle.dump(dates_list, f)
    with open(os.path.join(PROCESSED_DIR, "all_tickers.pkl"), "wb") as f:
        pickle.dump(all_list, f)
    with open(os.path.join(PROCESSED_DIR, "tradeable_tickers.pkl"), "wb") as f:
        pickle.dump(trade_list, f)
    print(f"Saved to {PROCESSED_DIR}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_data(device="cpu"):
    """
    Load all processed data. Returns dict with:

        ohlcv              — (D, N_all, 5) float tensor   [O, H, L, C, V]
        forward_returns    — (D, N_tradeable) float tensor (simple returns)
        all_tickers        — list[str], length N_all
        tradeable_tickers  — list[str], length N_tradeable
        tradeable_indices  — LongTensor, map tradeable → ohlcv ticker dim
        macro_indices      — LongTensor, map macro → ohlcv ticker dim
        dates              — list[datetime.date]
        train_end_idx      — int
        test_start_idx     — int
        num_tradeable      — int
        num_all            — int

    train.py owns ALL feature engineering. This just gives you raw OHLCV.
    """
    P = PROCESSED_DIR
    data = {}
    data["ohlcv"] = torch.load(os.path.join(P, "ohlcv.pt"), map_location=device)
    data["forward_returns"] = torch.load(os.path.join(P, "forward_returns.pt"), map_location=device)
    data["tradeable_indices"] = torch.load(os.path.join(P, "tradeable_indices.pt"), map_location=device)
    data["macro_indices"] = torch.load(os.path.join(P, "macro_indices.pt"), map_location=device)
    data["train_end_idx"] = torch.load(os.path.join(P, "train_end_idx.pt")).item()
    data["test_start_idx"] = torch.load(os.path.join(P, "test_start_idx.pt")).item()
    with open(os.path.join(P, "dates.pkl"), "rb") as f:
        data["dates"] = pickle.load(f)
    with open(os.path.join(P, "all_tickers.pkl"), "rb") as f:
        data["all_tickers"] = pickle.load(f)
    with open(os.path.join(P, "tradeable_tickers.pkl"), "rb") as f:
        data["tradeable_tickers"] = pickle.load(f)
    data["num_tradeable"] = len(data["tradeable_tickers"])
    data["num_all"] = len(data["all_tickers"])
    return data


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(predict_fn, data, device="cuda"):
    """
    Walk-forward backtest on the test period.

    Interface contract for predict_fn:
    ──────────────────────────────────
        predict_fn(ohlcv_history, meta) -> position_scores

        ohlcv_history : (T, N_all, 5) float tensor
            Raw OHLCV for ALL tickers, sliced up to and including today.
            Channels: [O=0, H=1, L=2, C=3, V=4].
            No future data is ever included.

        meta : dict
            'all_tickers'        — list[str], all ticker names
            'tradeable_tickers'  — list[str], tradeable ticker names
            'tradeable_indices'  — LongTensor, positions in ohlcv ticker dim
            'macro_indices'      — LongTensor, positions in ohlcv ticker dim
            'today_idx'          — int, today's index in the full dataset

        returns : (num_tradeable,) tensor of raw position scores.

    Protocol:
        For each test day t  (test_start_idx ≤ t < D-1):
        1. predict_fn sees ohlcv[:t+1] — everything up to today's close.
        2. Scores normalized: weights = scores × (TARGET_LEVERAGE / Σ|scores|).
        3. Portfolio earns forward_returns[t]  (today close → tomorrow close).
        4. Cost = TRANSACTION_COST_BPS/10000 × Σ|weight_change|.

    Returns dict:
        sharpe_ratio  — annualized Sharpe ratio (PRIMARY METRIC)
        total_return  — cumulative simple return over test period
        max_drawdown  — worst peak-to-trough decline
        avg_turnover  — mean daily turnover
        num_test_days — number of days in backtest
    """
    ohlcv = data["ohlcv"].to(device)
    fwd = data["forward_returns"].to(device)
    test_start = data["test_start_idx"]
    D = ohlcv.shape[0]

    meta = {
        "all_tickers": data["all_tickers"],
        "tradeable_tickers": data["tradeable_tickers"],
        "tradeable_indices": data["tradeable_indices"].to(device),
        "macro_indices": data["macro_indices"].to(device),
    }

    daily_returns = []
    daily_turnover = []
    prev_weights = torch.zeros(data["num_tradeable"], device=device)

    for t in range(test_start, D - 1):
        meta["today_idx"] = t
        history = ohlcv[:t + 1]

        raw_scores = predict_fn(history, meta)
        if not isinstance(raw_scores, torch.Tensor):
            raw_scores = torch.tensor(raw_scores, dtype=torch.float32, device=device)
        raw_scores = raw_scores.float().to(device)
        assert raw_scores.shape == (data["num_tradeable"],), \
            f"predict_fn must return ({data['num_tradeable']},), got {raw_scores.shape}"

        abs_sum = raw_scores.abs().sum() + 1e-10
        weights = raw_scores * (TARGET_LEVERAGE / abs_sum)

        turnover = (weights - prev_weights).abs().sum().item()
        tc = TRANSACTION_COST_BPS / 10000 * turnover
        port_ret = (weights * fwd[t]).sum().item() - tc

        daily_returns.append(port_ret)
        daily_turnover.append(turnover)
        prev_weights = weights.clone()

    rets = np.array(daily_returns)
    turns = np.array(daily_turnover)

    if len(rets) < 2 or rets.std() < 1e-10:
        sharpe = 0.0
    else:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252)

    cumulative = np.cumprod(1 + rets)
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / (peak + 1e-10)
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
    avg_turn = turns.mean() if len(turns) > 0 else 0.0

    return {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turn,
        "num_test_days": len(rets),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    print(f"Cache: {CACHE_DIR}\n")
    df = download_data(args.tickers)
    print()
    process_data(df, args.tickers)
    print()

    data = load_data()
    print("Summary:")
    print(f"  Tradeable : {data['num_tradeable']}  {data['tradeable_tickers'][:5]}...")
    print(f"  Macro     : {len(data['macro_indices'])}  (VIX, TNX, GLD, ...)")
    print(f"  Days      : {data['ohlcv'].shape[0]}  ({data['dates'][0]} → {data['dates'][-1]})")
    print(f"  Train     : 0–{data['train_end_idx']}")
    print(f"  Test      : {data['test_start_idx']}–{data['ohlcv'].shape[0]-2}")
    print(f"\nDone! Ready to train.")