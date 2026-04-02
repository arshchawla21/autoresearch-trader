#!/usr/bin/env python3
"""
prepare.py — Data Download & Fixed Backtesting Harness
=======================================================
READ-ONLY: The AI agent must NOT modify this file.

What it does:
  1. Downloads 60 days of 5-minute OHLCV candles for a predefined universe
     of stocks, ETFs, and market indicators via yfinance.
  2. Caches the data to ~/.cache/autoresearch-trader/ so we don't re-download
     every run.
  3. Provides a deterministic backtesting evaluation that:
       - Uses days 1–30 as "history" the strategy can see from the start.
       - Steps through days 31–60 candle by candle.
       - At each step, calls train.trade(prices_so_far) and receives a
         position vector (weights that sum in absolute value to ≤ 1.0).
       - Computes PnL, Sharpe ratio, max drawdown, total return, and more.
       - Prints a summary and returns a dict of metrics.

Usage:
    uv run prepare.py              # download data + run backtest
    uv run prepare.py --download   # only download / refresh data
    uv run prepare.py --eval       # only run backtest (data must exist)

The strategy must live in train.py and expose:
    trade(prices: dict[str, pd.DataFrame],
          current_idx: int,
          symbols: list[str]) -> list[float]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

# --- Tradeable assets (the strategy allocates across these) ---
STOCKS = [
    "AAPL",   # Apple — mega-cap tech
    "MSFT",   # Microsoft — mega-cap tech
    "NVDA",   # NVIDIA — AI/semis momentum
    "AMZN",   # Amazon — e-comm / cloud
    "TSLA",   # Tesla — high-vol EV play
    "META",   # Meta — social / AI
    "GOOGL",  # Alphabet — search / cloud
    "JPM",    # JPMorgan — financials bellwether
    "XOM",    # ExxonMobil — energy
    "UNH",    # UnitedHealth — healthcare
]

ETFS = [
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "IWM",    # Russell 2000 (small cap)
    "XLF",    # Financials sector
    "XLE",    # Energy sector
]

TRADEABLE = STOCKS + ETFS  # 15 instruments

# --- Market context indicators (strategy can read but NOT trade) ---
INDICATORS = [
    "^VIX",   # CBOE Volatility Index
    "^TNX",   # 10-Year Treasury Yield
    "GLD",    # Gold ETF (proxy)
    "TLT",    # 20+ Year Treasury Bond ETF
    "DX-Y.NYB",  # US Dollar Index
]

ALL_SYMBOLS = TRADEABLE + INDICATORS

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INTERVAL = "5m"           # 5-minute candles
LOOKBACK_DAYS = 60        # total calendar days to fetch
HISTORY_DAYS = 30         # first 30 days = warmup / training data
CACHE_DIR = Path.home() / ".cache" / "autoresearch-trader"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# yfinance caps 5m data at ~60 days, so we fetch in chunks if needed
# but typically a single call works for 60 calendar days.

RISK_FREE_RATE_ANNUAL = 0.05  # 5% annual for Sharpe calc


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _download_symbol(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Download 5-min OHLCV for a single symbol. Returns None on failure."""
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=INTERVAL,
                progress=False,
                auto_adjust=True,
            )
            if df is not None and not df.empty:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Standardize column names
                df.columns = [c.lower() for c in df.columns]
                # Keep only OHLCV
                keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[keep]
                return df
        except Exception as e:
            print(f"  [!] {symbol} attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download (or load from cache) 60 days of 5-min OHLCV data
    for the full universe.

    Returns: dict mapping symbol -> DataFrame with OHLCV columns
             and a DatetimeIndex.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Cache key based on date range + symbols
    cache_key = hashlib.md5(
        f"{start_str}_{end_str}_{INTERVAL}_{'_'.join(ALL_SYMBOLS)}".encode()
    ).hexdigest()[:12]
    cache_meta = CACHE_DIR / f"meta_{cache_key}.json"

    data: dict[str, pd.DataFrame] = {}

    if not force and cache_meta.exists():
        print(f"[✓] Loading cached data ({cache_key})...")
        meta = json.loads(cache_meta.read_text())
        all_good = True
        for sym in ALL_SYMBOLS:
            parquet = CACHE_DIR / f"{sym.replace('^', '_')}_{cache_key}.parquet"
            if parquet.exists():
                data[sym] = pd.read_parquet(parquet)
            else:
                all_good = False
                break
        if all_good and len(data) == len(ALL_SYMBOLS):
            print(f"    Loaded {len(data)} symbols from cache.")
            return data
        print("    Cache incomplete — re-downloading...")
        data = {}

    print(f"[↓] Downloading {len(ALL_SYMBOLS)} symbols, {start_str} → {end_str}, interval={INTERVAL}")
    for sym in ALL_SYMBOLS:
        print(f"  → {sym}...", end=" ", flush=True)
        df = _download_symbol(sym, start_str, end_str)
        if df is not None and len(df) > 0:
            data[sym] = df
            # Save to cache
            parquet = CACHE_DIR / f"{sym.replace('^', '_')}_{cache_key}.parquet"
            df.to_parquet(parquet)
            print(f"OK ({len(df)} rows)")
        else:
            print("FAILED — will be excluded")

    # Write metadata
    cache_meta.write_text(json.dumps({
        "start": start_str,
        "end": end_str,
        "interval": INTERVAL,
        "symbols": list(data.keys()),
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    print(f"\n[✓] Downloaded {len(data)}/{len(ALL_SYMBOLS)} symbols.")
    if missing := set(ALL_SYMBOLS) - set(data.keys()):
        print(f"    Missing: {missing}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _split_by_trading_days(data: dict[str, pd.DataFrame]) -> list[str]:
    """
    Get sorted list of unique trading dates across all symbols.
    """
    all_dates: set[str] = set()
    for df in data.values():
        dates = df.index.normalize().unique()
        all_dates.update(d.strftime("%Y-%m-%d") for d in dates)
    return sorted(all_dates)


def _get_close_matrix(data: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    """
    Build an aligned close-price matrix (index = datetime, cols = symbols).
    Forward-fills missing values within trading hours.
    """
    frames = {}
    for sym in symbols:
        if sym in data:
            frames[sym] = data[sym]["close"].rename(sym)
    if not frames:
        raise ValueError("No close data available for tradeable symbols")
    merged = pd.concat(frames.values(), axis=1, join="outer")
    merged.sort_index(inplace=True)
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)
    return merged


def run_backtest(data: dict[str, pd.DataFrame]) -> dict:
    """
    Fixed backtesting loop.

    - Splits data by trading day.
    - Days 1..HISTORY_DAYS are warmup (the strategy can see them but is not
      evaluated on them).
    - Days HISTORY_DAYS+1..end are the evaluation window.
    - At each 5-min candle in the eval window, we call
          train.trade(prices, current_idx, tradeable_symbols)
      where `prices` is a dict of DataFrames from the start of data up to
      and including the current candle.
    - The strategy returns a weight vector of length len(TRADEABLE).
    - PnL for that step = sum(weights * pct_returns_of_next_candle).
    """
    # Import the strategy
    try:
        import train
    except ImportError:
        print("[✗] Cannot import train.py — make sure it exists in cwd.")
        sys.exit(1)

    trading_days = _split_by_trading_days(data)
    if len(trading_days) < HISTORY_DAYS + 1:
        print(f"[✗] Only {len(trading_days)} trading days found, need >{HISTORY_DAYS}.")
        print("    Try running with --download to refresh data.")
        sys.exit(1)

    eval_start_day = trading_days[HISTORY_DAYS]  # first eval day
    print(f"\n{'='*70}")
    print(f"BACKTEST")
    print(f"{'='*70}")
    print(f"  Total trading days : {len(trading_days)}")
    print(f"  Warmup (history)   : days 1–{HISTORY_DAYS} (up to {eval_start_day})")
    print(f"  Evaluation window  : days {HISTORY_DAYS+1}–{len(trading_days)}")
    print(f"  Tradeable symbols  : {len(TRADEABLE)} ({', '.join(TRADEABLE[:5])}...)")
    print(f"  Indicator symbols  : {len(INDICATORS)} ({', '.join(INDICATORS[:3])}...)")
    print(f"{'='*70}\n")

    # Build close matrix for tradeable assets
    close_matrix = _get_close_matrix(data, TRADEABLE)

    # Determine eval candle indices
    # Make tz-aware to match yfinance's UTC-indexed data
    eval_ts = pd.Timestamp(eval_start_day)
    if close_matrix.index.tz is not None:
        eval_ts = eval_ts.tz_localize(close_matrix.index.tz)
    eval_mask = close_matrix.index >= eval_ts
    eval_indices = np.where(eval_mask)[0]

    if len(eval_indices) < 2:
        print("[✗] Not enough eval candles.")
        sys.exit(1)

    # Prepare sliced data views for the strategy
    # The strategy sees ALL data (including indicators) up to current_idx
    portfolio_returns: list[float] = []
    timestamps: list[pd.Timestamp] = []

    n_candles = len(eval_indices)
    print(f"  Running {n_candles} eval steps...\n")

    for step, idx in enumerate(eval_indices[:-1]):  # last candle has no next return
        # Slice all data up to and including current candle
        current_time = close_matrix.index[idx]
        prices_so_far: dict[str, pd.DataFrame] = {}
        for sym in ALL_SYMBOLS:
            if sym in data:
                mask = data[sym].index <= current_time
                prices_so_far[sym] = data[sym].loc[mask].copy()

        # Call strategy
        try:
            weights = train.trade(prices_so_far, idx, TRADEABLE)
        except Exception as e:
            print(f"  [!] trade() raised at step {step}: {e}")
            weights = [0.0] * len(TRADEABLE)

        # Validate weights
        weights = np.array(weights, dtype=float)
        if len(weights) != len(TRADEABLE):
            print(f"  [!] Expected {len(TRADEABLE)} weights, got {len(weights)} — zeroing.")
            weights = np.zeros(len(TRADEABLE))

        # Enforce leverage constraint: sum(|w|) <= 1.0
        total_leverage = np.sum(np.abs(weights))
        if total_leverage > 1.0 + 1e-9:
            weights = weights / total_leverage  # rescale to 1.0

        # Compute next-candle returns
        next_idx = idx + 1
        current_close = close_matrix.iloc[idx].values
        next_close = close_matrix.iloc[next_idx].values

        # Handle NaN / zero closes
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_returns = np.where(
                current_close > 0,
                (next_close - current_close) / current_close,
                0.0,
            )
        pct_returns = np.nan_to_num(pct_returns, nan=0.0)

        step_return = float(np.dot(weights, pct_returns))
        portfolio_returns.append(step_return)
        timestamps.append(current_time)

        if (step + 1) % 500 == 0 or step == 0:
            cum_ret = float(np.prod([1 + r for r in portfolio_returns]) - 1)
            print(f"    Step {step+1:>5}/{n_candles-1}  |  cum return: {cum_ret:+.4%}")

    # ─── Compute Metrics ──────────────────────────────────────────────────
    returns = np.array(portfolio_returns)
    cum_returns = np.cumprod(1 + returns)
    total_return = float(cum_returns[-1] - 1)

    # Sharpe: annualize assuming ~78 5-min candles/day, ~252 trading days
    candles_per_day = 78  # 6.5 hours * 12 candles/hour
    candles_per_year = candles_per_day * 252
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    rf_per_candle = RISK_FREE_RATE_ANNUAL / candles_per_year
    sharpe = float((mean_ret - rf_per_candle) / std_ret * np.sqrt(candles_per_year)) if std_ret > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # Win rate
    win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0

    # Average win / loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    # Sortino ratio
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
    sortino = float((mean_ret - rf_per_candle) / downside_std * np.sqrt(candles_per_year)) if downside_std > 0 else 0.0

    # Calmar ratio
    calmar = float(total_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-9 else 0.0

    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "n_trades": len(returns),
        "eval_start": eval_start_day,
        "eval_end": trading_days[-1],
    }

    # ─── Print Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total Return      : {total_return:+.4%}")
    print(f"  Sharpe Ratio      : {sharpe:.4f}")
    print(f"  Sortino Ratio     : {sortino:.4f}")
    print(f"  Calmar Ratio      : {calmar:.4f}")
    print(f"  Max Drawdown      : {max_drawdown:.4%}")
    print(f"  Win Rate          : {win_rate:.2%}")
    print(f"  Avg Win           : {avg_win:.6%}")
    print(f"  Avg Loss          : {avg_loss:.6%}")
    print(f"  Profit Factor     : {profit_factor:.4f}")
    print(f"  Eval Candles      : {len(returns)}")
    print(f"  Eval Period       : {eval_start_day} → {trading_days[-1]}")
    print(f"{'='*70}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="autoresearch-trader: data + eval harness")
    parser.add_argument("--download", action="store_true", help="Only download data (skip eval)")
    parser.add_argument("--eval", action="store_true", help="Only run eval (skip download)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached")
    args = parser.parse_args()

    if args.eval:
        # Load cached data
        data = download_all(force=False)
        metrics = run_backtest(data)
        return metrics

    if args.download:
        download_all(force=args.force_download)
        return None

    # Default: download + eval
    data = download_all(force=args.force_download)
    metrics = run_backtest(data)
    return metrics


if __name__ == "__main__":
    main()