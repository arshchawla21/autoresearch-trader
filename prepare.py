"""
Data preparation and evaluation harness for autoresearch-trader.
Downloads the last ~60 days of 15-minute OHLCV data and provides a backtest
with stop-loss / take-profit simulation evaluated on a 30-day out-of-sample window.

Usage:
    python prepare.py download       # download + process data
    python prepare.py eval           # evaluate train.py strategy

Data is stored in ~/.cache/autoresearch-trader/.

DO NOT MODIFY THIS FILE during experimentation.
The only file you edit is train.py.
"""

import os
import sys
import time
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
TRANSACTION_COST_BPS = 0   # per side (entry + exit = 10 bps round-trip)

# yfinance intraday limit is 60 days. We use 59 to be safe.
DATA_START = (datetime.datetime.now() - datetime.timedelta(days=59)).strftime("%Y-%m-%d")
DATA_END = None            # None = today

# OHLCV channel indices (convenience constants for train.py)
O, H, L, C, V = 0, 1, 2, 3, 4

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trader")
DATA_DIR = os.path.join(CACHE_DIR, "data")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed_v2")

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
    """Download raw 15m OHLCV data from Yahoo Finance. Returns DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_ohlcv_15m.parquet")

    if os.path.exists(cache_path):
        print(f"Data: cached at {cache_path}")
        return pd.read_parquet(cache_path)

    trade_tickers = tradeable or TRADEABLE_TICKERS
    all_tickers = list(dict.fromkeys(trade_tickers + MACRO_TICKERS))
    end_date = DATA_END or datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading {len(all_tickers)} tickers from {DATA_START} to {end_date} (15m intervals)...")
    t0 = time.time()

    raw = yf.download(
        all_tickers, start=DATA_START, end=end_date,
        interval="15m",
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
            df_t.index.name = "Datetime"
            frames.append(df_t.reset_index())
        except Exception as e:
            print(f"  Warning: {ticker}: {e}")

    df = pd.concat(frames, ignore_index=True)
    # Rename Datetime to Date to maintain compatibility with the rest of the pipeline
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    df.to_parquet(cache_path, index=False)
    print(f"Data: {df['Ticker'].nunique()} tickers, {time.time()-t0:.1f}s, saved to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Processing: raw OHLCV -> aligned tensors
# ---------------------------------------------------------------------------
def process_data(df_raw, tradeable=None):
    """
    Align raw OHLCV into tensors. Saves to PROCESSED_DIR.

    Key alignment strategy:
    - Round all timestamps to the nearest 15-minute floor to normalize
      slight timestamp jitter between tickers (especially macro indices).
    - Use SPY's rounded timestamps as the master calendar.
    - For each ticker, join on the rounded timestamp. Only bars that
      genuinely exist (had real trading activity) are kept; gaps are NOT
      forward-filled in OHLCV to avoid creating fake flat candles.
    - Missing bars get NaN in the tensor (the strategy must handle them).
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

    # ── Step 1: Parse timestamps and round to 15-min floor ─────────────
    df_raw = df_raw.copy()
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], utc=True)
    df_raw["Date_15m"] = df_raw["Date"].dt.floor("15min")

    # ── Step 2: Build per-ticker DataFrames with rounded timestamps ────
    ticker_data = {}
    for ticker in all_list:
        sub = (df_raw[df_raw["Ticker"] == ticker]
               .sort_values("Date_15m")
               .drop_duplicates(subset="Date_15m", keep="last")  # if multiple bars map to same 15m slot
               .set_index("Date_15m")[["Open", "High", "Low", "Close", "Volume"]]
               .dropna(how="all"))
        if len(sub) < 100:
            print(f"  Skipping {ticker}: only {len(sub)} valid rows")
            continue
        ticker_data[ticker] = sub

    # Refresh lists
    trade_list = [t for t in trade_list if t in ticker_data]
    macro_list = [t for t in macro_list if t in ticker_data]
    all_list = trade_list + [t for t in macro_list if t not in trade_list]

    # ── Step 3: Master calendar from SPY (most complete equity schedule) ──
    if "SPY" in ticker_data:
        master_idx = ticker_data["SPY"].index
    else:
        master_idx = max([df.index for df in ticker_data.values()], key=len)
    master_idx = master_idx.sort_values().drop_duplicates()

    # ── Step 4: Align each ticker ──────────────────────────────────────
    # For equities: exact join — only keep bars where timestamps match.
    # For macro tickers (^VIX, ^TNX etc.): use merge_asof with a 15-min
    # tolerance window, since their timestamps can be offset by a few
    # seconds or minutes from the equity grid.
    macro_set = set(MACRO_TICKERS)
    master_df = pd.DataFrame(index=master_idx)

    for ticker in all_list:
        sub = ticker_data[ticker]

        if ticker in macro_set:
            # merge_asof: find nearest macro bar within ±15 min of each master bar
            left = master_df.reset_index().rename(columns={"Date_15m": "ts"})
            right = sub.reset_index().rename(columns={"Date_15m": "ts"}).sort_values("ts")
            left = left.sort_values("ts")
            merged = pd.merge_asof(
                left, right, on="ts",
                tolerance=pd.Timedelta("15min"),
                direction="nearest",
            ).set_index("ts")
            ticker_data[ticker] = merged[["Open", "High", "Low", "Close", "Volume"]]
        else:
            # Exact reindex — no fill, missing bars stay NaN
            ticker_data[ticker] = sub.reindex(master_idx)

    # ── Step 5: Report coverage ────────────────────────────────────────
    T_periods = len(master_idx)
    N = len(all_list)
    N_trade = len(trade_list)

    print(f"  {T_periods} 15-minute intervals, {N} tickers ({N_trade} tradeable)")
    print(f"  {master_idx[0]} -> {master_idx[-1]}")

    for ticker in all_list:
        n_valid = ticker_data[ticker]["Close"].notna().sum()
        n_total = T_periods
        pct = n_valid / n_total * 100
        flag = " *** LOW" if pct < 80 else ""
        print(f"    {ticker:6s}: {n_valid:5d}/{n_total:5d} bars ({pct:5.1f}%){flag}")

    # ── Step 6: Build OHLCV tensor ─────────────────────────────────────
    ohlcv = torch.full((T_periods, N, 5), float("nan"), dtype=torch.float32)
    for j, ticker in enumerate(all_list):
        sub = ticker_data[ticker]
        for k, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
            vals = sub[col].values.astype(np.float32)
            ohlcv[:, j, k] = torch.from_numpy(vals)

    tradeable_idx = torch.tensor([all_list.index(t) for t in trade_list], dtype=torch.long)
    macro_idx = torch.tensor([all_list.index(t) for t in macro_list], dtype=torch.long)
    dates_list = list(master_idx)

    # ── Step 7: Save ───────────────────────────────────────────────────
    torch.save(ohlcv, os.path.join(PROCESSED_DIR, "ohlcv.pt"))
    torch.save(tradeable_idx, os.path.join(PROCESSED_DIR, "tradeable_indices.pt"))
    torch.save(macro_idx, os.path.join(PROCESSED_DIR, "macro_indices.pt"))
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

        ohlcv              - (T, N_all, 5) float tensor   [O, H, L, C, V]
                             NaN where data was missing (no fake fill).
        all_tickers        - list[str], length N_all
        tradeable_tickers  - list[str], length N_tradeable
        tradeable_indices  - LongTensor, map tradeable -> ohlcv ticker dim
        macro_indices      - LongTensor, map macro -> ohlcv ticker dim
        dates              - list[datetime/Timestamp]
        num_tradeable      - int
        num_all            - int
    """
    P = PROCESSED_DIR
    if not os.path.exists(os.path.join(P, "ohlcv.pt")):
        print("Data not found. Downloading and processing...")
        df = download_data()
        process_data(df)

    data = {}
    data["ohlcv"] = torch.load(os.path.join(P, "ohlcv.pt"), map_location=device, weights_only=True)
    data["tradeable_indices"] = torch.load(os.path.join(P, "tradeable_indices.pt"), map_location=device, weights_only=True)
    data["macro_indices"] = torch.load(os.path.join(P, "macro_indices.pt"), map_location=device, weights_only=True)
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
# Day-trading simulation
# ---------------------------------------------------------------------------

def _simulate_interval_trades(orders, ohlcv_today, ticker_to_idx):
    """
    Simulate trading within a 15-minute interval.

    Each order enters at the Open price. During the bar, if the price hits
    the stop-loss or take-profit level (checked against High/Low), the trade
    exits at that level. Otherwise the trade closes at the bar's Close.

    Returns (interval_return, trade_results).
    """
    interval_return = 0.0
    trade_results = []

    for order in orders:
        ticker = order["ticker"]
        idx = ticker_to_idx.get(ticker)
        if idx is None:
            continue

        direction = order["direction"]
        weight = order["weight"]
        stop_loss = order["stop_loss"]
        take_profit = order["take_profit"]

        entry_price = float(ohlcv_today[idx, O])
        high = float(ohlcv_today[idx, H])
        low = float(ohlcv_today[idx, L])
        close = float(ohlcv_today[idx, C])

        # Skip bars with missing data
        if np.isnan(entry_price) or np.isnan(close) or np.isnan(high) or np.isnan(low):
            continue
        if entry_price <= 0:
            continue

        # Determine which exit conditions triggered
        if direction == "long":
            stop_hit = low <= stop_loss
            limit_hit = high >= take_profit
        else:  # short
            stop_hit = high >= stop_loss
            limit_hit = low <= take_profit

        # Resolve exit price
        if stop_hit and limit_hit:
            # CONSERVATIVE: when both levels are hit within the same bar,
            # we cannot know which triggered first from OHLCV alone.
            # Assume the STOP-LOSS hit first (pessimistic assumption).
            # This prevents strategies from gaming tight SL/TP asymmetry.
            exit_price = stop_loss
            ambiguous = True
        elif stop_hit:
            exit_price = stop_loss
            ambiguous = False
        elif limit_hit:
            exit_price = take_profit
            ambiguous = False
        else:
            exit_price = close  # close at end of interval
            ambiguous = False

        # Compute return
        if direction == "long":
            trade_ret = (exit_price - entry_price) / entry_price
        else:
            trade_ret = (entry_price - exit_price) / entry_price

        # Transaction cost (entry + exit)
        cost = 2 * TRANSACTION_COST_BPS / 10_000
        trade_ret -= cost

        interval_return += weight * trade_ret

        trade_results.append({
            "ticker": ticker,
            "direction": direction,
            "weight": weight,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": trade_ret,
            "stop_hit": stop_hit and not limit_hit,
            "limit_hit": limit_hit and not stop_hit,
            "held_to_close": not stop_hit and not limit_hit,
            "ambiguous": ambiguous,
        })

    return interval_return, trade_results


def _validate_orders(orders, tradeable_tickers):
    """Validate and sanitize orders. Total weight capped at 1.0."""
    if not orders:
        return []

    valid = []
    total_weight = 0.0
    tradeable_set = set(tradeable_tickers)

    for order in orders:
        if not isinstance(order, dict):
            continue
        ticker = order.get("ticker")
        direction = order.get("direction")
        weight = order.get("weight", 0)
        stop_loss = order.get("stop_loss")
        take_profit = order.get("take_profit")

        if ticker not in tradeable_set:
            continue
        if direction not in ("long", "short"):
            continue
        if not (isinstance(weight, (int, float)) and 0 < weight <= 1.0):
            continue
        if stop_loss is None or take_profit is None:
            continue
        if not (isinstance(stop_loss, (int, float)) and isinstance(take_profit, (int, float))):
            continue

        total_weight += weight
        if total_weight > 1.0 + 1e-6:
            # Scale down this order to fit
            excess = total_weight - 1.0
            weight -= excess
            if weight <= 0:
                break
            order = dict(order)
            order["weight"] = weight
            total_weight = 1.0

        valid.append(order)

    return valid


def _compute_metrics(interval_returns, all_trades):
    """Compute strategy metrics from 15m interval returns."""
    if len(interval_returns) == 0:
        return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0,
                "win_rate": 0, "avg_daily_trades": 0}

    rets = np.array(interval_returns)

    # Sharpe ratio (annualized, 6552 15-min trading periods in a year)
    # 252 trading days * 26 periods (6.5 hours of 15m intervals)
    if len(rets) < 2 or rets.std() < 1e-10:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(6552) * rets.mean() / rets.std())

    # Total return
    total_return = float(np.prod(1 + rets) - 1)

    # Max drawdown
    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_drawdown = float(np.min(drawdowns))

    # Win rate (intervals with positive return)
    win_rate = float(np.mean(rets > 0))

    # Average trades per interval
    n_periods = len(rets)
    avg_trades = len(all_trades) / n_periods if n_periods > 0 else 0

    return {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_daily_trades": float(avg_trades),
    }


def _find_date_idx(dates, target_date):
    """Find index of nearest trading period >= target_date."""
    for i, d in enumerate(dates):
        if d >= target_date:
            return i
    return len(dates)


# ---------------------------------------------------------------------------
# Single backtest (used by evaluate and visualise.py)
# ---------------------------------------------------------------------------

def run_backtest(build_strategy_fn, generate_orders_fn, data,
                 train_end_idx, test_start_idx, test_end_idx,
                 device="cpu", verbose=True):
    """
    Run a single train/test backtest. Returns detailed results for analysis.
    """
    dates = data["dates"]
    ohlcv = data["ohlcv"]
    tradeable_idx = data["tradeable_indices"]
    ticker_to_idx = {t: i for i, t in enumerate(data["tradeable_tickers"])}

    # Build training data slice
    train_data = {
        "ohlcv": data["ohlcv"][:train_end_idx].to(device),
        "dates": data["dates"][:train_end_idx],
        "all_tickers": data["all_tickers"],
        "tradeable_tickers": data["tradeable_tickers"],
        "tradeable_indices": data["tradeable_indices"].to(device),
        "macro_indices": data["macro_indices"].to(device),
        "num_tradeable": data["num_tradeable"],
        "num_all": data["num_all"],
    }

    if verbose:
        print(f"  Train: {dates[0]} to {dates[train_end_idx-1]} ({train_end_idx} intervals)")
        print(f"  Test:  {dates[test_start_idx]} to {dates[test_end_idx-1]} ({test_end_idx - test_start_idx} intervals)")

    # Build strategy
    t0 = time.time()
    strategy = build_strategy_fn(train_data)
    build_time = time.time() - t0

    if build_time > TIME_BUDGET and verbose:
        print(f"  WARNING: build_strategy took {build_time:.1f}s (budget: {TIME_BUDGET}s)")

    # Run test period
    interval_returns = []
    all_trades = []
    per_interval_trades = []
    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tradeable_mask = tradeable_idx.numpy() if isinstance(tradeable_idx, torch.Tensor) else tradeable_idx

    for bar_idx in range(test_start_idx, test_end_idx):
        # Generate orders — strategy sees data up to bar_idx
        try:
            orders = generate_orders_fn(strategy, data, bar_idx)
        except Exception as e:
            if verbose:
                print(f"  Error on {dates[bar_idx]}: {e}")
            orders = []

        orders = _validate_orders(orders, data["tradeable_tickers"])

        # Simulate using tradeable tickers' OHLCV
        ohlcv_today = ohlcv_np[bar_idx, tradeable_mask]  # (N_tradeable, 5)
        interval_ret, trades = _simulate_interval_trades(orders, ohlcv_today, ticker_to_idx)

        interval_returns.append(interval_ret)
        all_trades.extend(trades)
        per_interval_trades.append(trades)

    metrics = _compute_metrics(interval_returns, all_trades)
    metrics["build_time"] = build_time
    metrics["train_intervals"] = train_end_idx
    metrics["test_intervals"] = test_end_idx - test_start_idx

    return {
        "interval_returns": np.array(interval_returns),
        "all_trades": all_trades,
        "per_interval_trades": per_interval_trades,
        "metrics": metrics,
        "build_time": build_time,
        "test_dates": dates[test_start_idx:test_end_idx],
    }


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(build_strategy_fn, generate_orders_fn, device="cpu"):
    """
    Evaluate an intraday trading strategy using a 30-day train / 30-day test split.

    Args:
        build_strategy_fn(train_data: dict) -> strategy: any
            Given training data, build/train a strategy.
            Must complete within TIME_BUDGET (300s).

        generate_orders_fn(strategy, data: dict, bar_idx: int) -> list[dict]
            Called sequentially for every 15-minute interval.

    Returns:
        dict with metrics for the 30-day evaluation slice.
    """
    data = load_data(device="cpu")
    dates = data["dates"]
    n_periods = len(dates)

    if n_periods < 100:
        print("ERROR: Not enough data points to evaluate.")
        return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0,
                "win_rate": 0, "avg_daily_trades": 0}

    # Split: Find the index roughly 30 days from the start date
    start_date = dates[0]
    split_date = start_date + datetime.timedelta(days=30)
    split_idx = _find_date_idx(dates, split_date)

    # Fallback in case the dates array is shorter than expected
    if split_idx <= 0 or split_idx >= n_periods:
        split_idx = n_periods // 2

    print(f"Evaluating: 30 days train, 30 days test...")
    print(f"Data: {n_periods} 15-minute intervals from {dates[0]} to {dates[-1]}")

    train_end = split_idx
    test_start = split_idx
    test_end = n_periods

    result = run_backtest(
        build_strategy_fn, generate_orders_fn, data,
        train_end_idx=train_end,
        test_start_idx=test_start,
        test_end_idx=test_end,
        device=device,
        verbose=True,
    )

    metrics = result["metrics"]
    metrics["test_start"] = str(dates[test_start])
    metrics["test_end"] = str(dates[test_end - 1])

    # Ambiguity report
    all_trades = result["all_trades"]
    n_total = len(all_trades)
    n_ambiguous = sum(1 for t in all_trades if t.get("ambiguous", False))
    ambig_pct = n_ambiguous / n_total * 100 if n_total > 0 else 0
    metrics["ambiguous_pct"] = ambig_pct

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (Out-of-sample):")
    print(f"  Sharpe:         {metrics['sharpe_ratio']:.4f}")
    print(f"  Return:         {metrics['total_return']*100:.2f}%")
    print(f"  MaxDD:          {metrics['max_drawdown']*100:.2f}%")
    print(f"  WinRate:        {metrics['win_rate']*100:.1f}%")
    print(f"  Avg Trades/bar: {metrics['avg_daily_trades']:.2f}")
    if ambig_pct > 5:
        print(f"  ⚠ AMBIGUOUS:    {n_ambiguous}/{n_total} trades ({ambig_pct:.1f}%) hit both SL and TP")
        print(f"                  (resolved conservatively as stop-loss)")
        print(f"                  Consider widening stops to reduce ambiguity.")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python prepare.py download   # download and process 15m data")
        print("  python prepare.py eval       # evaluate train.py strategy")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "download":
        print(f"Cache: {CACHE_DIR}\n")
        df = download_data()
        print()
        process_data(df)
        print()
        data = load_data()
        print("Summary:")
        print(f"  Tradeable : {data['num_tradeable']}  {data['tradeable_tickers'][:5]}...")
        print(f"  Macro     : {len(data['macro_indices'])}  (VIX, TNX, GLD, ...)")
        print(f"  Intervals : {data['ohlcv'].shape[0]}  ({data['dates'][0]} -> {data['dates'][-1]})")

        # NaN report
        ohlcv = data['ohlcv']
        nan_count = torch.isnan(ohlcv[:, :, C]).sum(dim=0)
        total = ohlcv.shape[0]
        print(f"\n  Close-price coverage per ticker:")
        for j, ticker in enumerate(data['all_tickers']):
            valid = total - nan_count[j].item()
            print(f"    {ticker:6s}: {valid:5d}/{total} ({valid/total*100:.1f}%)")

        print(f"\nDone! Ready to train.")

    elif cmd == "eval":
        from train import build_strategy, generate_orders
        results = evaluate(build_strategy, generate_orders)

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python prepare.py [download|eval]")