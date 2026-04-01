"""
Data preparation and evaluation harness for autoresearch-trader.
Downloads 10 years of daily OHLCV data and provides a day-trading backtest
with stop-loss / take-profit simulation evaluated across 19 expanding-window
6-month test slices.

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
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
TRANSACTION_COST_BPS = 5   # per side (entry + exit = 10 bps round-trip)
NUM_SLICES = 19            # number of 6-month test windows
SLICE_MONTHS = 6           # months per test slice

DATA_START = "2016-04-01"  # ~10 years of history
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
    """Download raw OHLCV data from Yahoo Finance. Returns DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_ohlcv_10y.parquet")

    if os.path.exists(cache_path):
        print(f"Data: cached at {cache_path}")
        return pd.read_parquet(cache_path)

    trade_tickers = tradeable or TRADEABLE_TICKERS
    all_tickers = list(dict.fromkeys(trade_tickers + MACRO_TICKERS))
    end_date = DATA_END or datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading {len(all_tickers)} tickers from {DATA_START} to {end_date}...")
    t0 = time.time()

    raw = yf.download(
        all_tickers, start=DATA_START, end=end_date,
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
# Processing: raw OHLCV -> aligned tensors
# ---------------------------------------------------------------------------

def process_data(df_raw, tradeable=None):
    """
    Align raw OHLCV into tensors. Saves to PROCESSED_DIR.

    Outputs (all torch tensors / pickle):
        ohlcv.pt              - (D, N_all, 5) float32 [Open,High,Low,Close,Volume]
        tradeable_indices.pt  - LongTensor, tradeable ticker positions in ohlcv
        macro_indices.pt      - LongTensor, macro ticker positions in ohlcv
        dates.pkl             - list of datetime.date
        all_tickers.pkl       - ordered list of all ticker names
        tradeable_tickers.pkl - ordered list of tradeable ticker names
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
    print(f"  {common_dates[0].date()} -> {common_dates[-1].date()}")

    # Build OHLCV tensor
    ohlcv = torch.zeros(D, N, 5, dtype=torch.float32)
    for j, ticker in enumerate(all_list):
        sub = ticker_data[ticker].loc[common_dates]
        for k, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
            ohlcv[:, j, k] = torch.from_numpy(sub[col].values.astype(np.float32))

    tradeable_idx = torch.tensor([all_list.index(t) for t in trade_list], dtype=torch.long)
    macro_idx = torch.tensor([all_list.index(t) for t in macro_list], dtype=torch.long)

    dates_list = [d.date() if hasattr(d, 'date') else d for d in common_dates]

    # Save
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

        ohlcv              - (D, N_all, 5) float tensor   [O, H, L, C, V]
        all_tickers        - list[str], length N_all
        tradeable_tickers  - list[str], length N_tradeable
        tradeable_indices  - LongTensor, map tradeable -> ohlcv ticker dim
        macro_indices      - LongTensor, map macro -> ohlcv ticker dim
        dates              - list[datetime.date]
        num_tradeable      - int
        num_all            - int

    train.py owns ALL feature engineering. This just gives you raw OHLCV.
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

def _simulate_day_trades(orders, ohlcv_today, ticker_to_idx):
    """
    Simulate one day of trading with stop-loss/take-profit.

    Each order enters at the Open price. During the day, if the price hits
    the stop-loss or take-profit level (checked against High/Low), the trade
    exits at that level. Otherwise the trade closes at the day's Close.

    Returns (daily_return, trade_results).
    """
    daily_return = 0.0
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

        if np.isnan(entry_price) or np.isnan(close) or entry_price <= 0:
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
            # Both could trigger — heuristic: close direction suggests which hit first
            if direction == "long":
                exit_price = take_profit if close > entry_price else stop_loss
            else:
                exit_price = take_profit if close < entry_price else stop_loss
        elif stop_hit:
            exit_price = stop_loss
        elif limit_hit:
            exit_price = take_profit
        else:
            exit_price = close  # close at end of day (day trading)

        # Compute return
        if direction == "long":
            trade_ret = (exit_price - entry_price) / entry_price
        else:
            trade_ret = (entry_price - exit_price) / entry_price

        # Transaction cost (entry + exit)
        cost = 2 * TRANSACTION_COST_BPS / 10_000
        trade_ret -= cost

        daily_return += weight * trade_ret

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
        })

    return daily_return, trade_results


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


def _compute_metrics(daily_returns, all_trades):
    """Compute strategy metrics from daily returns."""
    if len(daily_returns) == 0:
        return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0,
                "win_rate": 0, "avg_daily_trades": 0}

    rets = np.array(daily_returns)

    # Sharpe ratio (annualized, 252 trading days)
    if len(rets) < 2 or rets.std() < 1e-10:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(252) * rets.mean() / rets.std())

    # Total return
    total_return = float(np.prod(1 + rets) - 1)

    # Max drawdown
    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_drawdown = float(np.min(drawdowns))

    # Win rate (days with positive return)
    win_rate = float(np.mean(rets > 0))

    # Average daily trades
    n_days = len(rets)
    avg_daily_trades = len(all_trades) / n_days if n_days > 0 else 0

    return {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_daily_trades": float(avg_daily_trades),
    }


def _find_date_idx(dates, target_date):
    """Find index of nearest trading day >= target_date."""
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

    Args:
        build_strategy_fn: callable(train_data) -> strategy
        generate_orders_fn: callable(strategy, data, day_idx) -> list[order]
        data: full data dict from load_data()
        train_end_idx: last day index for training (exclusive)
        test_start_idx: first day index for testing
        test_end_idx: last day index for testing (exclusive)

    Returns dict with:
        daily_returns, all_trades, metrics, build_time,
        per_day_trades (list of list of trade dicts per test day)
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
        print(f"  Train: {dates[0]} to {dates[train_end_idx-1]} ({train_end_idx} days)")
        print(f"  Test:  {dates[test_start_idx]} to {dates[test_end_idx-1]} ({test_end_idx - test_start_idx} days)")

    # Build strategy
    t0 = time.time()
    strategy = build_strategy_fn(train_data)
    build_time = time.time() - t0

    if build_time > TIME_BUDGET and verbose:
        print(f"  WARNING: build_strategy took {build_time:.1f}s (budget: {TIME_BUDGET}s)")

    # Run test period
    daily_returns = []
    all_trades = []
    per_day_trades = []
    ohlcv_np = ohlcv.numpy() if isinstance(ohlcv, torch.Tensor) else ohlcv
    tradeable_mask = tradeable_idx.numpy() if isinstance(tradeable_idx, torch.Tensor) else tradeable_idx

    for day in range(test_start_idx, test_end_idx):
        # Generate orders — strategy sees data up to day_idx
        try:
            orders = generate_orders_fn(strategy, data, day)
        except Exception as e:
            if verbose:
                print(f"  Error on {dates[day]}: {e}")
            orders = []

        orders = _validate_orders(orders, data["tradeable_tickers"])

        # Simulate using tradeable tickers' OHLCV
        ohlcv_today = ohlcv_np[day, tradeable_mask]  # (N_tradeable, 5)
        daily_ret, trades = _simulate_day_trades(orders, ohlcv_today, ticker_to_idx)

        daily_returns.append(daily_ret)
        all_trades.extend(trades)
        per_day_trades.append(trades)

    metrics = _compute_metrics(daily_returns, all_trades)
    metrics["build_time"] = build_time
    metrics["train_days"] = train_end_idx
    metrics["test_days"] = test_end_idx - test_start_idx

    return {
        "daily_returns": np.array(daily_returns),
        "all_trades": all_trades,
        "per_day_trades": per_day_trades,
        "metrics": metrics,
        "build_time": build_time,
        "test_dates": dates[test_start_idx:test_end_idx],
    }


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(build_strategy_fn, generate_orders_fn, device="cpu"):
    """
    Evaluate a day-trading strategy using 19 expanding-window 6-month test slices.

    Args:
        build_strategy_fn(train_data: dict) -> strategy: any
            Given training data, build/train a strategy.
            Must complete within TIME_BUDGET (300s).

            train_data contains:
                ohlcv             - (D_train, N_all, 5) OHLCV tensor
                dates             - list[datetime.date] for training period
                all_tickers       - list[str]
                tradeable_tickers - list[str]
                tradeable_indices - LongTensor
                macro_indices     - LongTensor
                num_tradeable     - int
                num_all           - int

        generate_orders_fn(strategy, data: dict, day_idx: int) -> list[dict]
            Called once per trading day. Must return orders for the day.

            strategy  - whatever build_strategy_fn returned
            data      - full data dict (same as load_data() output)
                        only data["ohlcv"][:day_idx] is "known" history
                        data["ohlcv"][day_idx, :, O] is today's open (available)
                        data["ohlcv"][day_idx, :, H/L/C] is FUTURE — do not peek!
            day_idx   - current day index in the full dataset

            Each order dict:
                "ticker"      - str, must be in tradeable_tickers
                "direction"   - "long" or "short"
                "weight"      - float in (0, 1], fraction of portfolio
                "stop_loss"   - float, price level for stop-loss
                "take_profit" - float, price level for take-profit

            Total weight across all orders must be <= 1.0.
            All trades are day trades — positions close at end of day.

    Returns:
        dict with averaged metrics across all slices:
            sharpe_ratio    - annualized Sharpe (PRIMARY METRIC)
            total_return    - average cumulative return per slice
            max_drawdown    - average max drawdown per slice
            win_rate        - average fraction of profitable days
            avg_daily_trades - average trades per day
            per_slice       - list of per-slice metric dicts
    """
    data = load_data(device="cpu")
    dates = data["dates"]
    n_days = len(dates)

    # Compute 6-month slice boundaries from the data start date
    start_date = dates[0]
    boundaries = []
    d = start_date
    while True:
        idx = _find_date_idx(dates, d)
        if idx >= n_days:
            boundaries.append(n_days)
            break
        boundaries.append(idx)
        d = d + relativedelta(months=SLICE_MONTHS)

    # Ensure we have the final boundary
    if boundaries[-1] < n_days:
        boundaries.append(n_days)

    # Slice k: train=[0, boundaries[k+1]), test=[boundaries[k+1], boundaries[k+2])
    num_slices = min(NUM_SLICES, len(boundaries) - 2)

    print(f"Evaluating {num_slices} expanding-window slices...")
    print(f"Data: {n_days} trading days from {dates[0]} to {dates[-1]}")

    all_slice_metrics = []

    for k in range(num_slices):
        train_end = boundaries[k + 1]
        test_start = boundaries[k + 1]
        test_end = boundaries[k + 2] if k + 2 < len(boundaries) else n_days

        if test_start >= test_end or test_start >= n_days:
            continue

        print(f"\n--- Slice {k+1}/{num_slices} ---")

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
        metrics["test_end"] = str(dates[min(test_end - 1, n_days - 1)])

        all_slice_metrics.append(metrics)
        print(f"  Sharpe: {metrics['sharpe_ratio']:.4f} | "
              f"Return: {metrics['total_return']*100:.2f}% | "
              f"MaxDD: {metrics['max_drawdown']*100:.2f}% | "
              f"WinRate: {metrics['win_rate']*100:.1f}%")

    if not all_slice_metrics:
        print("ERROR: No valid slices found.")
        return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0,
                "win_rate": 0, "avg_daily_trades": 0, "per_slice": []}

    # Average across slices
    avg_metrics = {
        "sharpe_ratio": float(np.mean([m["sharpe_ratio"] for m in all_slice_metrics])),
        "total_return": float(np.mean([m["total_return"] for m in all_slice_metrics])),
        "max_drawdown": float(np.mean([m["max_drawdown"] for m in all_slice_metrics])),
        "win_rate": float(np.mean([m["win_rate"] for m in all_slice_metrics])),
        "avg_daily_trades": float(np.mean([m["avg_daily_trades"] for m in all_slice_metrics])),
        "per_slice": all_slice_metrics,
        "num_slices": len(all_slice_metrics),
    }

    print(f"\n{'='*60}")
    print(f"AVERAGE across {len(all_slice_metrics)} slices:")
    print(f"  Sharpe:     {avg_metrics['sharpe_ratio']:.4f}")
    print(f"  Return:     {avg_metrics['total_return']*100:.2f}%")
    print(f"  MaxDD:      {avg_metrics['max_drawdown']*100:.2f}%")
    print(f"  WinRate:    {avg_metrics['win_rate']*100:.1f}%")
    print(f"  Avg Trades: {avg_metrics['avg_daily_trades']:.1f}/day")

    return avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python prepare.py download   # download and process 10y data")
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
        print(f"  Days      : {data['ohlcv'].shape[0]}  ({data['dates'][0]} -> {data['dates'][-1]})")
        print(f"\nDone! Ready to train.")

    elif cmd == "eval":
        from train import build_strategy, generate_orders
        results = evaluate(build_strategy, generate_orders)

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python prepare.py [download|eval]")
