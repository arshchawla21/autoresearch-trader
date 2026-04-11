#!/usr/bin/env python3
"""
prepare.py — Forex Data Download & Fixed Backtesting Harness
=============================================================
READ-ONLY: The AI agent must NOT modify this file.

What it does:
  1. Downloads ~60 days of 15-minute OHLCV candles for USD/JPY (the sole
     tradeable pair) plus a set of macro / commodity indicators the strategy
     can read (DXY, US yields, gold, oil, VIX, Nikkei, bonds, S&P).
  2. Caches the data to ~/.cache/autoresearch-trader/ so we don't re-download
     every run.
  3. Provides a deterministic TP/SL backtesting evaluation that:
       - Uses days 1–30 as "history" the strategy can see from the start.
       - Steps through days 31..end candle by candle (15m bars).
       - At each 15m bar, if the bot is FLAT it calls
             train.trade(prices_so_far) -> {"direction", "tp_pips", "sl_pips"}
         and — if direction is non-zero — opens a fixed-size position at that
         bar's close.
       - While a position is open, the harness ignores trade() and walks the
         pair's high/low against the stored TP and SL levels. Ambiguous bars
         (both levels breached) resolve to SL (conservative).
       - Position size is fixed (notional = 1.0). The strategy only has to
         predict direction — it never sizes trades.
       - PnL is pure price-change: direction × (exit - entry) / entry.
       - Transaction costs are NOT modelled. Forex spreads are tiny relative
         to 15 pip TP levels, and this lets us focus on signal quality.

Usage:
    uv run prepare.py              # download data + run backtest
    uv run prepare.py --download   # only download / refresh data
    uv run prepare.py --eval       # only run backtest (data must exist)

The strategy must live in train.py and expose:
    trade(prices: dict[str, pd.DataFrame]) -> dict
        Returns {"direction": int in {-1, 0, 1}, "tp_pips": float, "sl_pips": float}

        - direction = 1  → go long  USD/JPY at this bar's close
        - direction = -1 → go short USD/JPY at this bar's close
        - direction = 0  → stay flat
        - tp_pips / sl_pips → take-profit / stop-loss distance in pips.
          For USD/JPY one pip = 0.01 in price (so 15 pips = 0.15 yen).
          If omitted, the harness defaults to 15 / 10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# --- Tradeable pair (single instrument — the strategy goes long/short/flat) ---
PAIR = "JPY=X"                # yfinance ticker for spot USD/JPY
TRADEABLE = [PAIR]

# --- Macro & commodity context indicators (read-only) ---
# These were chosen because they are the dominant drivers of USD/JPY flow:
#   - DX-Y.NYB : broad USD index — direct USD leg of the pair
#   - ^TNX     : US 10Y yield — yield differential drives carry trades
#   - GC=F     : gold futures — safe-haven inverse-USD signal
#   - CL=F     : WTI crude — inflation / risk proxy
#   - ^VIX     : risk sentiment — JPY is the classic risk-off haven
#   - ^N225    : Nikkei 225 — Japan equity flows, BoJ policy proxy
#   - TLT      : US 20Y+ treasury ETF — long-duration rate expectations
#   - SPY      : S&P 500 ETF — global risk-on proxy
INDICATORS = [
    "DX-Y.NYB",
    "^TNX",
    "GC=F",
    "CL=F",
    "^VIX",
    "^N225",
    "TLT",
    "SPY",
]

ALL_SYMBOLS = TRADEABLE + INDICATORS

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INTERVAL = "15m"          # 15-minute candles
LOOKBACK_DAYS = 60        # yfinance caps 15m data at ~60 calendar days
HISTORY_DAYS = 30         # first 30 trading days = warmup / training data
CACHE_DIR = Path.home() / ".cache" / "autoresearch-trader"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE_ANNUAL = 0.05  # 5% annual for Sharpe calc

DEFAULT_TP_PIPS = 15.0    # used if strategy omits tp_pips
DEFAULT_SL_PIPS = 10.0    # used if strategy omits sl_pips


def pip_size(pair: str) -> float:
    """Pip size for a currency pair. JPY crosses quote to 2 decimals → 0.01."""
    return 0.01 if "JPY" in pair.upper() else 0.0001


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _download_symbol(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Download 15-min OHLCV for a single symbol. Returns None on failure."""
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
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[keep]
                return df
        except Exception as e:
            print(f"  [!] {symbol} attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download (or load from cache) ~60 days of 15-min OHLCV data for
    USD/JPY + macro indicators.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    cache_key = hashlib.md5(
        f"forex_{start_str}_{end_str}_{INTERVAL}_{'_'.join(ALL_SYMBOLS)}".encode()
    ).hexdigest()[:12]
    cache_meta = CACHE_DIR / f"meta_{cache_key}.json"

    data: dict[str, pd.DataFrame] = {}

    if not force and cache_meta.exists():
        print(f"[✓] Loading cached forex data ({cache_key})...")
        all_good = True
        for sym in ALL_SYMBOLS:
            parquet = CACHE_DIR / f"{sym.replace('^', '_').replace('=', '_')}_{cache_key}.parquet"
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
            parquet = CACHE_DIR / f"{sym.replace('^', '_').replace('=', '_')}_{cache_key}.parquet"
            df.to_parquet(parquet)
            print(f"OK ({len(df)} rows)")
        else:
            print("FAILED — will be excluded")

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
    if PAIR not in data:
        print(f"[✗] Tradeable pair {PAIR} missing — cannot run backtest.")
        sys.exit(1)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE — POSITION MANAGEMENT WITH TP / SL
# ═══════════════════════════════════════════════════════════════════════════════

def _unique_trading_days(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Sorted list of unique calendar dates present in the index."""
    return sorted({ts.normalize() for ts in df.index})


def _slice_prices(
    data: dict[str, pd.DataFrame], upto: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Return each symbol's DataFrame sliced to index <= upto."""
    out: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        out[sym] = df.loc[df.index <= upto]
    return out


def _resolve_exit(
    position: dict, bar_high: float, bar_low: float
) -> float | None:
    """
    Check if the current bar's high/low breaches TP or SL.
    Returns the exit price if the position should close, else None.
    Conservative: if both are breached in the same bar, assume SL first.
    """
    direction = position["direction"]
    tp_price = position["tp_price"]
    sl_price = position["sl_price"]

    if direction == 1:
        hit_tp = bar_high >= tp_price
        hit_sl = bar_low <= sl_price
    else:  # direction == -1
        hit_tp = bar_low <= tp_price
        hit_sl = bar_high >= sl_price

    if hit_sl and hit_tp:
        return sl_price
    if hit_tp:
        return tp_price
    if hit_sl:
        return sl_price
    return None


def run_backtest(data: dict[str, pd.DataFrame]) -> dict:
    """
    Fixed TP/SL backtest for USD/JPY.

    Flow per 15m bar in the eval window:
        1. If a position is open, check this bar's high/low vs TP/SL.
             - If breached, exit at the breached level, record the trade,
               and this bar's return is the exit-minus-last-mark leg.
             - Else, mark-to-market to this bar's close.
        2. If the bot is now flat, call train.trade(prices_so_far) with all
           data up to and including this bar's close. If it returns a
           non-zero direction, open a fresh position at this bar's close.
           TP/SL checks for that position begin on the *next* bar.

    Metrics:
        - Sharpe / Sortino / total return / max drawdown → per-bar returns
        - Win rate / profit factor / n_trades → per-trade outcomes
    """
    try:
        import train
    except ImportError:
        print("[✗] Cannot import train.py — make sure it exists in cwd.")
        sys.exit(1)

    if PAIR not in data:
        print(f"[✗] {PAIR} missing from data.")
        sys.exit(1)

    pair_df = data[PAIR].sort_index().copy()
    pair_df = pair_df[~pair_df.index.duplicated(keep="first")]
    pip = pip_size(PAIR)

    trading_days = _unique_trading_days(pair_df)
    if len(trading_days) < HISTORY_DAYS + 2:
        print(f"[✗] Only {len(trading_days)} trading days found, need >{HISTORY_DAYS+1}.")
        print("    Try running with --download to refresh data.")
        sys.exit(1)

    eval_start_day = trading_days[HISTORY_DAYS]
    # yfinance forex data is tz-aware — match it
    if pair_df.index.tz is not None and eval_start_day.tz is None:
        eval_start_day = eval_start_day.tz_localize(pair_df.index.tz)
    eval_mask = pair_df.index >= eval_start_day
    eval_indices = np.where(eval_mask)[0]

    if len(eval_indices) < 2:
        print("[✗] Not enough eval bars.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"BACKTEST — {PAIR} — {INTERVAL}")
    print(f"{'='*70}")
    print(f"  Total trading days : {len(trading_days)}")
    print(f"  Warmup (history)   : days 1–{HISTORY_DAYS} (up to {eval_start_day.date()})")
    print(f"  Eval window        : days {HISTORY_DAYS+1}–{len(trading_days)}  ({len(eval_indices)} bars)")
    print(f"  Pip size           : {pip}")
    print(f"  Default TP / SL    : {DEFAULT_TP_PIPS} / {DEFAULT_SL_PIPS} pips")
    print(f"  Indicators         : {', '.join(s for s in INDICATORS if s in data)}")
    print(f"{'='*70}\n")

    # ─── Main loop ──────────────────────────────────────────────────────
    position: dict | None = None
    bar_returns: list[float] = []
    bar_times: list[pd.Timestamp] = []
    trade_records: list[dict] = []

    n_bars = len(eval_indices)
    for step, i in enumerate(eval_indices):
        row = pair_df.iloc[i]
        current_time = pair_df.index[i]
        close_px = float(row["close"])
        high_px = float(row["high"])
        low_px = float(row["low"])

        bar_ret = 0.0

        # ─── Phase 1: manage existing position ─────────────────────────
        if position is not None:
            exit_price = _resolve_exit(position, high_px, low_px)
            if exit_price is not None:
                bar_ret = (
                    position["direction"]
                    * (exit_price - position["last_mark"])
                    / position["last_mark"]
                )
                trade_records.append({
                    "entry_time": position["entry_time"],
                    "exit_time": current_time,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "tp_pips": position["tp_pips"],
                    "sl_pips": position["sl_pips"],
                    "pnl": position["direction"]
                    * (exit_price - position["entry_price"])
                    / position["entry_price"],
                    "outcome": "tp" if exit_price == position["tp_price"] else "sl",
                })
                position = None
            else:
                bar_ret = (
                    position["direction"]
                    * (close_px - position["last_mark"])
                    / position["last_mark"]
                )
                position["last_mark"] = close_px

        bar_returns.append(bar_ret)
        bar_times.append(current_time)

        # ─── Phase 2: open new position if flat ────────────────────────
        if position is None and step < n_bars - 1:
            prices_so_far = _slice_prices(data, current_time)
            try:
                signal = train.trade(prices_so_far)
            except Exception as e:
                print(f"  [!] trade() raised at step {step}: {e}")
                signal = {"direction": 0}

            if not isinstance(signal, dict):
                signal = {"direction": 0}

            direction = int(signal.get("direction", 0))
            if direction in (-1, 1):
                tp_pips = float(signal.get("tp_pips", DEFAULT_TP_PIPS))
                sl_pips = float(signal.get("sl_pips", DEFAULT_SL_PIPS))
                if tp_pips <= 0 or sl_pips <= 0:
                    tp_pips = DEFAULT_TP_PIPS
                    sl_pips = DEFAULT_SL_PIPS
                entry_price = close_px
                position = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "entry_time": current_time,
                    "tp_price": entry_price + direction * tp_pips * pip,
                    "sl_price": entry_price - direction * sl_pips * pip,
                    "last_mark": entry_price,
                    "tp_pips": tp_pips,
                    "sl_pips": sl_pips,
                }

        if (step + 1) % 500 == 0 or step == 0:
            cum = float(np.prod([1 + r for r in bar_returns]) - 1)
            print(f"    Bar {step+1:>5}/{n_bars}  |  trades={len(trade_records):>4}  |  cum={cum:+.4%}")

    # Close any still-open position at the last bar's close
    if position is not None:
        last_i = eval_indices[-1]
        last_close = float(pair_df.iloc[last_i]["close"])
        last_time = pair_df.index[last_i]
        trade_records.append({
            "entry_time": position["entry_time"],
            "exit_time": last_time,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": last_close,
            "tp_pips": position["tp_pips"],
            "sl_pips": position["sl_pips"],
            "pnl": position["direction"]
            * (last_close - position["entry_price"])
            / position["entry_price"],
            "outcome": "eof",
        })
        position = None

    # ─── Metrics ────────────────────────────────────────────────────────
    returns = np.array(bar_returns, dtype=float)
    cum_curve = np.cumprod(1 + returns)
    total_return = float(cum_curve[-1] - 1) if len(cum_curve) else 0.0

    # Annualisation from observed bar density
    eval_days_unique = sorted({t.normalize() for t in bar_times})
    bars_per_day = len(returns) / max(1, len(eval_days_unique))
    bars_per_year = bars_per_day * 252
    rf_per_bar = RISK_FREE_RATE_ANNUAL / bars_per_year if bars_per_year > 0 else 0.0

    mean_ret = float(np.mean(returns)) if len(returns) else 0.0
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float((mean_ret - rf_per_bar) / std_ret * np.sqrt(bars_per_year))
        if std_ret > 0
        else 0.0
    )
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float((mean_ret - rf_per_bar) / downside_std * np.sqrt(bars_per_year))
        if downside_std > 0
        else 0.0
    )

    peak = np.maximum.accumulate(cum_curve) if len(cum_curve) else np.array([1.0])
    drawdown = (cum_curve - peak) / peak if len(cum_curve) else np.array([0.0])
    max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0
    calmar = (
        float(total_return / abs(max_drawdown))
        if abs(max_drawdown) > 1e-9
        else 0.0
    )

    # Trade-level metrics
    n_trades = len(trade_records)
    if n_trades > 0:
        trade_pnls = np.array([t["pnl"] for t in trade_records])
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        win_rate = float(len(wins) / n_trades)
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 1e-12 else (float("inf") if gross_profit > 0 else 0.0)
        )
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else 0.0
        tp_hits = sum(1 for t in trade_records if t["outcome"] == "tp")
        sl_hits = sum(1 for t in trade_records if t["outcome"] == "sl")
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0.0
        tp_hits = sl_hits = 0

    trades_per_day = n_trades / max(1, len(eval_days_unique))

    metrics = {
        "pair": PAIR,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n_trades": n_trades,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "trades_per_day": trades_per_day,
        "bars_per_day": bars_per_day,
        "eval_start": str(eval_days_unique[0].date()) if eval_days_unique else "",
        "eval_end": str(eval_days_unique[-1].date()) if eval_days_unique else "",
    }

    print(f"\n{'='*70}")
    print(f"RESULTS — {PAIR}")
    print(f"{'='*70}")
    print(f"  Total Return      : {total_return:+.4%}")
    print(f"  Sharpe Ratio      : {sharpe:.4f}")
    print(f"  Sortino Ratio     : {sortino:.4f}")
    print(f"  Calmar Ratio      : {calmar:.4f}")
    print(f"  Max Drawdown      : {max_drawdown:.4%}")
    print(f"  Win Rate          : {win_rate:.2%}   (TP {tp_hits} / SL {sl_hits})")
    print(f"  Profit Factor     : {profit_factor:.4f}")
    print(f"  Avg Win           : {avg_win:+.5%}")
    print(f"  Avg Loss          : {avg_loss:+.5%}")
    print(f"  # Trades          : {n_trades}  ({trades_per_day:.1f} / day)")
    print(f"  Bars / day        : {bars_per_day:.1f}")
    print(f"  Eval Period       : {metrics['eval_start']} → {metrics['eval_end']}")
    print(f"{'='*70}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="autoresearch-trader: forex data + eval harness")
    parser.add_argument("--download", action="store_true", help="Only download data (skip eval)")
    parser.add_argument("--eval", action="store_true", help="Only run eval (data must exist)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached")
    args = parser.parse_args()

    if args.eval:
        data = download_all(force=False)
        return run_backtest(data)

    if args.download:
        download_all(force=args.force_download)
        return None

    data = download_all(force=args.force_download)
    return run_backtest(data)


if __name__ == "__main__":
    main()
