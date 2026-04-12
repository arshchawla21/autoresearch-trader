#!/usr/bin/env python3
"""
prepare.py — Forex Data Download & Fixed Backtesting Harness (v2)
===================================================================
READ-ONLY: The AI agent must NOT modify this file.

What changed from v1:
  - Primary data source is now the **OANDA v20 practice REST API**,
    which gives us 1+ year of 15m candles (vs yfinance's 60-day cap).
  - Economic calendar is pulled from **Finnhub** and exposed to
    strategies as a "_CALENDAR" key in the prices dict. Contains US/JP
    high & medium impact events with timestamps.
  - **Spread cost** of 0.8 pips is deducted from every round-trip trade,
    so strategy Sharpe now reflects realistic retail friction.
  - ^VIX is **synthesized** from USD/JPY 20-bar realized vol (annualized),
    because OANDA doesn't quote the VIX directly. It behaves like VIX for
    regime filtering purposes (the signal is "recent volatility" not
    literally "the equity vol index").
  - Warmup is now 90 trading days, eval is the remaining ~170 — so ML
    models have 3× the training data and out-of-sample is statistically
    meaningful.

Credentials are loaded from a gitignored `.env` file next to this script:
    OANDA_API_TOKEN=...
    FINNHUB_API_KEY=...

The strategy API in train.py is unchanged:
    trade(prices: dict[str, pd.DataFrame]) -> dict
        Returns {"direction": int in {-1, 0, 1}, "tp_pips": float, "sl_pips": float}

New keys strategies may read from `prices`:
    "_CALENDAR" — pd.DataFrame indexed by event time (UTC), columns:
                  ["country", "event", "impact"]. PASSED THROUGH UNSLICED,
                  so strategies can peek at upcoming events (no lookahead
                  on prices — only the schedule is visible, not actuals).
    "NAS100_USD", "BCO_USD" — bonus OANDA-native indicators.

Usage:
    uv run prepare.py              # download (cached) + run backtest
    uv run prepare.py --download   # only download / refresh data
    uv run prepare.py --eval       # only run backtest (data must exist)
    uv run prepare.py --force-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# ENV LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

OANDA_TOKEN = os.environ.get("OANDA_API_TOKEN", "")
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

# Legacy key → OANDA instrument mapping.
# We keep the old key names ("JPY=X", "^TNX", ...) so strategies written
# against v1 still work; under the hood they're backed by OANDA data.
OANDA_INSTRUMENTS: dict[str, str] = {
    "JPY=X":      "USD_JPY",      # the tradeable pair
    "DX-Y.NYB":   "USD_CHF",      # DXY proxy (CHF is in the DXY basket, same sign)
    "^TNX":       "USB10Y_USD",   # US 10Y T-note futures
    "GC=F":       "XAU_USD",      # Gold spot
    "CL=F":       "WTICO_USD",    # WTI crude
    "^N225":      "JP225_USD",    # Nikkei 225 CFD
    "SPY":        "SPX500_USD",   # S&P 500 CFD
    "NAS100_USD": "NAS100_USD",   # Nasdaq 100 CFD (bonus, native key)
    "BCO_USD":    "BCO_USD",      # Brent crude (bonus, native key)
}

PAIR = "JPY=X"
TRADEABLE = [PAIR]

INDICATORS = [
    "DX-Y.NYB",
    "^TNX",
    "GC=F",
    "CL=F",
    "^VIX",         # synthetic — computed from USD/JPY realized vol
    "^N225",
    "SPY",
    "NAS100_USD",
    "BCO_USD",
]

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INTERVAL = "15m"
OANDA_GRANULARITY = "M15"
LOOKBACK_DAYS = 365
HISTORY_DAYS = 90        # 90 trading days warmup, rest is eval
CACHE_DIR = Path.home() / ".cache" / "autoresearch-trader"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE_ANNUAL = 0.05
DEFAULT_TP_PIPS = 15.0
DEFAULT_SL_PIPS = 10.0

# Round-trip transaction cost in pips. 0.8 pips ≈ realistic retail spread
# for USD/JPY. Deducted from every trade's PnL on exit.
SPREAD_PIPS = 0.8

OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
FINNHUB_URL = "https://finnhub.io/api/v1"


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


# ═══════════════════════════════════════════════════════════════════════════════
# OANDA DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def _oanda_fetch_instrument(
    instrument: str,
    start: datetime,
    end: datetime,
    granularity: str = OANDA_GRANULARITY,
) -> pd.DataFrame | None:
    """Paginate OANDA candles backwards from `end` until reaching `start`."""
    if not OANDA_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {OANDA_TOKEN}",
        "Accept-Datetime-Format": "RFC3339",
    }
    all_candles: list[dict] = []
    cursor = end
    # 1y of 15m = ~35k bars; 5000/call → 7 calls. Cap at 20 for safety.
    MAX_PAGES = 20
    for _ in range(MAX_PAGES):
        if cursor <= start:
            break
        params = {
            "granularity": granularity,
            "count": 5000,
            "to": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M",
        }
        try:
            resp = requests.get(
                f"{OANDA_PRACTICE_URL}/instruments/{instrument}/candles",
                headers=headers,
                params=params,
                timeout=30,
            )
        except Exception as e:
            print(f"    [!] {instrument} request error: {e}")
            return None
        if resp.status_code != 200:
            print(f"    [!] {instrument} HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        payload = resp.json()
        candles = payload.get("candles", [])
        if not candles:
            break
        all_candles = candles + all_candles
        oldest = pd.Timestamp(candles[0]["time"])
        if oldest.tzinfo is None:
            oldest = oldest.tz_localize("UTC")
        if oldest.to_pydatetime() >= cursor:
            break  # no progress, pagination exhausted
        cursor = oldest.to_pydatetime()
        time.sleep(0.05)  # be polite

    if not all_candles:
        return None

    rows = []
    for c in all_candles:
        if not c.get("complete", True):
            continue
        mid = c.get("mid", {})
        try:
            rows.append({
                "time":   pd.Timestamp(c["time"]),
                "open":   float(mid["o"]),
                "high":   float(mid["h"]),
                "low":    float(mid["l"]),
                "close":  float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })
        except (KeyError, ValueError):
            continue
    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("time").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="first")]

    start_ts = pd.Timestamp(start)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")
    return df[(df.index >= start_ts) & (df.index <= end_ts)]


def _synthesize_vix(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a synthetic VIX-like index from USD/JPY realized vol.
    20-bar rolling std of log returns, annualized, scaled to VIX units.
    """
    closes = pair_df["close"].astype(float)
    log_ret = np.log(closes / closes.shift(1))
    rv = log_ret.rolling(20, min_periods=5).std()
    ann_factor = np.sqrt(96 * 252) * 100   # VIX is in "annualized %"
    vix_like = (rv * ann_factor).astype(float)
    out = pd.DataFrame({
        "open":  vix_like,
        "high":  vix_like,
        "low":   vix_like,
        "close": vix_like,
        "volume": 0,
    })
    out.index = pair_df.index
    return out.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# FINNHUB ECONOMIC CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_calendar(start: datetime, end: datetime) -> pd.DataFrame | None:
    """
    Pull US + JP high/medium impact events from Finnhub.
    Finnhub caps calendar queries at ~1 year, so we fetch in chunks to be safe.
    """
    if not FINNHUB_KEY:
        return None

    url = f"{FINNHUB_URL}/calendar/economic"
    rows: list[dict] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=180), end)
        params = {
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "token": FINNHUB_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
        except Exception as e:
            print(f"  [!] Finnhub request error: {e}")
            return None
        if resp.status_code != 200:
            print(f"  [!] Finnhub HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        payload = resp.json() or {}
        events = payload.get("economicCalendar", []) or []
        for e in events:
            country = str(e.get("country", "")).upper()
            if country not in ("US", "JP"):
                continue
            impact = str(e.get("impact", "")).lower()
            if impact not in ("high", "medium"):
                continue
            t = e.get("time") or ""
            if not t:
                continue
            try:
                ts = pd.Timestamp(t)
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
            except Exception:
                continue
            rows.append({
                "time": ts,
                "country": country,
                "event": str(e.get("event", ""))[:80],
                "impact": impact,
            })
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.2)

    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(legacy_key: str, cache_key: str) -> Path:
    safe = legacy_key.replace("^", "_").replace("=", "_").replace("-", "_")
    return CACHE_DIR / f"oanda_{safe}_{cache_key}.parquet"


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    end_date = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    cache_key = hashlib.md5(
        f"oanda_{start_str}_{end_str}_{OANDA_GRANULARITY}_{'_'.join(sorted(OANDA_INSTRUMENTS.values()))}".encode()
    ).hexdigest()[:12]
    cache_meta = CACHE_DIR / f"meta_{cache_key}.json"
    calendar_path = CACHE_DIR / f"calendar_{cache_key}.parquet"

    data: dict[str, pd.DataFrame] = {}

    if not force and cache_meta.exists():
        print(f"[✓] Loading cached OANDA data ({cache_key})...")
        all_good = True
        for legacy_key in OANDA_INSTRUMENTS:
            p = _cache_path(legacy_key, cache_key)
            if p.exists():
                data[legacy_key] = pd.read_parquet(p)
            else:
                all_good = False
                break
        if all_good and PAIR in data:
            data["^VIX"] = _synthesize_vix(data[PAIR])
            if calendar_path.exists():
                data["_CALENDAR"] = pd.read_parquet(calendar_path)
            print(f"    Loaded {len(data)} keys from cache (incl. synthetic VIX + calendar).")
            return data
        print("    Cache incomplete — re-downloading...")
        data = {}

    if not OANDA_TOKEN:
        print("[✗] No OANDA_API_TOKEN in environment. Put it in .env.")
        sys.exit(1)

    print(f"[↓] Downloading {len(OANDA_INSTRUMENTS)} instruments from OANDA practice API")
    print(f"    Range: {start_str} → {end_str}, granularity={OANDA_GRANULARITY}")
    for legacy_key, instr in OANDA_INSTRUMENTS.items():
        print(f"  → {legacy_key:<12} ({instr:<12})...", end=" ", flush=True)
        df = _oanda_fetch_instrument(instr, start_date, end_date)
        if df is not None and len(df) > 0:
            data[legacy_key] = df
            df.to_parquet(_cache_path(legacy_key, cache_key))
            print(f"OK ({len(df)} rows)")
        else:
            print("FAILED")

    if PAIR not in data:
        print(f"[✗] {PAIR} missing — aborting.")
        sys.exit(1)

    data["^VIX"] = _synthesize_vix(data[PAIR])
    print(f"  → ^VIX         (synthetic)  ... OK ({len(data['^VIX'])} rows)")

    cal = _fetch_calendar(start_date, end_date)
    if cal is not None and len(cal) > 0:
        data["_CALENDAR"] = cal
        cal.to_parquet(calendar_path)
        n_us = int((cal["country"] == "US").sum())
        n_jp = int((cal["country"] == "JP").sum())
        n_hi = int((cal["impact"] == "high").sum())
        print(f"  → _CALENDAR                ... OK ({len(cal)} events: US={n_us} JP={n_jp} high={n_hi})")
    else:
        print(f"  → _CALENDAR                ... (none / no key)")

    cache_meta.write_text(json.dumps({
        "start": start_str,
        "end": end_str,
        "interval": INTERVAL,
        "source": "oanda",
        "spread_pips": SPREAD_PIPS,
        "history_days": HISTORY_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "symbols": list(data.keys()),
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    print(f"\n[✓] Downloaded {len(data)} total keys.")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE — TP / SL WITH SPREAD COST
# ═══════════════════════════════════════════════════════════════════════════════

def _unique_trading_days(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted({ts.normalize() for ts in df.index})


def _slice_prices(
    data: dict[str, pd.DataFrame], upto: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """
    Slice each price DataFrame to index <= upto. The "_CALENDAR" key is
    passed through untouched — strategies are allowed to see the *schedule*
    of upcoming events (not their outcomes, which we never store).
    """
    out: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        if sym == "_CALENDAR":
            out[sym] = df
        else:
            out[sym] = df.loc[df.index <= upto]
    return out


def _resolve_exit(position: dict, bar_high: float, bar_low: float) -> float | None:
    direction = position["direction"]
    tp_price = position["tp_price"]
    sl_price = position["sl_price"]
    if direction == 1:
        hit_tp = bar_high >= tp_price
        hit_sl = bar_low <= sl_price
    else:
        hit_tp = bar_low <= tp_price
        hit_sl = bar_high >= sl_price
    if hit_sl and hit_tp:
        return sl_price  # conservative: assume SL first
    if hit_tp:
        return tp_price
    if hit_sl:
        return sl_price
    return None


def run_backtest(data: dict[str, pd.DataFrame]) -> dict:
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
        sys.exit(1)

    eval_start_day = trading_days[HISTORY_DAYS]
    if pair_df.index.tz is not None and eval_start_day.tz is None:
        eval_start_day = eval_start_day.tz_localize(pair_df.index.tz)
    eval_mask = pair_df.index >= eval_start_day
    eval_indices = np.where(eval_mask)[0]

    if len(eval_indices) < 2:
        print("[✗] Not enough eval bars.")
        sys.exit(1)

    n_cal = len(data["_CALENDAR"]) if "_CALENDAR" in data else 0
    print(f"\n{'='*70}")
    print(f"BACKTEST — {PAIR} — {INTERVAL}  (OANDA 1y + spread {SPREAD_PIPS}p)")
    print(f"{'='*70}")
    print(f"  Total trading days : {len(trading_days)}")
    print(f"  Warmup             : days 1–{HISTORY_DAYS} (up to {eval_start_day.date()})")
    print(f"  Eval window        : days {HISTORY_DAYS+1}–{len(trading_days)}  ({len(eval_indices)} bars)")
    print(f"  Pip size           : {pip}")
    print(f"  Default TP / SL    : {DEFAULT_TP_PIPS} / {DEFAULT_SL_PIPS} pips")
    print(f"  Spread cost        : {SPREAD_PIPS} pips / trade (round trip)")
    print(f"  Calendar events    : {n_cal}")
    print(f"{'='*70}\n")

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
                # Spread cost as fraction of entry price
                spread_cost_frac = (SPREAD_PIPS * pip) / position["entry_price"]
                raw_bar_ret = (
                    position["direction"]
                    * (exit_price - position["last_mark"])
                    / position["last_mark"]
                )
                bar_ret = raw_bar_ret - spread_cost_frac
                raw_pnl = (
                    position["direction"]
                    * (exit_price - position["entry_price"])
                    / position["entry_price"]
                )
                net_pnl = raw_pnl - spread_cost_frac
                trade_records.append({
                    "entry_time": position["entry_time"],
                    "exit_time": current_time,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "tp_pips": position["tp_pips"],
                    "sl_pips": position["sl_pips"],
                    "pnl": net_pnl,
                    "pnl_gross": raw_pnl,
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

        if (step + 1) % 2000 == 0 or step == 0:
            cum = float(np.prod([1 + r for r in bar_returns]) - 1)
            print(f"    Bar {step+1:>6}/{n_bars}  |  trades={len(trade_records):>5}  |  cum={cum:+.4%}")

    if position is not None:
        last_i = eval_indices[-1]
        last_close = float(pair_df.iloc[last_i]["close"])
        last_time = pair_df.index[last_i]
        spread_cost_frac = (SPREAD_PIPS * pip) / position["entry_price"]
        raw_pnl = (
            position["direction"]
            * (last_close - position["entry_price"])
            / position["entry_price"]
        )
        trade_records.append({
            "entry_time": position["entry_time"],
            "exit_time": last_time,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": last_close,
            "tp_pips": position["tp_pips"],
            "sl_pips": position["sl_pips"],
            "pnl": raw_pnl - spread_cost_frac,
            "pnl_gross": raw_pnl,
            "outcome": "eof",
        })
        position = None

    # ─── Metrics ────────────────────────────────────────────────────────
    returns = np.array(bar_returns, dtype=float)
    cum_curve = np.cumprod(1 + returns)
    total_return = float(cum_curve[-1] - 1) if len(cum_curve) else 0.0

    eval_days_unique = sorted({t.normalize() for t in bar_times})
    bars_per_day = len(returns) / max(1, len(eval_days_unique))
    bars_per_year = bars_per_day * 252
    rf_per_bar = RISK_FREE_RATE_ANNUAL / bars_per_year if bars_per_year > 0 else 0.0

    mean_ret = float(np.mean(returns)) if len(returns) else 0.0
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float((mean_ret - rf_per_bar) / std_ret * np.sqrt(bars_per_year))
        if std_ret > 0 else 0.0
    )
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float((mean_ret - rf_per_bar) / downside_std * np.sqrt(bars_per_year))
        if downside_std > 0 else 0.0
    )

    peak = np.maximum.accumulate(cum_curve) if len(cum_curve) else np.array([1.0])
    drawdown = (cum_curve - peak) / peak if len(cum_curve) else np.array([0.0])
    max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0
    calmar = float(total_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-9 else 0.0

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
        "spread_pips": SPREAD_PIPS,
        "eval_start": str(eval_days_unique[0].date()) if eval_days_unique else "",
        "eval_end": str(eval_days_unique[-1].date()) if eval_days_unique else "",
    }

    print(f"\n{'='*70}")
    print(f"RESULTS — {PAIR}   (net of {SPREAD_PIPS}p spread)")
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
    parser = argparse.ArgumentParser(description="autoresearch-trader: forex data + eval harness (OANDA)")
    parser.add_argument("--download", action="store_true", help="Only download data (skip eval)")
    parser.add_argument("--eval", action="store_true", help="Only run eval (data must exist)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached")
    args = parser.parse_args()

    if args.eval:
        return run_backtest(download_all(force=False))
    if args.download:
        download_all(force=args.force_download)
        return None
    data = download_all(force=args.force_download)
    return run_backtest(data)


if __name__ == "__main__":
    main()
