#!/usr/bin/env python3
"""
backtest.py — Visual Backtesting Tool for USD/JPY 15m
======================================================
Runs the same TP/SL backtest as prepare.py but produces an interactive HTML
chart showing the USD/JPY price path, every trade entry (long = green, short
= red), TP/SL outcomes, and the equity curve.

Usage:
    uv run backtest.py              # default
    uv run backtest.py --png        # placeholder for future PNG export
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import (
    ALL_SYMBOLS,
    DEFAULT_SL_PIPS,
    DEFAULT_TP_PIPS,
    HISTORY_DAYS,
    PAIR,
    _resolve_exit,
    _slice_prices,
    _unique_trading_days,
    download_all,
    pip_size,
)


def run_visual_backtest(data: dict[str, pd.DataFrame]) -> None:
    try:
        import train
    except ImportError:
        print("[✗] Cannot import train.py")
        sys.exit(1)

    if PAIR not in data:
        print(f"[✗] {PAIR} missing from data.")
        sys.exit(1)

    pair_df = data[PAIR].sort_index().copy()
    pair_df = pair_df[~pair_df.index.duplicated(keep="first")]
    pip = pip_size(PAIR)

    trading_days = _unique_trading_days(pair_df)
    if len(trading_days) < HISTORY_DAYS + 2:
        print(f"[✗] Only {len(trading_days)} trading days, need >{HISTORY_DAYS+1}")
        sys.exit(1)

    eval_start_day = trading_days[HISTORY_DAYS]
    if pair_df.index.tz is not None and eval_start_day.tz is None:
        eval_start_day = eval_start_day.tz_localize(pair_df.index.tz)
    eval_mask = pair_df.index >= eval_start_day
    eval_indices = np.where(eval_mask)[0]

    if len(eval_indices) < 2:
        print("[✗] Not enough eval bars.")
        sys.exit(1)

    records: list[dict] = []
    position: dict | None = None
    trade_records: list[dict] = []
    bar_returns: list[float] = []

    n_bars = len(eval_indices)
    print(f"Running {n_bars} eval bars on {PAIR}...")

    for step, i in enumerate(eval_indices):
        row = pair_df.iloc[i]
        current_time = pair_df.index[i]
        close_px = float(row["close"])
        high_px = float(row["high"])
        low_px = float(row["low"])
        bar_ret = 0.0
        exit_event = None

        if position is not None:
            exit_price = _resolve_exit(position, high_px, low_px)
            if exit_price is not None:
                bar_ret = (
                    position["direction"]
                    * (exit_price - position["last_mark"])
                    / position["last_mark"]
                )
                trade = {
                    "entry_time": str(position["entry_time"]),
                    "exit_time": str(current_time),
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": position["direction"]
                    * (exit_price - position["entry_price"])
                    / position["entry_price"],
                    "outcome": "tp" if exit_price == position["tp_price"] else "sl",
                }
                trade_records.append(trade)
                exit_event = trade
                position = None
            else:
                bar_ret = (
                    position["direction"]
                    * (close_px - position["last_mark"])
                    / position["last_mark"]
                )
                position["last_mark"] = close_px

        bar_returns.append(bar_ret)

        entry_event = None
        if position is None and step < n_bars - 1:
            prices_so_far = _slice_prices(data, current_time)
            try:
                signal = train.trade(prices_so_far)
            except Exception:
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
                entry_event = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "tp_price": position["tp_price"],
                    "sl_price": position["sl_price"],
                }

        records.append({
            "time": str(current_time),
            "open": float(row["open"]),
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "bar_ret": bar_ret,
            "entry": entry_event,
            "exit": exit_event,
        })

        if (step + 1) % 500 == 0:
            print(f"  Bar {step+1}/{n_bars}  trades={len(trade_records)}")

    print(f"Done. Total trades: {len(trade_records)}")
    html = _build_html(records, trade_records)
    out_path = Path("backtest_visual.html")
    out_path.write_text(html)
    print(f"\n[✓] Saved to {out_path.resolve()}")
    try:
        webbrowser.open(str(out_path.resolve()))
    except Exception:
        pass


def _build_html(records: list[dict], trades: list[dict]) -> str:
    times = [r["time"] for r in records]
    closes = [r["close"] for r in records]
    highs = [r["high"] for r in records]
    lows = [r["low"] for r in records]

    cum = []
    running = 1.0
    for r in records:
        running *= 1 + r["bar_ret"]
        cum.append(running - 1)

    long_entries = []
    short_entries = []
    tp_exits = []
    sl_exits = []
    for r in records:
        ent = r.get("entry")
        if ent:
            point = {"time": r["time"], "price": ent["entry_price"]}
            if ent["direction"] == 1:
                long_entries.append(point)
            else:
                short_entries.append(point)
        ex = r.get("exit")
        if ex:
            point = {"time": r["time"], "price": ex["exit_price"]}
            if ex["outcome"] == "tp":
                tp_exits.append(point)
            else:
                sl_exits.append(point)

    total_ret = cum[-1] if cum else 0.0
    max_dd = 0.0
    peak = 0.0
    for v in cum:
        if 1 + v > peak:
            peak = 1 + v
        dd = (1 + v - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd
    n_tp = sum(1 for t in trades if t["outcome"] == "tp")
    n_sl = sum(1 for t in trades if t["outcome"] == "sl")
    win_rate = (n_tp / len(trades)) if trades else 0.0

    payload = json.dumps({
        "times": times,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "cum": cum,
        "long_entries": long_entries,
        "short_entries": short_entries,
        "tp_exits": tp_exits,
        "sl_exits": sl_exits,
    })

    pair_label = PAIR
    header_subtitle = (
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"{records[0]['time'][:16]} → {records[-1]['time'][:16]} · "
        f"{len(records)} bars · {len(trades)} trades"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Visual — {pair_label}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: #0d1117; color: #c9d1d9; padding: 20px;
}}
h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 22px; }}
.subtitle {{ color: #8b949e; margin-bottom: 24px; font-size: 13px; }}
.chart-container {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; margin-bottom: 20px;
}}
.chart-container h2 {{
    color: #f0f6fc; font-size: 16px; margin-bottom: 12px;
}}
.chart-wrapper {{ position: relative; height: 340px; }}
.stats-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.stat-card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; min-width: 140px; flex: 1;
}}
.stat-card .label {{ color: #8b949e; font-size: 11px; text-transform: uppercase; }}
.stat-card .value {{ color: #f0f6fc; font-size: 20px; font-weight: 600; margin-top: 4px; }}
.stat-card .value.positive {{ color: #3fb950; }}
.stat-card .value.negative {{ color: #f85149; }}
.legend-note {{ color: #8b949e; font-size: 11px; margin-top: 8px; }}
</style>
</head>
<body>

<h1>autoresearch-trader — {pair_label} Backtest</h1>
<div class="subtitle">{header_subtitle}</div>

<div id="stats-row" class="stats-row"></div>
<div id="charts"></div>

<script>
const DATA = {payload};

const stats = [
    ['Total Return', ({total_ret:.6f} * 100).toFixed(2) + '%', {str(total_ret >= 0).lower()}],
    ['Max Drawdown', ({max_dd:.6f} * 100).toFixed(2) + '%', false],
    ['Win Rate', ({win_rate:.6f} * 100).toFixed(1) + '%', {str(win_rate >= 0.5).lower()}],
    ['Trades', '{len(trades)}  (TP {n_tp} / SL {n_sl})', true],
];
const statsRow = document.getElementById('stats-row');
stats.forEach(([label, value, pos]) => {{
    const card = document.createElement('div');
    card.className = 'stat-card';
    const cls = (label === 'Trades') ? '' : (pos ? ' positive' : ' negative');
    card.innerHTML = `<div class="label">${{label}}</div><div class="value${{cls}}">${{value}}</div>`;
    statsRow.appendChild(card);
}});

const chartsDiv = document.getElementById('charts');
function addChart(title, cfg) {{
    const container = document.createElement('div');
    container.className = 'chart-container';
    const h2 = document.createElement('h2');
    h2.textContent = title;
    container.appendChild(h2);
    const wrapper = document.createElement('div');
    wrapper.className = 'chart-wrapper';
    const canvas = document.createElement('canvas');
    wrapper.appendChild(canvas);
    container.appendChild(wrapper);
    chartsDiv.appendChild(container);
    return new Chart(canvas.getContext('2d'), cfg);
}}

const step = Math.max(1, Math.floor(DATA.times.length / 80));
const sparseLabels = DATA.times.map((l, i) => (i % step === 0 ? l.slice(5, 16) : ''));

const scatterLong = DATA.long_entries.map(p => ({{ x: p.time.slice(5, 16), y: p.price }}));
const scatterShort = DATA.short_entries.map(p => ({{ x: p.time.slice(5, 16), y: p.price }}));
const scatterTP = DATA.tp_exits.map(p => ({{ x: p.time.slice(5, 16), y: p.price }}));
const scatterSL = DATA.sl_exits.map(p => ({{ x: p.time.slice(5, 16), y: p.price }}));

addChart('{pair_label} — Price + Trade Markers', {{
    type: 'line',
    data: {{
        labels: sparseLabels,
        datasets: [
            {{
                label: 'Close',
                data: DATA.closes,
                borderColor: '#f0f6fc', borderWidth: 1, pointRadius: 0, tension: 0,
            }},
            {{
                label: 'Long entries', type: 'scatter', data: scatterLong,
                backgroundColor: '#3fb950', pointRadius: 4, pointStyle: 'triangle',
            }},
            {{
                label: 'Short entries', type: 'scatter', data: scatterShort,
                backgroundColor: '#f85149', pointRadius: 4, pointStyle: 'triangle', rotation: 180,
            }},
            {{
                label: 'TP exits', type: 'scatter', data: scatterTP,
                backgroundColor: '#58a6ff', pointRadius: 3, pointStyle: 'circle',
            }},
            {{
                label: 'SL exits', type: 'scatter', data: scatterSL,
                backgroundColor: '#db6d28', pointRadius: 3, pointStyle: 'crossRot',
            }},
        ],
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 11 }} }} }},
            tooltip: {{ mode: 'nearest', intersect: false }},
        }},
        scales: {{
            x: {{ ticks: {{ color: '#484f58', maxRotation: 45, font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }},
            y: {{ title: {{ display: true, text: 'USD/JPY', color: '#8b949e' }}, ticks: {{ color: '#484f58' }}, grid: {{ color: '#21262d' }} }},
        }},
    }},
}});

addChart('Equity Curve', {{
    type: 'line',
    data: {{
        labels: sparseLabels,
        datasets: [{{
            label: 'Cumulative Return',
            data: DATA.cum.map(v => (v * 100).toFixed(4)),
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.1)',
            fill: true, borderWidth: 1.5, pointRadius: 0,
        }}],
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
        scales: {{
            x: {{ ticks: {{ color: '#484f58', maxRotation: 45, font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }},
            y: {{ title: {{ display: true, text: 'Return %', color: '#8b949e' }}, ticks: {{ color: '#484f58' }}, grid: {{ color: '#21262d' }} }},
        }},
    }},
}});

const note = document.createElement('div');
note.className = 'legend-note';
note.textContent = 'Green ▲ = long entry · Red ▼ = short entry · Blue ● = TP hit · Orange ✕ = SL hit';
chartsDiv.appendChild(note);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visual forex backtesting tool")
    parser.add_argument("--png", action="store_true", help="Placeholder for future PNG export")
    _ = parser.parse_args()

    data = download_all(force=False)
    _ = ALL_SYMBOLS  # kept for future multi-symbol work
    run_visual_backtest(data)


if __name__ == "__main__":
    main()
