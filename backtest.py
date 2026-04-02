#!/usr/bin/env python3
"""
backtest.py — Visual Backtesting Tool
=======================================
Runs the same backtest as prepare.py but generates an interactive HTML chart
showing price action + trades for a chosen symbol.

Usage:
    uv run backtest.py                  # default: AAPL
    uv run backtest.py TSLA             # visualize TSLA
    uv run backtest.py NVDA --png       # also save a static PNG
    uv run backtest.py AAPL MSFT NVDA   # multi-symbol dashboard
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse prepare.py's data loading and universe definitions
from prepare import (
    ALL_SYMBOLS,
    HISTORY_DAYS,
    TRADEABLE,
    download_all,
    _split_by_trading_days,
    _get_close_matrix,
)


def run_visual_backtest(
    data: dict[str, pd.DataFrame],
    focus_symbols: list[str],
) -> None:
    """
    Run backtest and generate an interactive HTML chart per symbol
    showing: price, long entries (green), short entries (red), weight heatmap.
    """
    try:
        import train
    except ImportError:
        print("[✗] Cannot import train.py")
        sys.exit(1)

    trading_days = _split_by_trading_days(data)
    if len(trading_days) < HISTORY_DAYS + 1:
        print(f"[✗] Only {len(trading_days)} trading days, need >{HISTORY_DAYS}")
        sys.exit(1)

    eval_start_day = trading_days[HISTORY_DAYS]
    close_matrix = _get_close_matrix(data, TRADEABLE)

    # tz-aware comparison (same fix as prepare.py)
    eval_ts = pd.Timestamp(eval_start_day)
    if close_matrix.index.tz is not None:
        eval_ts = eval_ts.tz_localize(close_matrix.index.tz)
    eval_mask = close_matrix.index >= eval_ts
    eval_indices = np.where(eval_mask)[0]

    if len(eval_indices) < 2:
        print("[✗] Not enough eval candles.")
        sys.exit(1)

    # Collect per-candle data
    records: list[dict] = []
    n_candles = len(eval_indices)
    print(f"Running {n_candles} eval steps...")

    for step, idx in enumerate(eval_indices[:-1]):
        current_time = close_matrix.index[idx]
        prices_so_far: dict[str, pd.DataFrame] = {}
        for sym in ALL_SYMBOLS:
            if sym in data:
                mask = data[sym].index <= current_time
                prices_so_far[sym] = data[sym].loc[mask].copy()

        try:
            weights = train.trade(prices_so_far, idx, TRADEABLE)
        except Exception as e:
            weights = [0.0] * len(TRADEABLE)

        weights = np.array(weights, dtype=float)
        if len(weights) != len(TRADEABLE):
            weights = np.zeros(len(TRADEABLE))
        total_lev = np.sum(np.abs(weights))
        if total_lev > 1.0 + 1e-9:
            weights = weights / total_lev

        # Next candle returns
        next_idx = idx + 1
        current_close = close_matrix.iloc[idx].values
        next_close = close_matrix.iloc[next_idx].values
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_ret = np.where(current_close > 0, (next_close - current_close) / current_close, 0.0)
        pct_ret = np.nan_to_num(pct_ret, nan=0.0)

        port_ret = float(np.dot(weights, pct_ret))

        rec = {
            "time": str(current_time),
            "portfolio_return": port_ret,
        }
        for i, sym in enumerate(TRADEABLE):
            rec[f"close_{sym}"] = float(current_close[i])
            rec[f"weight_{sym}"] = float(weights[i])
            rec[f"ret_{sym}"] = float(pct_ret[i])
        records.append(rec)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1}/{n_candles-1}")

    print(f"Done. Generating charts for: {', '.join(focus_symbols)}")

    # Build HTML
    html = _build_html(records, focus_symbols, TRADEABLE)
    out_path = Path("backtest_visual.html")
    out_path.write_text(html)
    print(f"\n[✓] Saved to {out_path.resolve()}")
    webbrowser.open(str(out_path.resolve()))


def _build_html(records: list[dict], focus_symbols: list[str], all_symbols: list[str]) -> str:
    """Build a self-contained interactive HTML dashboard."""

    # Prepare JSON data for each focus symbol
    import json

    charts_data = {}
    for sym in focus_symbols:
        times = []
        closes = []
        weights = []
        cum_pnl = []
        running = 1.0
        for r in records:
            times.append(r["time"])
            closes.append(r.get(f"close_{sym}", 0))
            weights.append(r.get(f"weight_{sym}", 0))
            running *= (1 + r["portfolio_return"])
            cum_pnl.append(running - 1)

        charts_data[sym] = {
            "times": times,
            "closes": closes,
            "weights": weights,
            "cum_pnl": cum_pnl,
        }

    # Portfolio-level cumulative return
    port_cum = []
    running = 1.0
    for r in records:
        running *= (1 + r["portfolio_return"])
        port_cum.append(running - 1)

    portfolio_data = {
        "times": [r["time"] for r in records],
        "cum_pnl": port_cum,
    }

    data_json = json.dumps({"charts": charts_data, "portfolio": portfolio_data})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Visual — {', '.join(focus_symbols)}</title>
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
.chart-wrapper {{ position: relative; height: 280px; }}
.stats-row {{
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;
}}
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

<h1>autoresearch-trader — Backtest Visual</h1>
<div class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Eval window: {records[0]['time'][:10]} → {records[-1]['time'][:10]} · {len(records)} candles</div>

<div id="stats-row" class="stats-row"></div>
<div id="charts"></div>

<script>
const DATA = {data_json};
const FOCUS = {json.dumps(focus_symbols)};

// Stats
const portPnl = DATA.portfolio.cum_pnl;
const totalRet = portPnl[portPnl.length - 1];
let maxDD = 0, peak = 0;
for (let i = 0; i < portPnl.length; i++) {{
    const v = 1 + portPnl[i];
    if (v > peak) peak = v;
    const dd = (v - peak) / peak;
    if (dd < maxDD) maxDD = dd;
}}

const statsRow = document.getElementById('stats-row');
const stats = [
    ['Total Return', (totalRet * 100).toFixed(2) + '%', totalRet >= 0],
    ['Max Drawdown', (maxDD * 100).toFixed(2) + '%', false],
    ['Eval Candles', portPnl.length.toString(), true],
    ['Symbols Shown', FOCUS.join(', '), true],
];
stats.forEach(([label, value, pos]) => {{
    const card = document.createElement('div');
    card.className = 'stat-card';
    const cls = label === 'Eval Candles' || label === 'Symbols Shown' ? '' :
                (pos ? ' positive' : ' negative');
    card.innerHTML = `<div class="label">${{label}}</div><div class="value${{cls}}">${{value}}</div>`;
    statsRow.appendChild(card);
}});

// Portfolio cumulative chart
const chartsDiv = document.getElementById('charts');

function addChart(title, labels, datasets, yLabel) {{
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

    // Downsample labels for readability
    const step = Math.max(1, Math.floor(labels.length / 60));
    const sparseLabels = labels.map((l, i) => i % step === 0 ? l.slice(5, 16) : '');

    new Chart(canvas.getContext('2d'), {{
        type: 'line',
        data: {{ labels: sparseLabels, datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 11 }} }} }},
                tooltip: {{
                    backgroundColor: '#1c2128',
                    titleColor: '#f0f6fc',
                    bodyColor: '#c9d1d9',
                    borderColor: '#30363d',
                    borderWidth: 1,
                }},
            }},
            scales: {{
                x: {{
                    ticks: {{ color: '#484f58', maxRotation: 45, font: {{ size: 9 }} }},
                    grid: {{ color: '#21262d' }},
                }},
                y: {{
                    title: {{ display: true, text: yLabel, color: '#8b949e' }},
                    ticks: {{ color: '#484f58' }},
                    grid: {{ color: '#21262d' }},
                }},
            }},
        }},
    }});
}}

// Portfolio P&L chart
addChart(
    'Portfolio Cumulative P&L',
    DATA.portfolio.times,
    [{{
        label: 'Cumulative Return',
        data: DATA.portfolio.cum_pnl.map(v => (v * 100).toFixed(4)),
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        fill: true, borderWidth: 1.5, pointRadius: 0,
    }}],
    'Return %'
);

// Per-symbol charts
FOCUS.forEach(sym => {{
    const d = DATA.charts[sym];
    if (!d) return;

    // Price chart with weight-colored background segments
    const bgColors = d.weights.map(w =>
        w > 0.01 ? 'rgba(63,185,80,0.15)' :
        w < -0.01 ? 'rgba(248,81,73,0.15)' :
        'transparent'
    );

    // Weight trace
    addChart(
        sym + ' — Price + Position Weight',
        d.times,
        [
            {{
                label: sym + ' Close',
                data: d.closes,
                borderColor: '#f0f6fc',
                borderWidth: 1.5, pointRadius: 0,
                yAxisID: 'y',
            }},
            {{
                label: 'Weight',
                data: d.weights.map(w => (w * 100).toFixed(2)),
                borderColor: w => {{
                    // This won't work as a function in border, but we handle via segment
                    return '#58a6ff';
                }},
                backgroundColor: 'rgba(88,166,255,0.1)',
                fill: true, borderWidth: 1, pointRadius: 0,
                yAxisID: 'y1',
                segment: {{
                    borderColor: ctx => {{
                        const v = ctx.p1.parsed.y;
                        return v > 0 ? '#3fb950' : v < 0 ? '#f85149' : '#484f58';
                    }},
                    backgroundColor: ctx => {{
                        const v = ctx.p1.parsed.y;
                        return v > 0 ? 'rgba(63,185,80,0.1)' : v < 0 ? 'rgba(248,81,73,0.1)' : 'transparent';
                    }},
                }},
            }},
        ],
        'Price ($)'
    );

    // Patch the last chart to have dual y-axis
    const lastCanvas = chartsDiv.querySelectorAll('canvas');
    const lc = lastCanvas[lastCanvas.length - 1];
    const chartInstance = Chart.getChart(lc);
    if (chartInstance) {{
        chartInstance.options.scales.y1 = {{
            position: 'right',
            title: {{ display: true, text: 'Weight %', color: '#8b949e' }},
            ticks: {{ color: '#484f58' }},
            grid: {{ drawOnChartArea: false }},
        }};
        chartInstance.update();
    }}
}});

// Add legend note
const note = document.createElement('div');
note.className = 'legend-note';
note.textContent = 'Green fill = long position · Red fill = short position · White line = price';
chartsDiv.appendChild(note);
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Visual backtesting tool")
    parser.add_argument("symbols", nargs="*", default=["AAPL"],
                        help="Symbols to visualize (e.g. AAPL TSLA NVDA)")
    parser.add_argument("--png", action="store_true", help="Placeholder for future PNG export")
    args = parser.parse_args()

    # Validate symbols
    focus = []
    for s in args.symbols:
        s = s.upper()
        if s in TRADEABLE:
            focus.append(s)
        else:
            print(f"[!] {s} not in tradeable universe, skipping.")
    if not focus:
        print("[✗] No valid symbols. Choose from:", ", ".join(TRADEABLE))
        sys.exit(1)

    data = download_all(force=False)
    run_visual_backtest(data, focus)


if __name__ == "__main__":
    main()