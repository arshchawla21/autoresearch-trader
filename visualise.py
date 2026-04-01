"""
Visualize 15-minute intraday day-trading backtest results for a strategy.

Usage:
    python visualise.py                     # backtest out-of-sample period, show summary
    python visualise.py AAPL                # per-stock breakdown for AAPL
    python visualise.py --all-stocks        # show per-stock heatmap

Requires: matplotlib, prepare.py data cached.
"""

import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from prepare import (
    H, L, C, load_data, run_backtest, _find_date_idx
)

# ---------------------------------------------------------------------------
# Per-stock trade analysis
# ---------------------------------------------------------------------------

def plot_stock_trades(data, result, ticker):
    """
    Plot detailed trade-level analysis for a single stock.

    4-panel chart:
    1. Price with trade entry/exit markers
    2. Interval P&L from this stock
    3. Cumulative return from this stock
    4. Trade outcome distribution
    """
    tickers = data["tradeable_tickers"]
    if ticker not in tickers:
        print(f"ERROR: '{ticker}' not in tradeable universe.")
        print(f"Available: {', '.join(tickers)}")
        sys.exit(1)

    j = tickers.index(ticker)
    tradeable_idx = data["tradeable_indices"]
    test_dates = result["test_dates"]
    per_interval = result["per_interval_trades"]

    # Get price data for test period
    test_start_idx = next(i for i, d in enumerate(data["dates"]) if d == test_dates[0])
    test_end_idx = test_start_idx + len(test_dates)

    ohlcv = data["ohlcv"][test_start_idx:test_end_idx, tradeable_idx[j]].numpy()
    prices_close = ohlcv[:, C]
    prices_high = ohlcv[:, H]
    prices_low = ohlcv[:, L]

    # Extract trades for this ticker
    stock_interval_ret = np.zeros(len(test_dates))
    entry_bars, entry_prices, entry_dirs = [], [], []
    exit_bars, exit_prices = [], []
    stop_hits, limit_hits, close_exits = 0, 0, 0
    trade_returns = []

    for bar_i, trades in enumerate(per_interval):
        for t in trades:
            if t["ticker"] != ticker:
                continue
            stock_interval_ret[bar_i] += t["weight"] * t["return"]
            entry_bars.append(bar_i)
            entry_prices.append(t["entry_price"])
            entry_dirs.append(t["direction"])
            exit_bars.append(bar_i)
            exit_prices.append(t["exit_price"])
            trade_returns.append(t["return"])

            if t["stop_hit"]:
                stop_hits += 1
            elif t["limit_hit"]:
                limit_hits += 1
            elif t["held_to_close"]:
                close_exits += 1

    n_trades = len(trade_returns)
    if n_trades == 0:
        print(f"No trades found for {ticker} in this period.")
        return

    trade_returns = np.array(trade_returns)
    cum_ret = np.cumprod(1 + stock_interval_ret) - 1

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=False,
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.2]})

    bg_color = "#fafafa"
    fig.patch.set_facecolor(bg_color)
    for ax in axes[:3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(labelsize=9)

    d = test_dates

    # Panel 1: Price with trade markers
    ax = axes[0]
    ax.plot(d, prices_close, color="#2c3e50", linewidth=0.8, alpha=0.6, label="Close")

    # Shade high-low range
    ax.fill_between(d, prices_low, prices_high, color="#bdc3c7", alpha=0.2, label="High-Low range")

    # Entry/exit markers
    for i in range(len(entry_bars)):
        bar_i = entry_bars[i]
        is_long = entry_dirs[i] == "long"
        color = "#2ecc71" if is_long else "#e74c3c"
        marker_entry = "^" if is_long else "v"

        ax.scatter(d[bar_i], entry_prices[i], color=color, marker=marker_entry,
                   s=30, zorder=5, alpha=0.7)
        ax.scatter(d[bar_i], exit_prices[i], color=color, marker="x",
                   s=25, zorder=5, alpha=0.5)

    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title(f"{ticker} — Intraday Day Trading Backtest", fontsize=16, fontweight="bold", pad=15)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2c3e50", linewidth=1, label="Close"),
        Line2D([0], [0], marker="^", color="#2ecc71", linestyle="None", markersize=8, label="Long entry"),
        Line2D([0], [0], marker="v", color="#e74c3c", linestyle="None", markersize=8, label="Short entry"),
        Line2D([0], [0], marker="x", color="#666", linestyle="None", markersize=8, label="Exit"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    # Panel 2: Interval P&L
    ax = axes[1]
    colors_bar = ["#2ecc71" if p >= 0 else "#e74c3c" for p in stock_interval_ret]
    ax.bar(d, stock_interval_ret * 10000, color=colors_bar, alpha=0.6, width=0.01) # thin width for 15m
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Interval P&L (bps)", fontsize=11)
    ax.set_title(f"{ticker} — 15m P&L Contribution", fontsize=12, pad=8)
    ax.set_facecolor(bg_color)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Panel 3: Cumulative return
    ax = axes[2]
    ax.plot(d, cum_ret * 100, color="#2ecc71", linewidth=1.8, label=f"{ticker} strategy")
    buy_hold = prices_close / prices_close[0] - 1
    ax.plot(d, buy_hold * 100, color="#3498db", linewidth=1.2, alpha=0.7, label="Buy & hold")
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title("Strategy vs Buy & Hold", fontsize=12, pad=8)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_facecolor(bg_color)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Panel 4: Trade outcome distribution
    ax = axes[3]
    ax.set_facecolor(bg_color)

    # Left: histogram of trade returns
    ax.hist(trade_returns * 100, bins=30, color="#3498db", alpha=0.6, edgecolor="#2c3e50", linewidth=0.5)
    ax.axvline(0, color="#2c3e50", linewidth=1, alpha=0.5)
    ax.axvline(trade_returns.mean() * 100, color="#e74c3c", linewidth=1.5, linestyle="--",
               label=f"Mean: {trade_returns.mean()*100:.2f}%")
    ax.set_xlabel("Trade Return (%)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Trade Return Distribution", fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Date formatting for time-series panels (Intraday style)
    for ax_ts in axes[:3]:
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Stats annotation
    win_rate = np.mean(trade_returns > 0) * 100
    avg_win = trade_returns[trade_returns > 0].mean() * 100 if (trade_returns > 0).any() else 0
    avg_loss = trade_returns[trade_returns <= 0].mean() * 100 if (trade_returns <= 0).any() else 0
    total_ret = cum_ret[-1] * 100 if len(cum_ret) > 0 else 0
    sr = stock_interval_ret
    # 6552 periods = 252 days * 26 (15m periods per day)
    sharpe = float(np.sqrt(6552) * sr.mean() / sr.std()) if sr.std() > 1e-10 else 0

    stats_text = (
        f"  {ticker} Trade Stats\n"
        f"  ────────────────────────\n"
        f"  Total trades:    {n_trades}\n"
        f"  Win rate:        {win_rate:.1f}%\n"
        f"  Avg winner:      {avg_win:+.2f}%\n"
        f"  Avg loser:       {avg_loss:+.2f}%\n"
        f"  ────────────────────────\n"
        f"  Stop-loss hits:  {stop_hits}\n"
        f"  Take-profit:     {limit_hits}\n"
        f"  Held to close:   {close_exits}\n"
        f"  ────────────────────────\n"
        f"  Total return:    {total_ret:+.2f}%\n"
        f"  Sharpe (ann.):   {sharpe:.2f}\n"
        f"  Buy&hold:        {(buy_hold[-1])*100:+.2f}%"
    )
    fig.text(0.98, 0.98, stats_text, transform=fig.transFigure,
             fontsize=9, verticalalignment="top", horizontalalignment="right",
             fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5",
             facecolor="white", edgecolor="#cccccc", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    outfile = f"backtest_{ticker}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
    print(f"Saved to {outfile}")
    plt.show()


# ---------------------------------------------------------------------------
# Portfolio-level overview
# ---------------------------------------------------------------------------

def plot_portfolio_overview(data, result):
    """
    Plot portfolio-level backtest overview.

    3-panel chart:
    1. Equity curve with drawdown shading
    2. Per-stock cumulative P&L heatmap
    3. Interval trade count and portfolio utilization
    """
    test_dates = result["test_dates"]
    interval_returns = result["interval_returns"]
    per_interval = result["per_interval_trades"]
    tickers = data["tradeable_tickers"]

    d = test_dates
    cum = np.cumprod(1 + interval_returns)
    peak = np.maximum.accumulate(cum)
    drawdown = cum / peak - 1

    # Per-stock interval returns
    n_tickers = len(tickers)
    stock_rets = np.zeros((len(d), n_tickers))
    interval_trade_count = np.zeros(len(d))
    interval_utilization = np.zeros(len(d))

    for bar_i, trades in enumerate(per_interval):
        interval_trade_count[bar_i] = len(trades)
        for t in trades:
            j = tickers.index(t["ticker"]) if t["ticker"] in tickers else -1
            if j >= 0:
                stock_rets[bar_i, j] += t["weight"] * t["return"]
                interval_utilization[bar_i] += t["weight"]

    stock_cum = np.cumsum(stock_rets, axis=0)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(18, 16),
                              gridspec_kw={"height_ratios": [2, 2, 1]})

    bg_color = "#fafafa"
    fig.patch.set_facecolor(bg_color)

    # Panel 1: Equity curve + drawdown
    ax = axes[0]
    ax.set_facecolor(bg_color)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    ax.plot(d, (cum - 1) * 100, color="#2ecc71", linewidth=2, label="Portfolio")
    ax.fill_between(d, drawdown * 100, 0, color="#e74c3c", alpha=0.15, label="Drawdown")
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Return / Drawdown (%)", fontsize=11)
    ax.set_title("Portfolio Equity Curve (Out-of-Sample)", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    # Panel 2: Per-stock cumulative P&L
    ax = axes[1]
    ax.set_facecolor(bg_color)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Only show stocks that were actually traded
    traded_mask = np.abs(stock_cum[-1]) > 1e-6
    colors_cycle = plt.cm.tab20(np.linspace(0, 1, n_tickers))

    for j in range(n_tickers):
        if not traded_mask[j]:
            continue
        ax.plot(d, stock_cum[:, j] * 100, linewidth=1.2, alpha=0.7,
                color=colors_cycle[j], label=tickers[j])

    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Cumulative P&L (%)", fontsize=11)
    ax.set_title("Per-Stock Cumulative P&L", fontsize=12, pad=8)
    n_traded = traded_mask.sum()
    if n_traded <= 15:
        ax.legend(fontsize=7, loc="upper left", ncol=min(4, n_traded), framealpha=0.9)
    else:
        ax.legend(fontsize=6, loc="upper left", ncol=5, framealpha=0.9)

    # Panel 3: Interval trade count + utilization
    ax = axes[2]
    ax.set_facecolor(bg_color)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    ax.bar(d, interval_trade_count, color="#3498db", alpha=0.5, width=0.01, label="# Trades")
    ax2 = ax.twinx()
    ax2.plot(d, interval_utilization * 100, color="#e67e22", linewidth=1, alpha=0.7, label="Utilization %")
    ax2.set_ylabel("Portfolio Utilization (%)", fontsize=10, color="#e67e22")

    ax.set_ylabel("# Trades per 15m", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Activity Profile", fontsize=12, pad=8)
    ax.legend(fontsize=9, loc="upper left")
    ax2.legend(fontsize=9, loc="upper right")

    # Date formatting (Intraday style)
    for ax_i in axes:
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax_i.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Stats
    metrics = result["metrics"]
    total_trades = sum(len(t) for t in per_interval)
    active_intervals = sum(1 for t in per_interval if len(t) > 0)

    stats_text = (
        f"  Portfolio Summary\n"
        f"  ────────────────────────\n"
        f"  Sharpe (ann.):   {metrics['sharpe_ratio']:.4f}\n"
        f"  Total return:    {metrics['total_return']*100:+.2f}%\n"
        f"  Max drawdown:    {metrics['max_drawdown']*100:.2f}%\n"
        f"  Win rate:        {metrics['win_rate']*100:.1f}%\n"
        f"  ────────────────────────\n"
        f"  Total trades:    {total_trades}\n"
        f"  Active 15m bars: {active_intervals}/{len(d)}\n"
        f"  Avg trades/bar:  {metrics['avg_daily_trades']:.2f}\n"  # Variable remains daily_trades for backward compat
        f"  Stocks traded:   {int(n_traded)}/{n_tickers}\n"
        f"  Build time:      {result['build_time']:.1f}s"
    )
    fig.text(0.98, 0.98, stats_text, transform=fig.transFigure,
             fontsize=9, verticalalignment="top", horizontalalignment="right",
             fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5",
             facecolor="white", edgecolor="#cccccc", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    outfile = "backtest_portfolio.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
    print(f"Saved to {outfile}")
    plt.show()


# ---------------------------------------------------------------------------
# Per-stock heatmap
# ---------------------------------------------------------------------------

def plot_stock_heatmap(data, result):
    """Show a heatmap of per-stock total returns and trade counts."""
    tickers = data["tradeable_tickers"]
    per_interval = result["per_interval_trades"]

    n_tickers = len(tickers)
    stock_total_ret = np.zeros(n_tickers)
    stock_trade_count = np.zeros(n_tickers, dtype=int)
    stock_win_count = np.zeros(n_tickers, dtype=int)

    for trades in per_interval:
        for t in trades:
            j = tickers.index(t["ticker"]) if t["ticker"] in tickers else -1
            if j >= 0:
                stock_total_ret[j] += t["weight"] * t["return"]
                stock_trade_count[j] += 1
                if t["return"] > 0:
                    stock_win_count[j] += 1

    # Sort by total return
    order = np.argsort(stock_total_ret)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    bg_color = "#fafafa"
    fig.patch.set_facecolor(bg_color)

    # Bar chart: total return
    ax = axes[0]
    ax.set_facecolor(bg_color)
    colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in stock_total_ret[order]]
    ax.barh([tickers[i] for i in order], stock_total_ret[order] * 100, color=colors, alpha=0.7)
    ax.axvline(0, color="#2c3e50", linewidth=0.5)
    ax.set_xlabel("Total Return Contribution (%)")
    ax.set_title("Per-Stock Returns", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="x")

    # Bar chart: trade count
    ax = axes[1]
    ax.set_facecolor(bg_color)
    ax.barh([tickers[i] for i in order], stock_trade_count[order], color="#3498db", alpha=0.7)
    ax.set_xlabel("# Trades")
    ax.set_title("Trade Count", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="x")

    # Bar chart: win rate
    ax = axes[2]
    ax.set_facecolor(bg_color)
    win_rates = np.where(stock_trade_count > 0,
                          stock_win_count / stock_trade_count * 100, 0)
    colors_wr = ["#2ecc71" if w >= 50 else "#e74c3c" for w in win_rates[order]]
    ax.barh([tickers[i] for i in order], win_rates[order], color=colors_wr, alpha=0.7)
    ax.axvline(50, color="#2c3e50", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("Win Rate", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.15, axis="x")

    fig.suptitle("Per-Stock Breakdown", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    outfile = "backtest_stocks.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
    print(f"Saved to {outfile}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize day-trading backtest results")
    parser.add_argument("ticker", nargs="?", default=None,
                        help="Ticker to visualize (omit for portfolio overview)")
    parser.add_argument("--all-stocks", action="store_true",
                        help="Show per-stock heatmap")
    args = parser.parse_args()

    from opt import build_strategy, generate_orders

    print("Loading data...")
    data = load_data(device="cpu")
    dates = data["dates"]
    n_periods = len(dates)

    if n_periods < 100:
        print("ERROR: Not enough data points to evaluate.")
        sys.exit(1)

    # Split: Find the index roughly 30 days from the start date (matches prepare.py)
    start_date = dates[0]
    split_date = start_date + datetime.timedelta(days=30)
    split_idx = _find_date_idx(dates, split_date)

    if split_idx <= 0 or split_idx >= n_periods:
        split_idx = n_periods // 2

    train_end = split_idx
    test_start = split_idx
    test_end = n_periods

    print("Running backtest (evaluating out-of-sample window)...")
    result = run_backtest(
        build_strategy, generate_orders, data,
        train_end_idx=train_end,
        test_start_idx=test_start,
        test_end_idx=test_end,
        device="cpu",
        verbose=True,
    )

    metrics = result["metrics"]
    print(f"\nResults: Sharpe={metrics['sharpe_ratio']:.4f} | "
          f"Return={metrics['total_return']*100:.2f}% | "
          f"MaxDD={metrics['max_drawdown']*100:.2f}% | "
          f"WinRate={metrics['win_rate']*100:.1f}%")

    if args.all_stocks:
        plot_stock_heatmap(data, result)
    elif args.ticker:
        ticker = args.ticker.upper()
        print(f"\nPlotting {ticker}...")
        plot_stock_trades(data, result, ticker)
    else:
        plot_portfolio_overview(data, result)