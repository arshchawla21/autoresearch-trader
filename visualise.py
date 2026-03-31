"""
Visualize backtest decisions for a specific stock.

Usage:
    python visualize.py AAPL
    python visualize.py NVDA --all-period    # show full history, not just test
    python visualize.py                      # defaults to SPY

Requires: matplotlib, prepare.py data cached.
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from datetime import date

from prepare import C, load_data, evaluate_sharpe, TRANSACTION_COST_BPS, TARGET_LEVERAGE

# ---------------------------------------------------------------------------
# Strategy (best config from opt.py — no sweep, just the winner)
# ---------------------------------------------------------------------------

def cross_zscore(x):
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (x - mu) / std

def compute_betas(daily_ret, ref_ret, lookback, D, N):
    betas = torch.zeros(D, N)
    for t in range(lookback, D):
        ref_w = ref_ret[t - lookback:t]
        ref_dm = ref_w - ref_w.mean()
        ref_var = (ref_dm ** 2).mean().clamp(min=1e-10)
        for j in range(N):
            asset_w = daily_ret[t - lookback:t, j]
            cov = ((asset_w - asset_w.mean()) * ref_dm).mean()
            betas[t, j] = cov / ref_var
    betas[:lookback] = 1.0
    return betas

def risk_adj_momentum(ret_tensor, horizon, skip, D, N):
    sig = torch.zeros(D, N)
    for t in range(horizon + skip, D):
        window = ret_tensor[t - horizon - skip:t - skip]
        cum_ret = window.sum(dim=0)
        vol = window.std(dim=0).clamp(min=1e-8) * (horizon ** 0.5)
        sig[t] = cum_ret / vol
    return sig

def simple_momentum(log_close, horizon):
    sig = torch.zeros_like(log_close)
    sig[horizon:] = log_close[horizon:] - log_close[:-horizon]
    return sig

def run_strategy(data):
    """
    Run the best strategy config. Returns per-day position weights (D, N_trade).
    This is the winning config from your sweep experiments.
    Adjust these if you find better params.
    """
    ohlcv = data["ohlcv"]
    tradeable_idx = data["tradeable_indices"]
    N_trade = data["num_tradeable"]
    D = ohlcv.shape[0]

    close = ohlcv[:, tradeable_idx, C]
    log_close = torch.log(close.clamp(min=1e-8))
    daily_ret = torch.zeros(D, N_trade)
    daily_ret[1:] = log_close[1:] - log_close[:-1]

    spy_ret = daily_ret[:, 0]
    qqq_ret = daily_ret[:, 1]

    print("Computing betas...")
    betas_spy = compute_betas(daily_ret, spy_ret, 42, D, N_trade)
    betas_qqq = compute_betas(daily_ret, qqq_ret, 42, D, N_trade)

    # 2-factor beta-neutral residuals (best config: SPY + 0.3*QQQ)
    bn = daily_ret - betas_spy * spy_ret.unsqueeze(1) \
                    - 0.3 * betas_qqq * qqq_ret.unsqueeze(1)

    # Signals
    z_rev5 = cross_zscore(simple_momentum(log_close, 5))
    ram_21 = cross_zscore(risk_adj_momentum(bn, 21, 5, D, N_trade))
    ram_63 = cross_zscore(risk_adj_momentum(bn, 63, 5, D, N_trade))

    # Dispersion scaling
    cs_disp = daily_ret.std(dim=1)
    disp_ma42 = torch.zeros(D)
    for t in range(42, D):
        disp_ma42[t] = cs_disp[t - 42:t].mean()
    disp_ma42[:42] = cs_disp[:42].mean()
    dr = (cs_disp / disp_ma42.clamp(min=1e-8)).unsqueeze(1).clamp(0.2, 4.0)

    # Acceleration
    mom_21_raw = torch.zeros(D, N_trade)
    for t in range(21, D):
        mom_21_raw[t] = daily_ret[t-21:t].sum(dim=0)
    sig_accel = torch.zeros(D, N_trade)
    for t in range(31, D):
        sig_accel[t] = mom_21_raw[t] - mom_21_raw[t - 10]
    sig_accel = cross_zscore(sig_accel)

    # Underreaction
    expected_ret_spy = betas_spy * spy_ret.unsqueeze(1)
    underreaction = expected_ret_spy - daily_ret
    sig_ur = torch.zeros(D, N_trade)
    for t in range(5, D):
        sig_ur[t] = underreaction[t-5:t].sum(dim=0)
    sig_ur = cross_zscore(sig_ur)

    # Composite signal
    sig = (-0.50 * z_rev5 + 0.20 * ram_21 + 0.30 * ram_63
           - 0.10 * sig_accel + 0.12 * sig_ur) * dr

    # EMA smooth
    smooth = torch.zeros(D, N_trade)
    smooth[0] = sig[0]
    for t in range(1, D):
        smooth[t] = 0.035 * sig[t] + 0.965 * smooth[t - 1]

    # Normalize to target leverage per day
    abs_sum = smooth.abs().sum(dim=1, keepdim=True).clamp(min=1e-10)
    weights = smooth * (TARGET_LEVERAGE / abs_sum)

    return weights, {
        "raw_signal": sig,
        "z_rev5": z_rev5,
        "ram_21": ram_21,
        "ram_63": ram_63,
        "sig_accel": sig_accel,
        "sig_ur": sig_ur,
        "dispersion_ratio": dr,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stock(data, weights, signals, ticker, show_train=False):
    """Generate a 5-panel visualization for a single stock."""
    tickers = data["tradeable_tickers"]
    if ticker not in tickers:
        print(f"ERROR: '{ticker}' not in tradeable universe.")
        print(f"Available: {', '.join(tickers)}")
        sys.exit(1)

    j = tickers.index(ticker)
    ohlcv = data["ohlcv"]
    tradeable_idx = data["tradeable_indices"]
    fwd_ret = data["forward_returns"]
    test_start = data["test_start_idx"]
    train_end = data["train_end_idx"]
    dates = data["dates"]

    # Slice
    if show_train:
        start = 63  # skip warmup
    else:
        start = test_start
    end = len(dates) - 1

    d = dates[start:end]
    price = ohlcv[start:end, tradeable_idx[j], C].numpy()
    pos = weights[start:end, j].numpy()
    fwd = fwd_ret[start:end, j].numpy()
    raw_sig = signals["raw_signal"][start:end, j].numpy()

    # Per-stock daily P&L (weight * forward return, with transaction costs)
    N_trade = data["num_tradeable"]
    turnover = np.zeros_like(pos)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    tc = TRANSACTION_COST_BPS / 10000 * turnover
    daily_pnl = pos * fwd - tc

    # Passive equal-weight baseline: hold this stock at 1/N weight (fair comparison)
    passive_weight = 1.0 / N_trade
    passive_pnl = passive_weight * fwd

    # Cumulative returns
    cum_strategy = np.cumprod(1 + daily_pnl) - 1
    cum_passive = np.cumprod(1 + passive_pnl) - 1
    buy_hold_ret = price / price[0] - 1

    # Mark test boundary
    if show_train:
        test_date = dates[test_start]
    else:
        test_date = None

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5, 1.2]})

    bg_color = "#fafafa"
    fig.patch.set_facecolor(bg_color)
    for ax in axes:
        ax.set_facecolor(bg_color)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(labelsize=9)
        if test_date and show_train:
            ax.axvline(test_date, color="#e74c3c", linestyle="--", alpha=0.4, linewidth=1)

    # Panel 1: Price with position coloring
    ax = axes[0]
    ax.plot(d, price, color="#2c3e50", linewidth=0.8, alpha=0.5, zorder=1)

    # Color price line by position: green=long, red=short, gray=flat
    points = np.array([mdates.date2num(di) for di in d])
    segments = np.column_stack([points[:-1], price[:-1], points[1:], price[1:]])
    segments = segments.reshape(-1, 2, 2)

    # Normalize position for color intensity
    pos_clip = np.clip(pos[:-1], -0.15, 0.15)
    colors = np.zeros((len(pos_clip), 4))
    for i, p in enumerate(pos_clip):
        if p > 0.001:
            intensity = min(abs(p) / 0.08, 1.0)
            colors[i] = (0.18, 0.8, 0.34, 0.3 + 0.7 * intensity)  # green
        elif p < -0.001:
            intensity = min(abs(p) / 0.08, 1.0)
            colors[i] = (0.91, 0.30, 0.24, 0.3 + 0.7 * intensity)  # red
        else:
            colors[i] = (0.5, 0.5, 0.5, 0.2)

    lc = LineCollection(segments, colors=colors, linewidths=2.5, zorder=2)
    ax.add_collection(lc)
    ax.set_xlim(d[0], d[-1])
    ax.set_ylim(price.min() * 0.97, price.max() * 1.03)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title(f"{ticker} — Backtest Decisions", fontsize=16, fontweight="bold", pad=15)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=(0.18, 0.8, 0.34), linewidth=3, label="Long position"),
        Line2D([0], [0], color=(0.91, 0.30, 0.24), linewidth=3, label="Short position"),
        Line2D([0], [0], color=(0.5, 0.5, 0.5), linewidth=2, alpha=0.4, label="Flat"),
    ]
    if test_date and show_train:
        legend_elements.append(
            Line2D([0], [0], color="#e74c3c", linestyle="--", alpha=0.5, label="Train/Test split"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    # Panel 2: Position weight
    ax = axes[1]
    ax.fill_between(d, 0, pos, where=(pos > 0), color="#2ecc71", alpha=0.5, step="post")
    ax.fill_between(d, 0, pos, where=(pos < 0), color="#e74c3c", alpha=0.5, step="post")
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Position Weight", fontsize=11)
    ax.set_title("Portfolio Weight", fontsize=12, pad=8)

    # Panel 3: Raw signal (pre-EMA, pre-normalization)
    ax = axes[2]
    ax.plot(d, raw_sig, color="#8e44ad", linewidth=0.7, alpha=0.7)
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Signal", fontsize=11)
    ax.set_title("Raw Composite Signal (pre-EMA)", fontsize=12, pad=8)

    # Panel 4: Cumulative P&L — fair comparison at portfolio-weight scale
    ax = axes[3]
    ax.plot(d, cum_strategy * 100, color="#2ecc71", linewidth=1.8,
            label=f"Strategy (active weight)")
    ax.plot(d, cum_passive * 100, color="#f39c12", linewidth=1.2, alpha=0.7,
            linestyle="--", label=f"Passive (1/{N_trade} = {passive_weight:.1%} weight)")
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title(f"Strategy vs Passive Equal-Weight — this stock's portfolio contribution", fontsize=12, pad=8)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    # Secondary axis: 100% buy-and-hold for context
    ax2 = ax.twinx()
    ax2.plot(d, buy_hold_ret * 100, color="#3498db", linewidth=1.0, alpha=0.3)
    ax2.set_ylabel(f"100% Buy & Hold (%)", fontsize=9, color="#3498db", alpha=0.5)
    ax2.tick_params(axis="y", labelcolor="#3498db", labelsize=8)

    # Panel 5: Daily P&L bars
    ax = axes[4]
    colors_bar = ["#2ecc71" if p >= 0 else "#e74c3c" for p in daily_pnl]
    ax.bar(d, daily_pnl * 10000, color=colors_bar, alpha=0.6, width=1.5)
    ax.axhline(0, color="#2c3e50", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Daily P&L (bps)", fontsize=11)
    ax.set_title("Daily P&L Contribution (basis points)", fontsize=12, pad=8)
    ax.set_xlabel("Date", fontsize=11)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Stats annotation
    test_mask = np.array([i >= (test_start - start) for i in range(len(d))])
    if test_mask.any():
        test_pnl = daily_pnl[test_mask]
        test_passive = passive_pnl[test_mask]
        test_sr = (test_pnl.mean() / test_pnl.std() * np.sqrt(252)) if test_pnl.std() > 1e-10 else 0
        test_cum = np.prod(1 + test_pnl) - 1
        passive_cum = np.prod(1 + test_passive) - 1
        bh_cum = price[test_mask][-1] / price[test_mask][0] - 1
        avg_pos = np.mean(np.abs(pos[test_mask]))
        avg_turn = np.mean(turnover[test_mask])

        stats_text = (
            f"Test period — {ticker}:\n"
            f"  Strategy contribution: {test_cum*100:+.2f}%\n"
            f"  Passive (1/{N_trade}):       {passive_cum*100:+.2f}%\n"
            f"  Alpha vs passive:     {(test_cum - passive_cum)*100:+.2f}%\n"
            f"  100% buy & hold:      {bh_cum*100:+.2f}%\n"
            f"  ─────────────────────────\n"
            f"  Sharpe (this stock):  {test_sr:.2f}\n"
            f"  Avg |weight|:         {avg_pos:.4f} ({avg_pos*100:.1f}%)\n"
            f"  Avg turnover:         {avg_turn:.4f}\n"
            f"  Days long / short:    {np.sum(pos[test_mask] > 0.001)} / {np.sum(pos[test_mask] < -0.001)}"
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
# Signal decomposition (bonus: show what drives the position)
# ---------------------------------------------------------------------------

def plot_signal_decomposition(data, signals, ticker, show_train=False):
    """Show how individual signals contribute to the composite for a stock."""
    tickers = data["tradeable_tickers"]
    j = tickers.index(ticker)
    test_start = data["test_start_idx"]
    dates = data["dates"]

    start = 63 if show_train else test_start
    end = len(dates) - 1
    d = dates[start:end]

    components = {
        "5d Reversal (w=-0.50)": -0.50 * signals["z_rev5"][start:end, j].numpy(),
        "21d RA-Mom (w=0.20)": 0.20 * signals["ram_21"][start:end, j].numpy(),
        "63d RA-Mom (w=0.30)": 0.30 * signals["ram_63"][start:end, j].numpy(),
        "Acceleration (w=-0.10)": -0.10 * signals["sig_accel"][start:end, j].numpy(),
        "Underreaction (w=0.12)": 0.12 * signals["sig_ur"][start:end, j].numpy(),
    }
    dr = signals["dispersion_ratio"][start:end, 0].numpy()

    fig, axes = plt.subplots(len(components) + 1, 1, figsize=(18, 14), sharex=True)
    fig.patch.set_facecolor("#fafafa")

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

    for ax, (name, vals), color in zip(axes[:-1], components.items(), colors):
        ax.set_facecolor("#fafafa")
        ax.fill_between(d, 0, vals, where=(vals > 0), color=color, alpha=0.4, step="post")
        ax.fill_between(d, 0, vals, where=(vals < 0), color=color, alpha=0.2, step="post")
        ax.plot(d, vals, color=color, linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="#2c3e50", linewidth=0.3, alpha=0.3)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.1)
        ax.tick_params(labelsize=8)

    # Dispersion ratio
    ax = axes[-1]
    ax.set_facecolor("#fafafa")
    ax.plot(d, dr, color="#2c3e50", linewidth=0.8)
    ax.axhline(1, color="#e74c3c", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Disp. Ratio", fontsize=9)
    ax.set_xlabel("Date", fontsize=11)
    ax.grid(True, alpha=0.1)
    ax.tick_params(labelsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle(f"{ticker} — Signal Decomposition", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    outfile = f"signals_{ticker}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    print(f"Saved to {outfile}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize backtest decisions for a stock")
    parser.add_argument("ticker", nargs="?", default="SPY",
                        help="Ticker to visualize (default: SPY)")
    parser.add_argument("--all-period", action="store_true",
                        help="Show full history including training period")
    parser.add_argument("--decompose", action="store_true",
                        help="Also plot signal decomposition")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    print("Loading data...")
    data = load_data(device="cpu")

    print("Running strategy...")
    weights, signals = run_strategy(data)

    print(f"Plotting {ticker}...")
    plot_stock(data, weights, signals, ticker, show_train=args.all_period)

    if args.decompose:
        plot_signal_decomposition(data, signals, ticker, show_train=args.all_period)