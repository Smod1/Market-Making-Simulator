"""
Plotting module — publication-quality charts for the simulation results.

All functions return matplotlib Figure objects so they can be saved or
displayed independently.
"""

from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from analytics.metrics import StrategyMetrics

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE = {
    "Strategy A – Fixed Spread":        "#E63946",  # red
    "Strategy B – Inventory-Aware":     "#457B9D",  # blue
    "Strategy C – Avellaneda-Stoikov":  "#2A9D8F",  # teal
    "mid":  "#F4A261",
    "grid": "#2A2A2A",
    "bg":   "#0D1117",
    "text": "#E6EDF3",
}

ALPHA_FILL  = 0.15
LINE_WIDTH  = 1.6

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#21262D",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "legend.framealpha": 0.3,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
})


def _colour(name: str) -> str:
    return PALETTE.get(name, "#AAAAAA")


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_pnl(metrics_list: List[StrategyMetrics]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Mark-to-Market PnL over Simulation", fontsize=14, pad=12)
    ax.set_xlabel("Tick")
    ax.set_ylabel("PnL (price units)")

    for m in metrics_list:
        colour = _colour(m.name)
        y = np.array(m.pnl_history)
        x = np.arange(len(y))
        ax.plot(x, y, lw=LINE_WIDTH, color=colour, label=m.name)
        ax.fill_between(x, 0, y, alpha=ALPHA_FILL, color=colour)

    ax.axhline(0, color="#666", lw=0.8, ls="--")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_inventory(metrics_list: List[StrategyMetrics]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title("Inventory over Simulation", fontsize=14, pad=12)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Inventory (units)")

    for m in metrics_list:
        colour = _colour(m.name)
        y = np.array(m.inventory_history)
        ax.plot(np.arange(len(y)), y, lw=LINE_WIDTH, color=colour,
                label=m.name, alpha=0.85)

    ax.axhline(0, color="#666", lw=0.8, ls="--")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_drawdown(metrics_list: List[StrategyMetrics]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title("Drawdown Profile", fontsize=14, pad=12)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Drawdown (price units)")

    for m in metrics_list:
        pnl = np.array(m.pnl_history)
        running_max = np.maximum.accumulate(pnl)
        dd = pnl - running_max
        colour = _colour(m.name)
        ax.plot(np.arange(len(dd)), dd, lw=LINE_WIDTH, color=colour, label=m.name)
        ax.fill_between(np.arange(len(dd)), dd, 0, alpha=ALPHA_FILL, color=colour)

    ax.axhline(0, color="#666", lw=0.8, ls="--")
    ax.legend(loc="lower left")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_summary_bar(metrics_list: List[StrategyMetrics]) -> plt.Figure:
    """Bar chart comparing key metrics across strategies."""
    labels   = [m.name.split("–")[1].strip() for m in metrics_list]
    colours  = [_colour(m.name) for m in metrics_list]

    stats = {
        "Total PnL":       [m.total_pnl          for m in metrics_list],
        "Sharpe Ratio":    [m.sharpe              for m in metrics_list],
        "Max Drawdown":    [m.max_drawdown        for m in metrics_list],
        "Max |Inventory|": [-m.inv_max_abs        for m in metrics_list],  # negate for visual
    }
    display_titles = {
        "Total PnL":       "Total PnL",
        "Sharpe Ratio":    "Sharpe Ratio",
        "Max Drawdown":    "Max Drawdown (negative = bad)",
        "Max |Inventory|": "Max |Inventory| (negative = more risk)",
    }

    n = len(stats)
    fig, axes = plt.subplots(1, n, figsize=(16, 5))
    fig.suptitle("Strategy Comparison — Key Metrics", fontsize=15, y=1.02)

    for ax, (key, vals) in zip(axes, stats.items()):
        bars = ax.bar(labels, vals, color=colours, width=0.5, edgecolor="#333", linewidth=0.8)
        ax.set_title(display_titles[key], fontsize=10)
        ax.axhline(0, color="#555", lw=0.8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8, color=PALETTE["text"])

    fig.tight_layout()
    return fig


def plot_dashboard(metrics_list: List[StrategyMetrics],
                   mid_history: List[float]) -> plt.Figure:
    """
    Full 4-panel dashboard:
      [0] Mid-price path
      [1] PnL curves
      [2] Inventory
      [3] Drawdown
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

    # Panel 0: Mid price
    ax0 = fig.add_subplot(gs[0, 0])
    mid = np.array(mid_history)
    ax0.plot(mid, lw=1.2, color=PALETTE["mid"], alpha=0.9)
    ax0.set_title("Simulated Mid-Price (GBM)", fontsize=12)
    ax0.set_xlabel("Tick"); ax0.set_ylabel("Price")
    ax0.grid(True)

    # Panel 1: PnL
    ax1 = fig.add_subplot(gs[0, 1])
    for m in metrics_list:
        c = _colour(m.name)
        y = np.array(m.pnl_history)
        ax1.plot(y, lw=LINE_WIDTH, color=c, label=m.name.split("–")[0].strip())
        ax1.fill_between(np.arange(len(y)), 0, y, alpha=ALPHA_FILL, color=c)
    ax1.axhline(0, color="#666", lw=0.8, ls="--")
    ax1.set_title("Mark-to-Market PnL", fontsize=12)
    ax1.set_xlabel("Tick"); ax1.set_ylabel("PnL")
    ax1.legend(fontsize=9); ax1.grid(True)

    # Panel 2: Inventory
    ax2 = fig.add_subplot(gs[1, 0])
    for m in metrics_list:
        c = _colour(m.name)
        y = np.array(m.inventory_history)
        ax2.plot(y, lw=LINE_WIDTH, color=c, label=m.name.split("–")[0].strip(), alpha=0.85)
    ax2.axhline(0, color="#666", lw=0.8, ls="--")
    ax2.set_title("Inventory", fontsize=12)
    ax2.set_xlabel("Tick"); ax2.set_ylabel("Units")
    ax2.legend(fontsize=9); ax2.grid(True)

    # Panel 3: Drawdown
    ax3 = fig.add_subplot(gs[1, 1])
    for m in metrics_list:
        c   = _colour(m.name)
        pnl = np.array(m.pnl_history)
        dd  = pnl - np.maximum.accumulate(pnl)
        ax3.plot(dd, lw=LINE_WIDTH, color=c, label=m.name.split("–")[0].strip())
        ax3.fill_between(np.arange(len(dd)), dd, 0, alpha=ALPHA_FILL, color=c)
    ax3.axhline(0, color="#666", lw=0.8, ls="--")
    ax3.set_title("Drawdown", fontsize=12)
    ax3.set_xlabel("Tick"); ax3.set_ylabel("Drawdown")
    ax3.legend(fontsize=9); ax3.grid(True)

    # Legend handles for whole figure
    legend_handles = [
        Line2D([0], [0], color=_colour(m.name), lw=2, label=m.name)
        for m in metrics_list
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(metrics_list), fontsize=10,
               framealpha=0.3, facecolor="#161B22",
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Market Making Simulator — Strategy Comparison", fontsize=16, y=1.01)
    return fig
