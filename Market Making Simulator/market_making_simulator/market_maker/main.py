"""
main.py — Market Making Simulator
==================================

Run all three strategies on identical price paths, compare metrics,
and generate publication-quality charts.

Usage
-----
    python main.py                          # run with defaults
    python main.py --steps 10000           # longer simulation
    python main.py --seed 7 --no-plots     # reproducible, no plots
    python main.py --help                  # all options

Output
------
  - Console: formatted metrics table
  - Files:   results/dashboard.png
             results/summary_bars.png
             results/pnl.png
             results/inventory.png
             results/drawdown.png
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so relative imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")       # headless rendering for saving

from core.price_process import PriceProcessConfig
from core.exchange import Exchange, FlowConfig
from strategies.strategy_a import FixedSpreadMM
from strategies.strategy_b import InventoryAwareMM
from strategies.strategy_c import AvellanedaStoikovMM
from analytics.metrics import compute_metrics, print_report
from analytics.plots import (
    plot_dashboard, plot_summary_bar,
    plot_pnl, plot_inventory, plot_drawdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_strategy(strategy, n_steps: int, price_cfg: PriceProcessConfig,
                 flow_cfg: FlowConfig, seed: int):
    """Instantiate a fresh exchange and run one strategy."""
    ex = Exchange(n_steps=n_steps, price_cfg=price_cfg, flow_cfg=flow_cfg, seed=seed)
    account = ex.run(strategy)
    return account, ex.mid_history


def build_metrics_table(all_metrics) -> str:
    """Pretty ASCII comparison table."""
    cols = ["Strategy", "Total PnL", "Sharpe", "Max DD", "Inv Std", "Max|Inv|"]
    rows = []
    for m in all_metrics:
        short = m.name.split("–")[-1].strip()
        rows.append([
            short,
            f"{m.total_pnl:+.3f}",
            f"{m.sharpe:+.4f}",
            f"{m.max_drawdown:+.3f}",
            f"{m.inv_std:.3f}",
            f"{m.inv_max_abs:.2f}",
        ])

    widths = [max(len(cols[i]), max(len(r[i]) for r in rows)) + 2
              for i in range(len(cols))]
    sep  = "+" + "+".join("-" * w for w in widths) + "+"
    hdr  = "|" + "|".join(c.center(widths[i]) for i, c in enumerate(cols)) + "|"
    body = "\n".join(
        "|" + "|".join(r[i].center(widths[i]) for i in range(len(cols))) + "|"
        for r in rows
    )
    return "\n".join([sep, hdr, sep, body, sep])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Market Making Simulator")
    parser.add_argument("--steps",    type=int,   default=5_000,  help="Simulation length in ticks")
    parser.add_argument("--seed",     type=int,   default=42,     help="Random seed")
    parser.add_argument("--sigma",    type=float, default=0.015,  help="Mid-price vol per tick")
    parser.add_argument("--no-plots", action="store_true",        help="Skip plot generation")
    parser.add_argument("--output",   type=str,   default="results", help="Output directory for plots")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  MARKET MAKING SIMULATOR")
    print(f"  Ticks: {args.steps:,}   Seed: {args.seed}   Sigma: {args.sigma}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Price process & flow configuration
    # ------------------------------------------------------------------
    price_cfg = PriceProcessConfig(
        S0=100.0,
        sigma=args.sigma,
        mu=0.0,
        dt=1.0,
        # Mean-reverting (OU) process: realistic intraday behaviour
        mean_revert=True,
        kappa=0.05,           # speed of reversion
        theta=100.0,          # long-run mean
        # Jump component: models news shocks / large prints
        jumps=True,
        jump_intensity=0.003,
        jump_mean=0.0,
        jump_std=0.003,
    )

    flow_cfg = FlowConfig(
        noise_arrival_prob=0.25,    # noise traders
        noise_qty_min=1.0,
        noise_qty_max=3.0,
        informed_arrival_prob=0.03, # informed flow (adverse selection source)
        informed_qty_min=2.0,
        informed_qty_max=6.0,
        info_horizon=5,
    )

    # ------------------------------------------------------------------
    # Define strategies
    # ------------------------------------------------------------------
    strategies = [
        FixedSpreadMM(
            half_spread=0.10,
            order_qty=3.0,
            max_inventory=30.0,
        ),
        InventoryAwareMM(
            half_spread=0.08,
            skew_factor=0.005,
            spread_kappa=0.003,
            order_qty=3.0,
            max_inventory=30.0,
        ),
        AvellanedaStoikovMM(
            gamma=0.10,              # risk aversion
            kappa=2.0,               # order-arrival intensity
            sigma=args.sigma,
            T=float(args.steps),
            order_qty=3.0,
            max_inv=30.0,
            estimate_vol=True,
            vol_window=100,
        ),
    ]

    # ------------------------------------------------------------------
    # Run simulations (same seed → identical price path)
    # ------------------------------------------------------------------
    all_metrics = []
    mid_history  = None

    for strat in strategies:
        print(f"  Running {strat.name} ...", end=" ", flush=True)
        account, mh = run_strategy(strat, args.steps, price_cfg, flow_cfg, args.seed)
        if mid_history is None:
            mid_history = mh

        m = compute_metrics(strat.name, account.pnl_history, account.inventory_history)
        all_metrics.append(m)
        print("done")

    print()

    # ------------------------------------------------------------------
    # Print individual reports
    # ------------------------------------------------------------------
    for m in all_metrics:
        print_report(m)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("COMPARISON TABLE")
    print(build_metrics_table(all_metrics))
    print()

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        os.makedirs(args.output, exist_ok=True)

        print(f"  Generating plots → {args.output}/")

        dashboard = plot_dashboard(all_metrics, mid_history)
        dashboard.savefig(f"{args.output}/dashboard.png",
                          dpi=150, bbox_inches="tight",
                          facecolor=dashboard.get_facecolor())
        print(f"    ✓ dashboard.png")

        summary = plot_summary_bar(all_metrics)
        summary.savefig(f"{args.output}/summary_bars.png",
                        dpi=150, bbox_inches="tight",
                        facecolor=summary.get_facecolor())
        print(f"    ✓ summary_bars.png")

        for fig, fname in [
            (plot_pnl(all_metrics),       "pnl.png"),
            (plot_inventory(all_metrics), "inventory.png"),
            (plot_drawdown(all_metrics),  "drawdown.png"),
        ]:
            fig.savefig(f"{args.output}/{fname}",
                        dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"    ✓ {fname}")
            import matplotlib.pyplot as plt
            plt.close(fig)

        print()
        print(f"  All plots saved to ./{args.output}/")

    print()
    print("  Simulation complete.")
    print()


if __name__ == "__main__":
    main()
