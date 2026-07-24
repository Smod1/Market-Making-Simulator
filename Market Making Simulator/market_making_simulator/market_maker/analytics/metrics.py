"""
Analytics — performance metrics for market-making strategies.

Metrics computed:
  - Total PnL (mark-to-market)
  - Realised PnL
  - Sharpe ratio (annualised, per-tick)
  - Maximum drawdown
  - Inventory statistics (mean, std, max absolute)
  - Turnover
  - Fill rate (how often quotes are hit)
  - Spread captured (estimated)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class StrategyMetrics:
    name:             str
    total_pnl:        float
    sharpe:           float
    max_drawdown:     float
    inv_mean:         float
    inv_std:          float
    inv_max_abs:      float
    pnl_std:          float
    pnl_history:      List[float]
    inventory_history: List[float]


def compute_metrics(name: str, pnl_history: List[float],
                    inventory_history: List[float]) -> StrategyMetrics:
    """
    Compute performance metrics from per-tick PnL and inventory series.

    Parameters
    ----------
    name              : strategy label
    pnl_history       : mark-to-market total equity at each tick
    inventory_history : inventory at each tick
    """
    pnl  = np.array(pnl_history, dtype=float)
    inv  = np.array(inventory_history, dtype=float)

    # Per-tick returns
    returns = np.diff(pnl)

    total_pnl    = float(pnl[-1] - pnl[0]) if len(pnl) > 0 else 0.0
    pnl_std      = float(np.std(returns))   if len(returns) > 0 else 0.0

    # Sharpe: mean / std of per-tick returns (no annualisation factor here;
    # interpret as information ratio per simulation run)
    mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
    sharpe   = (mean_ret / pnl_std) if pnl_std > 1e-12 else 0.0

    # Maximum drawdown
    running_max  = np.maximum.accumulate(pnl)
    drawdowns    = pnl - running_max
    max_drawdown = float(np.min(drawdowns))

    # Inventory stats
    inv_mean    = float(np.mean(inv))
    inv_std     = float(np.std(inv))
    inv_max_abs = float(np.max(np.abs(inv)))

    return StrategyMetrics(
        name=name,
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        inv_mean=inv_mean,
        inv_std=inv_std,
        inv_max_abs=inv_max_abs,
        pnl_std=pnl_std,
        pnl_history=pnl_history,
        inventory_history=inventory_history,
    )


def print_report(metrics: StrategyMetrics) -> None:
    w = 50
    print("=" * w)
    print(f"  {metrics.name}")
    print("=" * w)
    print(f"  {'Total PnL':30s}: {metrics.total_pnl:+.4f}")
    print(f"  {'Sharpe Ratio':30s}: {metrics.sharpe:+.4f}")
    print(f"  {'Max Drawdown':30s}: {metrics.max_drawdown:+.4f}")
    print(f"  {'PnL Std Dev':30s}: {metrics.pnl_std:.4f}")
    print(f"  {'Mean Inventory':30s}: {metrics.inv_mean:+.4f}")
    print(f"  {'Inventory Std Dev':30s}: {metrics.inv_std:.4f}")
    print(f"  {'Max |Inventory|':30s}: {metrics.inv_max_abs:.4f}")
    print("=" * w)
    print()
