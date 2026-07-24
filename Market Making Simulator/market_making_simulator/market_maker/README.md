# Market Making Simulator

A production-quality simulation of a limit order book exchange with three
market-making strategies of increasing sophistication.

Built from scratch in Python — no external trading libraries.

---

## Overview

This project implements a full simulation stack:

```
Exchange
├── core/
│   ├── order_book.py       — Price-time priority LOB (limit & market orders)
│   ├── price_process.py    — GBM / Ornstein-Uhlenbeck mid-price with jumps
│   └── exchange.py         — Simulation engine + external order flow
├── strategies/
│   ├── strategy_a.py       — Fixed-spread market maker (baseline)
│   ├── strategy_b.py       — Inventory-aware market maker
│   └── strategy_c.py       — Avellaneda-Stoikov (optimal control)
├── analytics/
│   ├── metrics.py          — PnL, Sharpe, drawdown, inventory stats
│   └── plots.py            — Publication-quality matplotlib charts
└── main.py                 — Runner + CLI
```

---

## Strategies

### Strategy A — Fixed Spread
The textbook baseline. Posts symmetric limit orders at `mid ± half_spread`
with constant quantity. Ignores inventory — accumulates directional risk.

### Strategy B — Inventory-Aware
Extends A with **quote skewing**: the reservation price shifts to incentivise
inventory reduction.

```
reservation = mid - gamma * inventory * sigma^2
spread      = base_spread + kappa * |inventory|
```

Reduces inventory variance at the cost of slightly wider spreads.

### Strategy C — Avellaneda-Stoikov (2008)
Optimal closed-form solution derived from maximising expected CARA utility
of terminal wealth over a finite horizon.

**Reservation price:**
```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

**Optimal spread:**
```
delta* = gamma * sigma^2 * (T - t)  +  (2/gamma) * ln(1 + gamma/kappa)
```

The spread widens dynamically as:
- Time horizon shortens (urgency to close inventory increases)
- Volatility rises (adverse selection risk increases)
- Risk aversion (gamma) increases

---

## Results (5,000 ticks, seed=42)

| Strategy            | Total PnL | Sharpe  | Max Drawdown |
|---------------------|-----------|---------|--------------|
| A — Fixed Spread    | +57.13    | +0.0164 | -28.80       |
| B — Inventory-Aware | +68.20    | +0.0190 | -29.88       |
| C — Avellaneda-Stoikov | **+262.49** | **+0.0717** | -28.80 |

Strategy C achieves **4.6× the PnL** and **4.4× the Sharpe ratio** of the
fixed-spread baseline, demonstrating the value of optimal spread computation.

---

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run with defaults (5,000 ticks)
python main.py

# Longer run, custom seed
python main.py --steps 10000 --seed 7

# All options
python main.py --help
```

Plots are saved to `./results/`.

---

## Price Process

The mid-price follows an **Ornstein-Uhlenbeck** (mean-reverting) process
with a **Poisson jump** component:

```
dS = kappa * (theta - S) * dt  +  sigma * dW  +  J * dN
```

where `dN` is a Poisson process (jump arrivals) and `J ~ N(0, sigma_jump)`.

This models realistic intraday price dynamics with occasional news shocks.

---

## Order Flow Model

Two types of external agents hit the book each tick:

- **Noise traders** (30% arrival probability): random direction, small size.
  These are the MM's profit source — they pay the spread.

- **Informed traders** (3% arrival probability): directional, larger size.
  They have a noisy signal of future price — the source of adverse selection.

The tension between capturing noise-trader spread and being adversely
selected by informed flow is the central trade-off in market making.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Adverse selection | Informed traders hit MM quotes, causing inventory accumulation |
| Reservation price | The price at which the MM is indifferent to buying/selling |
| Inventory risk | Holding a position exposes the MM to directional price moves |
| Spread | Revenue per round-trip; must exceed adverse selection losses |
| Sharpe ratio | Risk-adjusted return; penalises volatile PnL |

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit
  order book.* Quantitative Finance, 8(3), 217–224.
- Glosten, L. & Milgrom, P. (1985). *Bid, ask and transaction prices in a
  specialist market with heterogeneously informed traders.* Journal of
  Financial Economics, 14(1), 71–100.
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and
  High-Frequency Trading.* Cambridge University Press.
