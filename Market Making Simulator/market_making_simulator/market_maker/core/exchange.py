"""
Exchange — wraps the order book with:
  - Simulated external order flow (noise traders / informed traders)
  - Market-maker interface (quote / cancel / inventory tracking)
  - Tick-by-tick simulation loop
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.order_book import OrderBook, Order, Side, Trade
from core.price_process import MidPriceProcess, PriceProcessConfig


# ---------------------------------------------------------------------------
# External flow configuration
# ---------------------------------------------------------------------------

@dataclass
class FlowConfig:
    """Parameters governing exogenous (noise + informed) order flow."""
    # Probability a noise trader arrives each tick
    noise_arrival_prob: float = 0.30
    # Noise trader order sizes (uniform draw)
    noise_qty_min: float = 1.0
    noise_qty_max: float = 5.0

    # Probability an informed trader arrives each tick
    informed_arrival_prob: float = 0.05
    # Informed trader order sizes
    informed_qty_min: float = 3.0
    informed_qty_max: float = 10.0
    # How far ahead the informed trader can see (in ticks)
    info_horizon: int = 5


# ---------------------------------------------------------------------------
# Market maker position / PnL tracking
# ---------------------------------------------------------------------------

@dataclass
class MMAccount:
    cash:      float = 0.0
    inventory: float = 0.0       # positive = long, negative = short
    realised_pnl: float = 0.0

    # Recorded per-tick for analytics
    cash_history:      List[float] = field(default_factory=list)
    inventory_history: List[float] = field(default_factory=list)
    pnl_history:       List[float] = field(default_factory=list)   # total (mark-to-market)
    spread_earned:     List[float] = field(default_factory=list)

    def mark_to_market(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def record(self, mid: float) -> None:
        self.cash_history.append(self.cash)
        self.inventory_history.append(self.inventory)
        self.pnl_history.append(self.mark_to_market(mid))


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------

class Exchange:
    """
    Simulation engine.

    Usage:
        ex = Exchange(n_steps=5000)
        ex.run(strategy)   # strategy must implement .on_tick(exchange) -> quotes
    """

    def __init__(
        self,
        n_steps:   int = 5_000,
        price_cfg: Optional[PriceProcessConfig] = None,
        flow_cfg:  Optional[FlowConfig] = None,
        seed:      int = 42,
    ):
        self.n_steps  = n_steps
        self.price_cfg = price_cfg or PriceProcessConfig()
        self.flow_cfg  = flow_cfg  or FlowConfig()
        self.rng = np.random.default_rng(seed)

        self.book    = OrderBook()
        self.process = MidPriceProcess(self.price_cfg, seed=seed)
        self.account = MMAccount()

        self.tick:    int   = 0
        self.mid:     float = self.price_cfg.S0
        self.mid_history: List[float] = []

        # Active MM quotes (order_ids)
        self._bid_id: Optional[int] = None
        self._ask_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, strategy) -> MMAccount:
        """Run simulation; returns filled MMAccount."""
        self.book    = OrderBook()
        self.process = MidPriceProcess(self.price_cfg, seed=int(self.rng.integers(1e6)))
        self.account = MMAccount()
        self.mid     = self.price_cfg.S0
        self.mid_history = []
        self._bid_id = None
        self._ask_id = None

        for t in range(self.n_steps):
            self.tick = t
            self.mid  = self.process.step()
            self.mid_history.append(self.mid)

            # 1. Strategy quotes
            strategy.on_tick(self)

            # 2. External flow hits the book
            self._simulate_flow(t)

            # 3. Process any fills for the MM
            self._process_mm_fills()

            # 4. Record state
            self.account.record(self.mid)

        return self.account

    # ------------------------------------------------------------------
    # MM quoting interface (called by strategies)
    # ------------------------------------------------------------------

    def post_quotes(self, bid_price: float, bid_qty: float,
                         ask_price: float, ask_qty: float) -> None:
        """Cancel existing MM quotes and post fresh ones."""
        # Cancel stale quotes
        if self._bid_id is not None:
            self.book.cancel(self._bid_id)
        if self._ask_id is not None:
            self.book.cancel(self._ask_id)

        # Post new quotes
        if bid_qty > 0 and bid_price > 0:
            o = self.book.submit_limit(Side.BID, round(bid_price, 4), bid_qty)
            self._bid_id = o.order_id
        else:
            self._bid_id = None

        if ask_qty > 0 and ask_price > 0:
            o = self.book.submit_limit(Side.ASK, round(ask_price, 4), ask_qty)
            self._ask_id = o.order_id
        else:
            self._ask_id = None

    def cancel_quotes(self) -> None:
        if self._bid_id is not None:
            self.book.cancel(self._bid_id)
            self._bid_id = None
        if self._ask_id is not None:
            self.book.cancel(self._ask_id)
            self._ask_id = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_flow(self, t: int) -> None:
        cfg = self.flow_cfg

        # Noise traders (random direction market orders)
        if self.rng.random() < cfg.noise_arrival_prob:
            qty  = self.rng.uniform(cfg.noise_qty_min, cfg.noise_qty_max)
            side = Side.BID if self.rng.random() < 0.5 else Side.ASK
            self.book.submit_market(side, qty)

        # Informed traders (directional market orders)
        if self.rng.random() < cfg.informed_arrival_prob:
            # Look ahead `info_horizon` ticks in the price path (imperfect signal)
            future_idx = min(t + cfg.info_horizon, len(self.process.history) - 1)
            future_mid = self.process.history[future_idx] if future_idx < len(self.process.history) else self.mid
            qty  = self.rng.uniform(cfg.informed_qty_min, cfg.informed_qty_max)
            side = Side.BID if future_mid > self.mid else Side.ASK
            self.book.submit_market(side, qty)

    def _process_mm_fills(self) -> None:
        """
        Scan recent trades for fills against MM's resting orders;
        update cash & inventory accordingly.
        """
        mm_order_ids = set()
        if self._bid_id is not None:
            mm_order_ids.add(self._bid_id)
        if self._ask_id is not None:
            mm_order_ids.add(self._ask_id)

        for trade in self.book.trades:
            if trade.passive not in mm_order_ids:
                continue
            # Avoid double-counting — mark processed
            if hasattr(trade, "_mm_processed"):
                continue
            trade._mm_processed = True  # type: ignore[attr-defined]

            passive_order = self.book.get_order(trade.passive)
            if passive_order is None:
                continue

            if passive_order.side == Side.BID:
                # MM bought: pay cash, gain inventory
                self.account.cash      -= trade.price * trade.quantity
                self.account.inventory += trade.quantity
            else:
                # MM sold: receive cash, lose inventory
                self.account.cash      += trade.price * trade.quantity
                self.account.inventory -= trade.quantity
