"""
Strategy B — Inventory-Aware Market Maker
==========================================

Extends the fixed-spread baseline by skewing quotes toward reducing
inventory exposure.

Key ideas
---------
1. **Quote skew**: if the MM is long inventory, lower *both* bid and ask
   to incentivise selling and discourage more buying.

       reservation_price = mid - gamma * inventory * sigma^2 * T_remaining

   where gamma is the risk-aversion coefficient.

2. **Spread widening**: widen the spread when inventory risk is high.

       spread = base_spread + kappa * |inventory|

This keeps the MM profitable while explicitly managing directional risk —
a necessary step beyond Strategy A.

Parameters
----------
half_spread   : float  — minimum half-spread
skew_factor   : float  — gamma-like coefficient for reservation price shift
spread_kappa  : float  — additional spread per unit of inventory
order_qty     : float  — base order size
max_inventory : float  — absolute inventory limit (quotes pulled beyond this)
"""

from __future__ import annotations
import numpy as np
from strategies.base import BaseStrategy
from core.exchange import Exchange


class InventoryAwareMM(BaseStrategy):
    name = "Strategy B – Inventory-Aware"

    def __init__(
        self,
        half_spread:    float = 0.08,
        skew_factor:    float = 0.002,
        spread_kappa:   float = 0.002,
        order_qty:      float = 5.0,
        max_inventory:  float = 50.0,
        vol_window:     int   = 50,
    ):
        self.half_spread   = half_spread
        self.skew_factor   = skew_factor
        self.spread_kappa  = spread_kappa
        self.order_qty     = order_qty
        self.max_inventory = max_inventory
        self.vol_window    = vol_window

    def _estimate_vol(self, exchange: Exchange) -> float:
        """Rolling realised volatility from mid-price history."""
        h = exchange.mid_history
        if len(h) < 2:
            return exchange.price_cfg.sigma
        window = h[-self.vol_window:]
        log_rets = np.diff(np.log(window))
        return float(np.std(log_rets)) if len(log_rets) > 0 else exchange.price_cfg.sigma

    def on_tick(self, exchange: Exchange) -> None:
        mid = exchange.mid
        inv = exchange.account.inventory

        if abs(inv) >= self.max_inventory:
            exchange.cancel_quotes()
            return

        sigma = self._estimate_vol(exchange)

        # Reservation price: shift mid toward reducing inventory
        reservation = mid - self.skew_factor * inv * (sigma ** 2)

        # Spread: widen with inventory
        dynamic_half_spread = self.half_spread + self.spread_kappa * abs(inv)

        # Asymmetric sizing: reduce quote on the side that would worsen inventory
        if inv > 0:
            # Long: make it easier to sell (tighter ask), harder to buy (wider bid)
            bid_qty = max(self.order_qty - 0.5 * abs(inv) / self.max_inventory * self.order_qty, 1.0)
            ask_qty = self.order_qty
        elif inv < 0:
            # Short: easier to buy, harder to sell
            bid_qty = self.order_qty
            ask_qty = max(self.order_qty - 0.5 * abs(inv) / self.max_inventory * self.order_qty, 1.0)
        else:
            bid_qty = ask_qty = self.order_qty

        bid_price = reservation - dynamic_half_spread
        ask_price = reservation + dynamic_half_spread

        exchange.post_quotes(
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=ask_price,
            ask_qty=ask_qty,
        )
