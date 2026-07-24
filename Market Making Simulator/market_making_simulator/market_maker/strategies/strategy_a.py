"""
Strategy A — Fixed-Spread Market Maker
=======================================

The simplest possible strategy: always quote a symmetric spread of
`half_spread` around the current mid price, for a fixed quantity.

This is the textbook baseline.  Its flaw: it ignores inventory accumulation,
so adverse selection and drift bleed the PnL over time.

Parameters
----------
half_spread : float
    Half the quoted spread (in price units). Total spread = 2 * half_spread.
order_qty : float
    Size of each limit order (bid and ask).
max_inventory : float
    If |inventory| exceeds this, pull quotes entirely until inventory normalises.
"""

from __future__ import annotations
from strategies.base import BaseStrategy
from core.exchange import Exchange


class FixedSpreadMM(BaseStrategy):
    name = "Strategy A – Fixed Spread"

    def __init__(
        self,
        half_spread:    float = 0.10,
        order_qty:      float = 5.0,
        max_inventory:  float = 50.0,
    ):
        self.half_spread   = half_spread
        self.order_qty     = order_qty
        self.max_inventory = max_inventory

    def on_tick(self, exchange: Exchange) -> None:
        mid = exchange.mid
        inv = exchange.account.inventory

        # Risk check: stop quoting if inventory is too large
        if abs(inv) >= self.max_inventory:
            exchange.cancel_quotes()
            return

        bid_price = mid - self.half_spread
        ask_price = mid + self.half_spread

        exchange.post_quotes(
            bid_price=bid_price,
            bid_qty=self.order_qty,
            ask_price=ask_price,
            ask_qty=self.order_qty,
        )
