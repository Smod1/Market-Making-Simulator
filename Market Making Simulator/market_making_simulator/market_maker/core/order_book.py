"""
Limit Order Book (LOB) — price-time priority matching engine.

Supports:
  - Limit orders (GTC)
  - Market orders
  - Order cancellation
  - Best bid/ask, mid-price, spread queries
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Side(Enum):
    BID = auto()   # buy side
    ASK = auto()   # sell side


class OrderType(Enum):
    LIMIT  = auto()
    MARKET = auto()


class OrderStatus(Enum):
    OPEN      = auto()
    PARTIALLY_FILLED = auto()
    FILLED    = auto()
    CANCELLED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id:   int
    side:       Side
    order_type: OrderType
    price:      Optional[float]   # None for market orders
    quantity:   float
    timestamp:  float = field(default_factory=time.time)
    filled_qty: float = 0.0
    status:     OrderStatus = OrderStatus.OPEN

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)


@dataclass
class Trade:
    trade_id:    int
    price:       float
    quantity:    float
    aggressor:   int      # order_id of the aggressor
    passive:     int      # order_id of the resting order
    timestamp:   float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Price-time priority limit order book.

    Internal structure:
        bids: {price -> [Order, ...]}  (descending)
        asks: {price -> [Order, ...]}  (ascending)
    """

    def __init__(self, symbol: str = "ASSET"):
        self.symbol = symbol

        # price level -> list of resting orders (FIFO within level)
        self._bids: Dict[float, List[Order]] = defaultdict(list)
        self._asks: Dict[float, List[Order]] = defaultdict(list)

        self._orders: Dict[int, Order] = {}
        self._trades: List[Trade] = []

        self._order_counter = 0
        self._trade_counter = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit_limit(self, side: Side, price: float, quantity: float) -> Order:
        """Place a resting limit order; returns the Order object."""
        order = self._new_order(side, OrderType.LIMIT, price, quantity)
        self._match(order)
        if order.is_active:
            self._add_to_book(order)
        return order

    def submit_market(self, side: Side, quantity: float) -> Order:
        """Place an aggressive market order; matches immediately."""
        order = self._new_order(side, OrderType.MARKET, None, quantity)
        self._match(order)
        if order.remaining > 0:
            # Unfilled residual — mark cancelled (no price to rest at)
            order.status = OrderStatus.CANCELLED
        return order

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting limit order. Returns True if successful."""
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False
        order.status = OrderStatus.CANCELLED
        book = self._bids if order.side == Side.BID else self._asks
        level = book.get(order.price, [])
        try:
            level.remove(order)
        except ValueError:
            pass
        if not level:
            book.pop(order.price, None)
        return True

    # ------------------------------------------------------------------
    # Book queries
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys()) if self._asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def depth(self, levels: int = 5) -> Dict:
        """Return top-N price levels on each side."""
        bid_levels = sorted(self._bids.keys(), reverse=True)[:levels]
        ask_levels = sorted(self._asks.keys())[:levels]

        bids = [(p, sum(o.remaining for o in self._bids[p])) for p in bid_levels]
        asks = [(p, sum(o.remaining for o in self._asks[p])) for p in ask_levels]
        return {"bids": bids, "asks": asks}

    @property
    def trades(self) -> List[Trade]:
        return list(self._trades)

    def get_order(self, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_order(self, side, order_type, price, quantity) -> Order:
        self._order_counter += 1
        order = Order(
            order_id=self._order_counter,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
        )
        self._orders[order.order_id] = order
        return order

    def _add_to_book(self, order: Order) -> None:
        if order.side == Side.BID:
            self._bids[order.price].append(order)
        else:
            self._asks[order.price].append(order)

    def _match(self, aggressor: Order) -> None:
        """Core matching loop (price-time priority)."""
        if aggressor.side == Side.BID:
            resting_book = self._asks
            price_ok = lambda rp: (aggressor.price is None) or (rp <= aggressor.price)
            sorted_levels = lambda: sorted(resting_book.keys())
        else:
            resting_book = self._bids
            price_ok = lambda rp: (aggressor.price is None) or (rp >= aggressor.price)
            sorted_levels = lambda: sorted(resting_book.keys(), reverse=True)

        while aggressor.remaining > 0:
            levels = sorted_levels()
            if not levels:
                break
            best_price = levels[0]
            if not price_ok(best_price):
                break

            queue = resting_book[best_price]
            while queue and aggressor.remaining > 0:
                passive = queue[0]
                if not passive.is_active:
                    queue.pop(0)
                    continue

                fill_qty = min(aggressor.remaining, passive.remaining)
                fill_price = passive.price  # passive sets the price

                # Record fills
                self._trade_counter += 1
                trade = Trade(
                    trade_id=self._trade_counter,
                    price=fill_price,
                    quantity=fill_qty,
                    aggressor=aggressor.order_id,
                    passive=passive.order_id,
                )
                self._trades.append(trade)

                aggressor.filled_qty += fill_qty
                passive.filled_qty   += fill_qty

                aggressor.status = (
                    OrderStatus.FILLED if aggressor.remaining == 0
                    else OrderStatus.PARTIALLY_FILLED
                )
                passive.status = (
                    OrderStatus.FILLED if passive.remaining == 0
                    else OrderStatus.PARTIALLY_FILLED
                )

                if passive.remaining == 0:
                    queue.pop(0)

            if not queue:
                del resting_book[best_price]
