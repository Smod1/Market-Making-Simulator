"""Base class all market-making strategies inherit from."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.exchange import Exchange


class BaseStrategy(ABC):
    """
    Minimal interface: implement `on_tick(exchange)`.

    Each tick the strategy may:
      - Read exchange.mid (current mid price)
      - Read exchange.account (inventory, cash, PnL)
      - Call exchange.post_quotes(bid_p, bid_q, ask_p, ask_q)
      - Call exchange.cancel_quotes()
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def on_tick(self, exchange: "Exchange") -> None:
        ...
