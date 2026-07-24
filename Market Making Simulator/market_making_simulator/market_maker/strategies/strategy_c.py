"""
Strategy C — Avellaneda-Stoikov Market Maker
=============================================

Implements the closed-form solution from:
    Avellaneda, M. & Stoikov, S. (2008).
    "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.

The model
---------
The MM maximises expected exponential utility of terminal wealth W(T):

    U(W) = -exp(-gamma * W)

Given inventory q and time-to-horizon T, the *reservation price* is:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

where:
    s      = current mid price
    q      = inventory
    gamma  = absolute risk aversion
    sigma  = volatility
    T - t  = time remaining (fraction of horizon)

The *optimal spread* (sum of both half-spreads) is:

    delta^* = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)

where kappa is the order-arrival intensity (how quickly limit orders fill).

The strategy posts:
    bid = r - delta^*/2
    ask = r + delta^*/2

Key properties
--------------
- Reservation price skews quotes to reduce inventory risk
- Spread widens as time horizon shrinks (risk aversion increases near close)
- Spread widens with volatility
- Spread narrows with higher order arrival intensity (more fill opportunities)

Parameters
----------
gamma       : float  — risk aversion (0 → risk-neutral → narrow spread)
kappa       : float  — order arrival intensity (higher → tighter spread)
sigma       : float  — volatility (or estimated online)
T           : float  — total horizon in 'ticks'
order_qty   : float  — size per quote
max_inv     : float  — max |inventory| before quotes are pulled
estimate_vol: bool   — if True, estimate sigma online from price history
vol_window  : int    — rolling window for vol estimation
"""

from __future__ import annotations

import numpy as np
from strategies.base import BaseStrategy
from core.exchange import Exchange


class AvellanedaStoikovMM(BaseStrategy):
    name = "Strategy C – Avellaneda-Stoikov"

    def __init__(
        self,
        gamma:        float = 0.05,
        kappa:        float = 1.5,
        sigma:        float = 0.02,
        T:            float = 5_000.0,
        order_qty:    float = 5.0,
        max_inv:      float = 50.0,
        estimate_vol: bool  = True,
        vol_window:   int   = 100,
    ):
        self.gamma        = gamma
        self.kappa        = kappa
        self._sigma_fixed = sigma
        self.T            = T
        self.order_qty    = order_qty
        self.max_inv      = max_inv
        self.estimate_vol = estimate_vol
        self.vol_window   = vol_window

    # ------------------------------------------------------------------

    def _get_sigma(self, exchange: Exchange) -> float:
        if not self.estimate_vol:
            return self._sigma_fixed
        h = exchange.mid_history
        if len(h) < max(2, self.vol_window // 4):
            return self._sigma_fixed
        window = h[-self.vol_window:]
        log_rets = np.diff(np.log(np.maximum(window, 1e-9)))
        if len(log_rets) == 0:
            return self._sigma_fixed
        est = float(np.std(log_rets))
        # Blend with prior to avoid instability early on
        alpha = min(len(h) / self.vol_window, 1.0)
        return alpha * est + (1 - alpha) * self._sigma_fixed

    def on_tick(self, exchange: Exchange) -> None:
        mid = exchange.mid
        q   = exchange.account.inventory
        t   = exchange.tick

        if abs(q) >= self.max_inv:
            exchange.cancel_quotes()
            return

        sigma    = self._get_sigma(exchange)
        tau      = max((self.T - t) / self.T, 1e-6)   # normalised time remaining

        # Reservation price
        reservation = mid - q * self.gamma * (sigma ** 2) * tau

        # Optimal total spread (Avellaneda-Stoikov closed form)
        # delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)
        term1 = self.gamma * (sigma ** 2) * tau
        term2 = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        total_spread = term1 + term2
        half_spread  = max(total_spread / 2.0, 1e-4)   # minimum tick floor

        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        exchange.post_quotes(
            bid_price=bid_price,
            bid_qty=self.order_qty,
            ask_price=ask_price,
            ask_qty=self.order_qty,
        )

    # ------------------------------------------------------------------
    # Analytical helpers (useful for write-ups)
    # ------------------------------------------------------------------

    @staticmethod
    def theoretical_spread(
        gamma: float, sigma: float, kappa: float, tau: float = 1.0
    ) -> float:
        """Return the A-S optimal spread analytically."""
        return gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / kappa)

    @staticmethod
    def reservation_price(s: float, q: float, gamma: float,
                          sigma: float, tau: float) -> float:
        return s - q * gamma * sigma**2 * tau
