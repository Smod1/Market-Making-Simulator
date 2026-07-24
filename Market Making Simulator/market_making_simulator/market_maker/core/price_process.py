"""
Simulated mid-price process for the exchange.

Uses Geometric Brownian Motion (GBM) with optional mean-reversion
(Ornstein-Uhlenbeck) and Poisson jump component to mimic realistic
intraday price dynamics.

  dS = mu*S*dt + sigma*S*dW  (GBM)
  or
  dS = kappa*(theta - S)*dt + sigma*dW  (OU, mean-reverting)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class PriceProcessConfig:
    S0:       float = 100.0       # initial price
    sigma:    float = 0.02        # volatility per step (annualised: 0.20 / sqrt(252*steps_per_day))
    mu:       float = 0.0         # drift per step (usually 0 for intraday)
    dt:       float = 1.0         # time step (arbitrary units, e.g. 1 second)

    # Optional mean-reversion (OU)
    mean_revert: bool  = False
    kappa:       float = 0.1      # speed of mean reversion
    theta:       float = 100.0    # long-run mean

    # Optional jumps
    jumps:      bool  = False
    jump_intensity: float = 0.01  # Poisson rate per step
    jump_mean:  float = 0.0
    jump_std:   float = 0.005


class MidPriceProcess:
    """
    Generates a sequence of mid-prices representing the 'true' fair value
    of the asset.  Market makers observe this indirectly through the order
    flow they face.
    """

    def __init__(self, cfg: PriceProcessConfig, seed: int = 42):
        self.cfg = cfg
        self.rng  = np.random.default_rng(seed)
        self.price = cfg.S0
        self.history: List[float] = [cfg.S0]

    def step(self) -> float:
        cfg = self.cfg
        S = self.price

        if cfg.mean_revert:
            # Ornstein-Uhlenbeck
            drift    = cfg.kappa * (cfg.theta - S) * cfg.dt
            diffusion = cfg.sigma * np.sqrt(cfg.dt) * self.rng.standard_normal()
            dS = drift + diffusion
        else:
            # GBM (log-normal)
            z  = self.rng.standard_normal()
            dS = cfg.mu * S * cfg.dt + cfg.sigma * S * np.sqrt(cfg.dt) * z

        # Add jump component
        if cfg.jumps:
            n_jumps = self.rng.poisson(cfg.jump_intensity * cfg.dt)
            if n_jumps > 0:
                jump_size = self.rng.normal(cfg.jump_mean, cfg.jump_std, n_jumps).sum()
                dS += S * jump_size

        self.price = max(self.price + dS, 1e-3)   # keep price positive
        self.history.append(self.price)
        return self.price

    def run(self, n_steps: int) -> np.ndarray:
        """Generate n_steps of prices.  Resets state."""
        self.price = self.cfg.S0
        self.history = [self.cfg.S0]
        for _ in range(n_steps):
            self.step()
        return np.array(self.history)

    # ------------------------------------------------------------------
    # Convenience: realised volatility of generated path
    # ------------------------------------------------------------------

    @staticmethod
    def realised_vol(prices: np.ndarray) -> float:
        log_ret = np.diff(np.log(prices))
        return float(np.std(log_ret))
