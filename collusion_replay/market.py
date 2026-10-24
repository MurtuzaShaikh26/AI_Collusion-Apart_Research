"""Wrapper around the vendored llm-cartel OligopolyGame.

The demand system, Nash and monopoly prices, and the profit-gain formula come from
the pre-existing OligopolyGame. This module adds per-round cost updates (benchmarks
are rebuilt for that round's costs) and pure profit calculations used by replay and
by the unilateral-sacrifice signal.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from third_party.llm_cartel.environment.oligopoly import OligopolyGame


class Market:
    def __init__(
        self,
        costs=(1.0, 1.0),
        a0: float = 0.0,
        ais=(2.0, 2.0),
        mu: float = 0.25,
        beta: float = 100.0,
        alpha: float = 1.0,
    ):
        self.a0 = float(a0)
        self.ais = [float(a) for a in ais]
        self.mu = float(mu)
        self.beta = float(beta)
        self.alpha = float(alpha)
        if len(self.ais) != 2:
            raise ValueError("this harness is a two-firm market")
        self.costs = [float(c) for c in costs]
        self.log: list[dict] = []
        self.env: OligopolyGame | None = None
        self.nash_prices: list[float] = []
        self.monopoly_prices: list[float] = []
        self.nash_profits: list[float] = []
        self.monopoly_profits: list[float] = []
        self._rebuild()

    def _rebuild(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = OligopolyGame(
                alpha=self.alpha,
                a0=self.a0,
                ais=list(self.ais),
                costs=list(self.costs),
                mu=self.mu,
                beta=self.beta,
                prefix="P0",
                window_size=1,
                think=False,
                verbose=False,
            )
        self.nash_prices = [float(x) for x in self.env.lower_prices]
        self.monopoly_prices = [float(x) for x in self.env.upper_prices]
        self.nash_profits = [float(x) for x in self.env.lower_profits]
        self.monopoly_profits = [float(x) for x in self.env.upper_profits]

    def set_costs(self, costs) -> None:
        self.costs = [float(c) for c in costs]
        self._rebuild()

    def profits_for(self, prices, costs) -> tuple[list[float], list[float]]:
        """Quantities and profits. Matches OligopolyGame.step for symmetric alpha."""
        price_v = np.asarray(prices, dtype=float)
        cost_v = np.asarray(costs, dtype=float)
        base = np.exp(self.a0 / self.mu)
        demands = np.exp((np.asarray(self.ais, dtype=float) - price_v / self.alpha) / self.mu)
        demands = self.beta * demands / (demands.sum() + base)
        profits = demands * (price_v - self.alpha * cost_v)
        return demands.tolist(), profits.tolist()

    def best_response_price(self, agent: int, other_price: float, costs) -> tuple[float, float]:
        lo = 1e-3
        hi = max(float(c) for c in costs) * self.alpha * 8.0 + 5.0

        def objective(price: float) -> float:
            pair = [0.0, 0.0]
            pair[agent] = float(price)
            pair[1 - agent] = float(other_price)
            _demands, profits = self.profits_for(pair, costs)
            return -profits[agent]

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        return float(res.x), float(-res.fun)

    def sacrifice(self, agent: int, prices, costs) -> float:
        """Profit left on the table by not best-responding to the other firm's price."""
        other = float(prices[1 - agent])
        _br_price, br_profit = self.best_response_price(agent, other, costs)
        _demands, profits = self.profits_for(prices, costs)
        return float(br_profit - profits[agent])

    def step(self, prices, round_num: int) -> dict:
        if self.env is None:
            raise RuntimeError("market is not initialised")
        clean = [float(p) for p in prices]
        nash_p = list(self.nash_prices)
        mono_p = list(self.monopoly_prices)
        nash_pi = list(self.nash_profits)
        mono_pi = list(self.monopoly_profits)
        _prices, demands, profits, gains, _prompts = self.env.step(list(clean))
        record = {
            "round": int(round_num),
            "costs": list(self.costs),
            "actions": clean,
            "demands": [float(x) for x in demands],
            "profits": [float(x) for x in profits],
            "gains": [float(x) for x in gains],
            "nash_prices": nash_p,
            "monopoly_prices": mono_p,
            "nash_profits": nash_pi,
            "monopoly_profits": mono_pi,
        }
        self.log.append(record)
        return record
