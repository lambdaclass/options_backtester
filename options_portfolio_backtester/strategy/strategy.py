"""Strategy container — preserved interface with richer execution support."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from options_portfolio_backtester.execution.cost_model import TransactionCostModel, NoCosts
from options_portfolio_backtester.execution.sizer import PositionSizer, CapitalBased
from options_portfolio_backtester.execution.signal_selector import SignalSelector, FirstMatch

if TYPE_CHECKING:
    from options_portfolio_backtester.data.schema import Schema
    from .strategy_leg import StrategyLeg


class Strategy:
    """Options strategy — collection of legs with exit thresholds.

    API-compatible with backtester.strategy.strategy.Strategy, adding optional
    cost_model, sizer, and signal_selector at the strategy level.
    """

    def __init__(
        self,
        schema: "Schema",
        cost_model: TransactionCostModel | None = None,
        sizer: PositionSizer | None = None,
        signal_selector: SignalSelector | None = None,
    ) -> None:
        self.schema = schema
        self.legs: list[StrategyLeg] = []
        self.conditions: list = []
        self.exit_thresholds: tuple[float, float] = (math.inf, math.inf)
        self.cost_model = cost_model or NoCosts()
        self.sizer = sizer or CapitalBased()
        self.signal_selector = signal_selector or FirstMatch()

    def add_leg(self, leg: "StrategyLeg") -> "Strategy":
        assert self.schema == leg.schema
        leg.name = f"leg_{len(self.legs) + 1}"
        self.legs.append(leg)
        return self

    def add_legs(self, legs: list["StrategyLeg"]) -> "Strategy":
        for leg in legs:
            self.add_leg(leg)
        return self

    def remove_leg(self, leg_number: int) -> "Strategy":
        self.legs.pop(leg_number)
        return self

    def clear_legs(self) -> "Strategy":
        self.legs = []
        return self

    def add_exit_thresholds(self, profit_pct: float = math.inf,
                            loss_pct: float = math.inf) -> None:
        assert profit_pct >= 0
        assert loss_pct >= 0
        self.exit_thresholds = (profit_pct, loss_pct)

    def filter_thresholds(self, entry_cost: pd.Series,
                          current_cost: pd.Series) -> pd.Series:
        profit_pct, loss_pct = self.exit_thresholds
        excess_return = (current_cost / entry_cost + 1) * -np.sign(entry_cost)
        return (excess_return >= profit_pct) | (excess_return <= -loss_pct)

    def __repr__(self) -> str:
        return f"Strategy(legs={self.legs}, exit_thresholds={self.exit_thresholds})"
