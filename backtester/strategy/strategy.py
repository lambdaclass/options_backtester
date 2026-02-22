from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtester.datahandler.schema import Schema
    from .strategy_leg import StrategyLeg


class Strategy:
    """Options strategy class.
    Takes in a number of `StrategyLeg`'s (option contracts), and filters that determine
    entry and exit conditions.
    """
    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.legs: list[StrategyLeg] = []
        self.conditions: list = []
        self.exit_thresholds: tuple[float, float] = (math.inf, math.inf)

    def add_leg(self, leg: StrategyLeg) -> Strategy:
        """Adds leg to the strategy"""
        assert self.schema == leg.schema
        leg.name = "leg_{}".format(len(self.legs) + 1)
        self.legs.append(leg)
        return self

    def add_legs(self, legs: list[StrategyLeg]) -> Strategy:
        """Adds legs to the strategy"""
        for leg in legs:
            self.add_leg(leg)
        return self

    def remove_leg(self, leg_number: int) -> Strategy:
        """Removes leg from the strategy"""
        self.legs.pop(leg_number)
        return self

    def clear_legs(self) -> Strategy:
        """Removes *all* legs from the strategy"""
        self.legs = []
        return self

    def add_exit_thresholds(self, profit_pct: float = math.inf, loss_pct: float = math.inf) -> None:
        """Adds maximum profit/loss thresholds. Both **must** be >= 0.0

        Args:
            profit_pct (float, optional):   Max profit level. Defaults to math.inf
            loss_pct (float, optional):     Max loss level. Defaults to math.inf
        """

        assert profit_pct >= 0
        assert loss_pct >= 0
        self.exit_thresholds = (profit_pct, loss_pct)

    def filter_thresholds(self, entry_cost: pd.Series, current_cost: pd.Series) -> pd.Series:
        """Returns a `pd.Series` of booleans indicating where profit (loss) levels
        exceed the given thresholds.

        Args:
            entry_cost (pd.Series):     Total _entry_ cost of inventory row.
            current_cost (pd.Series):   Present cost of inventory row.

        Returns:
            pd.Series:                  Indicator series with `True` for every row that
            exceeds the specified profit/loss thresholds.
        """

        profit_pct, loss_pct = self.exit_thresholds

        excess_return = (current_cost / entry_cost + 1) * -np.sign(entry_cost)
        return (excess_return >= profit_pct) | (excess_return <= -loss_pct)

    def __repr__(self) -> str:
        return "Strategy(legs={}, exit_thresholds={})".format(self.legs, self.exit_thresholds)
