"""Multi-strategy engine — run N strategies with shared capital and risk budget."""

from __future__ import annotations

from typing import Any

import pandas as pd

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import TransactionCostModel, NoCosts
from options_portfolio_backtester.portfolio.risk import RiskManager

from options_portfolio_backtester.core.types import Stock


class StrategyAllocation:
    """Configuration for one strategy within a multi-strategy engine."""

    def __init__(
        self,
        name: str,
        engine: BacktestEngine,
        weight: float = 1.0,
    ) -> None:
        self.name = name
        self.engine = engine
        self.weight = weight


class MultiStrategyEngine:
    """Run multiple strategies with shared capital allocation.

    Each strategy gets a fraction of total capital proportional to its weight.
    Results are combined into a single balance sheet.
    """

    def __init__(
        self,
        strategies: list[StrategyAllocation],
        initial_capital: int = 1_000_000,
    ) -> None:
        self.strategies = strategies
        self.initial_capital = initial_capital
        total_weight = sum(s.weight for s in strategies)
        self._weights = {s.name: s.weight / total_weight for s in strategies}

    def run(self, rebalance_freq: int = 0, monthly: bool = False,
            sma_days: int | None = None) -> dict[str, pd.DataFrame]:
        """Run all strategies and return per-strategy trade logs.

        Returns:
            Dict mapping strategy name to its trade log DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}

        for sa in self.strategies:
            capital_share = int(self.initial_capital * self._weights[sa.name])
            # Override the engine's initial capital with its share
            sa.engine.initial_capital = capital_share
            trade_log = sa.engine.run(
                rebalance_freq=rebalance_freq,
                monthly=monthly,
                sma_days=sma_days,
            )
            results[sa.name] = trade_log

        # Build combined balance
        self._build_combined_balance()
        return results

    def _build_combined_balance(self) -> None:
        """Combine balance sheets from all strategies."""
        balances = []
        for sa in self.strategies:
            if hasattr(sa.engine, "balance"):
                b = sa.engine.balance[["total capital", "% change"]].copy()
                b.columns = [f"{sa.name}_capital", f"{sa.name}_pct_change"]
                balances.append(b)

        if balances:
            self.balance = pd.concat(balances, axis=1)
            capital_cols = [f"{sa.name}_capital" for sa in self.strategies]
            self.balance["total capital"] = self.balance[capital_cols].sum(axis=1)
            self.balance["% change"] = self.balance["total capital"].pct_change()
            self.balance["accumulated return"] = (
                1.0 + self.balance["% change"]
            ).cumprod()
        else:
            self.balance = pd.DataFrame()
