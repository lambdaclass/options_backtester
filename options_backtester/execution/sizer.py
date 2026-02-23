"""Position sizing models — determine how many contracts to trade."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PositionSizer(ABC):
    """Determines the number of contracts to trade."""

    @abstractmethod
    def size(self, cost_per_contract: float, available_capital: float,
             total_capital: float) -> int:
        """Return the number of contracts to trade.

        Args:
            cost_per_contract: Dollar cost for one contract (absolute value).
            available_capital: Capital allocated to this trade.
            total_capital: Total portfolio value.
        """
        ...


class CapitalBased(PositionSizer):
    """Buy as many contracts as the allocation allows — matches original behavior.

    qty = available_capital // cost_per_contract
    """

    def size(self, cost_per_contract: float, available_capital: float,
             total_capital: float) -> int:
        if cost_per_contract == 0:
            return 0
        return int(available_capital // abs(cost_per_contract))


class FixedQuantity(PositionSizer):
    """Always trade a fixed number of contracts."""

    def __init__(self, quantity: int = 1) -> None:
        self.quantity = quantity

    def size(self, cost_per_contract: float, available_capital: float,
             total_capital: float) -> int:
        if abs(cost_per_contract) * self.quantity > available_capital:
            return int(available_capital // abs(cost_per_contract)) if cost_per_contract != 0 else 0
        return self.quantity


class FixedDollar(PositionSizer):
    """Size positions to a fixed dollar amount."""

    def __init__(self, amount: float = 10_000.0) -> None:
        self.amount = amount

    def size(self, cost_per_contract: float, available_capital: float,
             total_capital: float) -> int:
        if cost_per_contract == 0:
            return 0
        target = min(self.amount, available_capital)
        return int(target // abs(cost_per_contract))


class PercentOfPortfolio(PositionSizer):
    """Size positions as a percentage of total portfolio value."""

    def __init__(self, pct: float = 0.01) -> None:
        assert 0.0 < pct <= 1.0
        self.pct = pct

    def size(self, cost_per_contract: float, available_capital: float,
             total_capital: float) -> int:
        if cost_per_contract == 0:
            return 0
        target = min(self.pct * total_capital, available_capital)
        return int(target // abs(cost_per_contract))
