"""Transaction cost models for options and stocks."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TransactionCostModel(ABC):
    """Base class for all transaction cost models."""

    @abstractmethod
    def option_cost(self, price: float, quantity: int, shares_per_contract: int) -> float:
        """Return total commission for an options trade."""
        ...

    @abstractmethod
    def stock_cost(self, price: float, quantity: float) -> float:
        """Return total commission for a stock trade."""
        ...


class NoCosts(TransactionCostModel):
    """Zero transaction costs — matches original behavior."""

    def option_cost(self, price: float, quantity: int, shares_per_contract: int) -> float:
        return 0.0

    def stock_cost(self, price: float, quantity: float) -> float:
        return 0.0


class PerContractCommission(TransactionCostModel):
    """Fixed per-contract commission (e.g., $0.65/contract for IBKR)."""

    def __init__(self, rate: float = 0.65, stock_rate: float = 0.005) -> None:
        self.rate = rate
        self.stock_rate = stock_rate  # per-share

    def option_cost(self, price: float, quantity: int, shares_per_contract: int) -> float:
        return self.rate * abs(quantity)

    def stock_cost(self, price: float, quantity: float) -> float:
        return self.stock_rate * abs(quantity)


class TieredCommission(TransactionCostModel):
    """Tiered commission schedule with volume discounts.

    Tiers are (max_contracts, rate) pairs sorted by max_contracts ascending.
    Contracts beyond the last tier use the last tier's rate.
    """

    def __init__(self, tiers: list[tuple[int, float]] | None = None,
                 stock_rate: float = 0.005) -> None:
        # Default: IBKR-style tiers
        self.tiers = tiers or [
            (10_000, 0.65),
            (50_000, 0.50),
            (100_000, 0.25),
        ]
        self.stock_rate = stock_rate

    def option_cost(self, price: float, quantity: int, shares_per_contract: int) -> float:
        qty = abs(quantity)
        total = 0.0
        remaining = qty
        prev_bound = 0
        for max_qty, rate in self.tiers:
            tier_qty = min(remaining, max_qty - prev_bound)
            if tier_qty <= 0:
                prev_bound = max_qty
                continue
            total += tier_qty * rate
            remaining -= tier_qty
            prev_bound = max_qty
            if remaining <= 0:
                break
        if remaining > 0:
            total += remaining * self.tiers[-1][1]
        return total

    def stock_cost(self, price: float, quantity: float) -> float:
        return self.stock_rate * abs(quantity)


class SpreadSlippage(TransactionCostModel):
    """Model slippage as a fraction of the bid-ask spread.

    Example: SpreadSlippage(pct=0.5) means you pay half the spread on top
    of the execution price.
    """

    def __init__(self, pct: float = 0.5) -> None:
        assert 0.0 <= pct <= 1.0
        self.pct = pct

    def option_cost(self, price: float, quantity: int, shares_per_contract: int) -> float:
        # Slippage is modeled separately via fill_model; this returns 0
        # so it can be composed with a commission model.
        return 0.0

    def stock_cost(self, price: float, quantity: float) -> float:
        return 0.0

    def slippage(self, bid: float, ask: float, quantity: int, shares_per_contract: int) -> float:
        """Compute dollar slippage from the spread."""
        spread = abs(ask - bid)
        return self.pct * spread * abs(quantity) * shares_per_contract
