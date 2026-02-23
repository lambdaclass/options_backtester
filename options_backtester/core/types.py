"""Core domain types for options backtesting.

Direction is decoupled from column names — use Direction.price_column instead of
Direction.value to get the DataFrame column for pricing.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

    def __invert__(self) -> OptionType:
        return OptionType.PUT if self == OptionType.CALL else OptionType.CALL


class Direction(Enum):
    """Trade direction. price_column gives the DataFrame column name."""
    BUY = "buy"
    SELL = "sell"

    @property
    def price_column(self) -> str:
        """Column name used for trade execution pricing."""
        return "ask" if self == Direction.BUY else "bid"

    def __invert__(self) -> Direction:
        return Direction.SELL if self == Direction.BUY else Direction.BUY


class Signal(Enum):
    ENTRY = "entry"
    EXIT = "exit"


class Order(Enum):
    BTO = "BTO"  # Buy to Open
    BTC = "BTC"  # Buy to Close
    STO = "STO"  # Sell to Open
    STC = "STC"  # Sell to Close

    def __invert__(self) -> Order:
        _inv = {Order.BTO: Order.STC, Order.STC: Order.BTO,
                Order.STO: Order.BTC, Order.BTC: Order.STO}
        return _inv[self]


def get_order(direction: Direction, signal: Signal) -> Order:
    """Map (direction, signal) to the appropriate Order type."""
    if direction == Direction.BUY:
        return Order.BTO if signal == Signal.ENTRY else Order.STC
    return Order.STO if signal == Signal.ENTRY else Order.BTC


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Greeks:
    """Option Greeks for a single contract or aggregated position.

    Supports addition (aggregation) and scalar multiplication (scaling by qty).
    """
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    def __add__(self, other: Greeks) -> Greeks:
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
        )

    def __mul__(self, scalar: float) -> Greeks:
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
        )

    def __rmul__(self, scalar: float) -> Greeks:
        return self.__mul__(scalar)

    def __neg__(self) -> Greeks:
        return self * -1.0

    @property
    def as_dict(self) -> dict[str, float]:
        return {"delta": self.delta, "gamma": self.gamma,
                "theta": self.theta, "vega": self.vega}


@dataclass(frozen=True, slots=True)
class Fill:
    """A single execution fill.

    Captures price, quantity, commission, slippage, and computes notional.
    """
    price: float
    quantity: int
    direction: Direction
    shares_per_contract: int = 100
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def direction_sign(self) -> int:
        return -1 if self.direction == Direction.BUY else 1

    @property
    def notional(self) -> float:
        """Net cash impact: direction_sign * (price * qty * spc) - commission - slippage."""
        raw = self.direction_sign * self.price * self.quantity * self.shares_per_contract
        return raw - self.commission - self.slippage


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Identifies a specific option contract."""
    contract_id: str
    underlying: str
    expiration: Any  # pd.Timestamp or str
    option_type: OptionType
    strike: float


# Re-use namedtuple for backward compatibility
StockAllocation = namedtuple("StockAllocation", "symbol percentage")

# Backward-compatible alias
Stock = StockAllocation
