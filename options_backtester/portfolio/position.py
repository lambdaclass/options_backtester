"""Option position and position leg — replaces MultiIndex inventory rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from options_backtester.core.types import (
    Direction, OptionType, Order, Greeks, get_order, Signal,
)


@dataclass
class PositionLeg:
    """A single leg within an option position."""
    name: str
    contract_id: str
    underlying: str
    expiration: Any  # pd.Timestamp
    option_type: OptionType
    strike: float
    entry_price: float
    direction: Direction
    order: Order

    @property
    def exit_order(self) -> Order:
        return ~self.order

    def current_value(self, current_price: float, quantity: int,
                      shares_per_contract: int) -> float:
        """Mark-to-market value of this leg.

        For a BUY leg, value = current_price * qty * spc (we own it).
        For a SELL leg, value = -current_price * qty * spc (we owe it).
        """
        sign = -1 if self.direction == Direction.SELL else 1
        return sign * current_price * quantity * shares_per_contract


@dataclass
class OptionPosition:
    """A multi-leg option position.

    Replaces one row in the old MultiIndex _options_inventory DataFrame.
    """
    position_id: int
    legs: dict[str, PositionLeg] = field(default_factory=dict)
    quantity: int = 0
    entry_cost: float = 0.0  # total cost at entry (negative for debit)
    entry_date: Any = None  # pd.Timestamp

    def add_leg(self, leg: PositionLeg) -> None:
        self.legs[leg.name] = leg

    def current_value(self, current_prices: dict[str, float],
                      shares_per_contract: int) -> float:
        """Total MTM value of this position across all legs.

        Args:
            current_prices: {leg_name: exit_price} for each leg.
            shares_per_contract: Contract multiplier.
        """
        total = 0.0
        for leg_name, leg in self.legs.items():
            price = current_prices.get(leg_name, 0.0)
            total += leg.current_value(price, self.quantity, shares_per_contract)
        return total

    def greeks(self, leg_greeks: dict[str, Greeks]) -> Greeks:
        """Aggregate Greeks across all legs, scaled by quantity.

        Args:
            leg_greeks: {leg_name: Greeks} for each leg.
        """
        total = Greeks()
        for leg_name, leg in self.legs.items():
            g = leg_greeks.get(leg_name, Greeks())
            sign = 1 if leg.direction == Direction.BUY else -1
            total = total + g * (sign * self.quantity)
        return total
