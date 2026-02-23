"""Portfolio — clean replacement for MultiIndex DataFrames.

Uses plain dicts and dataclasses instead of MultiIndex DataFrames for
inventory tracking. Simpler, extensible, debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from options_backtester.core.types import Greeks
from options_backtester.portfolio.position import OptionPosition
from options_backtester.portfolio.greeks import aggregate_greeks


@dataclass
class StockHolding:
    """A stock position in the portfolio."""
    symbol: str
    quantity: float
    cost_basis: float  # average price paid


class Portfolio:
    """Portfolio state — cash, option positions, stock holdings.

    Replaces the old MultiIndex _options_inventory and _stocks_inventory
    DataFrames with typed, inspectable data structures.
    """

    def __init__(self, initial_cash: float = 0.0) -> None:
        self.cash: float = initial_cash
        self.option_positions: dict[int, OptionPosition] = {}
        self.stock_holdings: dict[str, StockHolding] = {}
        self._next_position_id: int = 0

    def next_position_id(self) -> int:
        pid = self._next_position_id
        self._next_position_id += 1
        return pid

    # -- Option positions --

    def add_option_position(self, pos: OptionPosition) -> None:
        self.option_positions[pos.position_id] = pos

    def remove_option_position(self, position_id: int) -> OptionPosition | None:
        return self.option_positions.pop(position_id, None)

    def options_value(self, current_prices: dict[int, dict[str, float]],
                      shares_per_contract: int) -> float:
        """Total mark-to-market value of all option positions.

        Args:
            current_prices: {position_id: {leg_name: exit_price}}.
            shares_per_contract: Contract multiplier.
        """
        total = 0.0
        for pid, pos in self.option_positions.items():
            prices = current_prices.get(pid, {})
            total += pos.current_value(prices, shares_per_contract)
        return total

    # -- Stock holdings --

    def set_stock_holding(self, symbol: str, quantity: float,
                          price: float) -> None:
        self.stock_holdings[symbol] = StockHolding(
            symbol=symbol, quantity=quantity, cost_basis=price,
        )

    def clear_stock_holdings(self) -> None:
        self.stock_holdings.clear()

    def stocks_value(self, current_prices: dict[str, float]) -> float:
        """Total value of stock holdings at current prices."""
        total = 0.0
        for symbol, holding in self.stock_holdings.items():
            price = current_prices.get(symbol, holding.cost_basis)
            total += holding.quantity * price
        return total

    # -- Portfolio totals --

    def total_value(self, stock_prices: dict[str, float],
                    option_prices: dict[int, dict[str, float]],
                    shares_per_contract: int) -> float:
        """Total portfolio value: cash + stocks + options."""
        return (self.cash
                + self.stocks_value(stock_prices)
                + self.options_value(option_prices, shares_per_contract))

    def portfolio_greeks(self,
                         leg_greeks_by_position: dict[int, dict[str, Greeks]]) -> Greeks:
        """Aggregate Greeks across all option positions."""
        return aggregate_greeks(self.option_positions, leg_greeks_by_position)
