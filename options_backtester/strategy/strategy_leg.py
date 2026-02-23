"""Strategy leg — re-exports the original StrategyLeg for now.

The new StrategyLeg is API-compatible with the original and adds support
for the new execution components (signal_selector, fill_model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from options_backtester.core.types import Direction, OptionType
from options_backtester.execution.signal_selector import SignalSelector, FirstMatch
from options_backtester.execution.fill_model import FillModel, MarketAtBidAsk

if TYPE_CHECKING:
    from backtester.datahandler.schema import Filter, Schema


class StrategyLeg:
    """A single option leg in a strategy.

    API-compatible with backtester.strategy.strategy_leg.StrategyLeg, adding
    optional signal_selector and fill_model.
    """

    def __init__(
        self,
        name: str,
        schema: "Schema",
        option_type: OptionType = OptionType.CALL,
        direction: Direction = Direction.BUY,
        signal_selector: SignalSelector | None = None,
        fill_model: FillModel | None = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self.type = option_type
        self.direction = direction
        self.signal_selector = signal_selector  # None = use engine-level default
        self.fill_model = fill_model  # None = use engine-level default

        self.entry_sort: tuple[str, bool] | None = None
        self._entry_filter: "Filter" = self._base_entry_filter()
        self._exit_filter: "Filter" = self._base_exit_filter()

    @property
    def entry_filter(self) -> "Filter":
        return self._entry_filter

    @entry_filter.setter
    def entry_filter(self, flt: "Filter") -> None:
        self._entry_filter = self._base_entry_filter() & flt

    @property
    def exit_filter(self) -> "Filter":
        return self._exit_filter

    @exit_filter.setter
    def exit_filter(self, flt: "Filter") -> None:
        self._exit_filter = self._base_exit_filter() & flt

    def _base_entry_filter(self) -> "Filter":
        if self.direction == Direction.BUY:
            return (self.schema.type == self.type.value) & (self.schema.ask > 0)
        return (self.schema.type == self.type.value) & (self.schema.bid > 0)

    def _base_exit_filter(self) -> "Filter":
        return self.schema.type == self.type.value

    def __repr__(self) -> str:
        return (
            f"StrategyLeg(name={self.name}, type={self.type}, "
            f"direction={self.direction}, entry_filter={self._entry_filter}, "
            f"exit_filter={self._exit_filter})"
        )
