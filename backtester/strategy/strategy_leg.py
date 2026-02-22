from __future__ import annotations

from typing import TYPE_CHECKING

from backtester.enums import Type, Direction

if TYPE_CHECKING:
    from backtester.datahandler.schema import Filter, Schema


class StrategyLeg:
    """Strategy Leg data class"""
    def __init__(self, name: str, schema: Schema, option_type: Type = Type.CALL, direction: Direction = Direction.BUY) -> None:
        self.name = name
        self.schema = schema
        self.type = option_type
        self.direction = direction

        self.entry_sort: tuple[str, bool] | None = None

        self._entry_filter: Filter = self._base_entry_filter()
        self._exit_filter: Filter = self._base_exit_filter()

    @property
    def entry_filter(self) -> Filter:
        """Returns the entry filter"""
        return self._entry_filter

    @entry_filter.setter
    def entry_filter(self, flt: Filter) -> None:
        """Sets the entry filter"""
        self._entry_filter = self._base_entry_filter() & flt

    @property
    def exit_filter(self) -> Filter:
        """Returns the exit filter"""
        return self._exit_filter

    @exit_filter.setter
    def exit_filter(self, flt: Filter) -> None:
        """Sets the exit filter"""
        self._exit_filter = self._base_exit_filter() & flt

    def _base_entry_filter(self) -> Filter:
        if self.direction == Direction.BUY:
            return (self.schema.type == self.type.value) & (self.schema.ask > 0)
        else:
            return (self.schema.type == self.type.value) & (self.schema.bid > 0)

    def _base_exit_filter(self) -> Filter:
        return self.schema.type == self.type.value

    def __repr__(self) -> str:
        return "StrategyLeg(name={}, type={}, direction={}, entry_filter={}, exit_filter={})".format(
            self.name, self.type, self.direction, self._entry_filter, self._exit_filter)
