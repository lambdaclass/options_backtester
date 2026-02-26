"""Filter DSL — Schema, Field, and Filter for building query expressions."""

from __future__ import annotations

from typing import Any, Iterator, Union


class Schema:
    """Data schema class.
    Used provide uniform access to fields in the data set.
    """

    stock_columns = [
        "symbol", "date", "open", "close", "high", "low", "volume", "adjClose", "adjHigh", "adjLow", "adjOpen",
        "adjVolume", "divCash", "splitFactor"
    ]

    option_columns = [
        "underlying", "underlying_last", "date", "contract", "type", "expiration", "strike", "bid", "ask", "volume",
        "open_interest"
    ]

    @staticmethod
    def stocks() -> Schema:
        """Builder method that returns a `Schema` with default mappings for stocks"""
        mappings = {key: key for key in Schema.stock_columns}
        return Schema(mappings)

    @staticmethod
    def options() -> Schema:
        """Builder method that returns a `Schema` with default mappings for options"""
        mappings = {key: key for key in Schema.option_columns}
        return Schema(mappings)

    def __init__(self, mappings: dict[str, str]) -> None:
        assert all((key in mappings for key in Schema.stock_columns)) or all(
            (key in mappings for key in Schema.option_columns))

        self._mappings: dict[str, str] = mappings

    def update(self, mappings: dict[str, str]) -> Schema:
        """Update schema according to given `mappings`"""
        self._mappings.update(mappings)
        return self

    def __contains__(self, key: str) -> bool:
        """Returns True if key is in schema"""
        return key in self._mappings.keys()

    def __getattr__(self, key: str) -> Field:
        """Returns Field object used to build Filters"""
        return Field(key, self._mappings[key])

    def __setitem__(self, key: str, value: str) -> None:
        self._mappings[key] = value

    def __getitem__(self, key: str) -> str:
        """Returns mapping of given `key`"""
        return self._mappings[key]

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._mappings.items())

    def __repr__(self) -> str:
        return "Schema({})".format([Field(k, m) for k, m in self._mappings.items()])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return self._mappings == other._mappings


class Field:
    """Encapsulates data fields to build filters used by strategies"""

    __slots__ = ("name", "mapping")

    def __init__(self, name: str, mapping: str) -> None:
        self.name = name
        self.mapping = mapping

    def _create_filter(self, op: str, other: Union[Field, Any]) -> Filter:
        if isinstance(other, Field):

            query = Field._format_query(self.mapping, op, other.mapping)
        else:
            query = Field._format_query(self.mapping, op, other)
        return Filter(query)

    def _combine_fields(self, op: str, other: Union[Field, int, float], invert: bool = False) -> Field:
        if isinstance(other, Field):
            name = Field._format_query(self.name, op, other.name, invert)
            mapping = Field._format_query(self.mapping, op, other.mapping, invert)
        elif isinstance(other, (int, float)):
            name = Field._format_query(self.name, op, other, invert)
            mapping = Field._format_query(self.mapping, op, other, invert)
        else:
            raise TypeError

        return Field(name, mapping)

    @staticmethod
    def _format_query(left: Any, op: str, right: Any, invert: bool = False) -> str:
        if invert:
            left, right = right, left
        query = "{left} {op} {right}".format(left=left, op=op, right=right)
        return query

    def __add__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("+", value)

    def __radd__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("+", value, invert=True)

    def __sub__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("-", value)

    def __rsub__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("-", value, invert=True)

    def __mul__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("*", value)

    def __rmul__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("*", value, invert=True)

    def __truediv__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("/", value)

    def __rtruediv__(self, value: Union[Field, int, float]) -> Field:
        return self._combine_fields("/", value, invert=True)

    def __lt__(self, value: Union[Field, Any]) -> Filter:
        return self._create_filter("<", value)

    def __le__(self, value: Union[Field, Any]) -> Filter:
        return self._create_filter("<=", value)

    def __gt__(self, value: Union[Field, Any]) -> Filter:
        return self._create_filter(">", value)

    def __ge__(self, value: Union[Field, Any]) -> Filter:
        return self._create_filter(">=", value)

    def __eq__(self, value: Union[Field, Any]) -> Filter:  # type: ignore[override]
        if isinstance(value, str):
            value = "'{}'".format(value)
        return self._create_filter("==", value)

    def __ne__(self, value: Union[Field, Any]) -> Filter:  # type: ignore[override]
        return self._create_filter("!=", value)

    def __repr__(self) -> str:
        return "Field(name='{}', mapping='{}')".format(self.name, self.mapping)


class Filter:
    """This class determines entry/exit conditions for strategies"""

    __slots__ = ("query")

    def __init__(self, query: str) -> None:
        self.query = query

    def __and__(self, other: Filter) -> Filter:
        """Returns logical *and* between `self` and `other`"""
        assert isinstance(other, Filter)
        new_query = "({}) & ({})".format(self.query, other.query)
        return Filter(query=new_query)

    def __or__(self, other: Filter) -> Filter:
        """Returns logical *or* between `self` and `other`"""
        assert isinstance(other, Filter)
        new_query = "(({}) | ({}))".format(self.query, other.query)
        return Filter(query=new_query)

    def __invert__(self) -> Filter:
        """Negates filter"""
        return Filter("!({})".format(self.query))

    def __call__(self, data: 'pd.DataFrame') -> 'pd.Series':
        """Returns dataframe of filtered data"""
        return data.eval(self.query)

    def __repr__(self) -> str:
        return "Filter(query='{}')".format(self.query)


__all__ = ["Schema", "Field", "Filter"]
