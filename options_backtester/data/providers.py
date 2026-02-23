"""Data providers — ABCs and CSV implementations wrapping existing loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from backtester.datahandler.schema import Schema, Filter
from backtester.datahandler.historical_options_data import HistoricalOptionsData
from backtester.datahandler.tiingo_data import TiingoData


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """Base interface for all data providers."""

    @property
    @abstractmethod
    def schema(self) -> Schema:
        ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        ...

    @property
    @abstractmethod
    def start_date(self) -> pd.Timestamp:
        ...

    @property
    @abstractmethod
    def end_date(self) -> pd.Timestamp:
        ...

    @abstractmethod
    def apply_filter(self, f: Filter) -> pd.DataFrame:
        ...

    @abstractmethod
    def iter_dates(self) -> Any:
        ...

    @abstractmethod
    def iter_months(self) -> Any:
        ...


class OptionsDataProvider(DataProvider):
    """Options-specific data provider interface."""
    pass


class StocksDataProvider(DataProvider):
    """Stocks-specific data provider interface."""

    @abstractmethod
    def sma(self, periods: int) -> None:
        ...


# ---------------------------------------------------------------------------
# CSV implementations (wrap existing loaders)
# ---------------------------------------------------------------------------

class CsvOptionsProvider(OptionsDataProvider):
    """Load options data from CSV files using the existing HistoricalOptionsData loader."""

    def __init__(self, file: str, schema: Schema | None = None, **params: Any) -> None:
        self._loader = HistoricalOptionsData(file, schema=schema, **params)

    @property
    def schema(self) -> Schema:
        return self._loader.schema

    @property
    def data(self) -> pd.DataFrame:
        return self._loader._data

    @property
    def start_date(self) -> pd.Timestamp:
        return self._loader.start_date

    @property
    def end_date(self) -> pd.Timestamp:
        return self._loader.end_date

    def apply_filter(self, f: Filter) -> pd.DataFrame:
        return self._loader.apply_filter(f)

    def iter_dates(self) -> Any:
        return self._loader.iter_dates()

    def iter_months(self) -> Any:
        return self._loader.iter_months()

    # Pass-through to underlying loader for backward compat
    def __getitem__(self, item: Any) -> Any:
        return self._loader[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self._loader[key] = value

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def _data(self) -> pd.DataFrame:
        """Backward-compat access used by engine internals."""
        return self._loader._data


class CsvStocksProvider(StocksDataProvider):
    """Load stock data from CSV files using the existing TiingoData loader."""

    def __init__(self, file: str, schema: Schema | None = None, **params: Any) -> None:
        self._loader = TiingoData(file, schema=schema, **params)

    @property
    def schema(self) -> Schema:
        return self._loader.schema

    @property
    def data(self) -> pd.DataFrame:
        return self._loader._data

    @property
    def start_date(self) -> pd.Timestamp:
        return self._loader.start_date

    @property
    def end_date(self) -> pd.Timestamp:
        return self._loader.end_date

    def apply_filter(self, f: Filter) -> pd.DataFrame:
        return self._loader.apply_filter(f)

    def iter_dates(self) -> Any:
        return self._loader.iter_dates()

    def iter_months(self) -> Any:
        return self._loader.iter_months()

    def sma(self, periods: int) -> None:
        self._loader.sma(periods)

    def __getitem__(self, item: Any) -> Any:
        return self._loader[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self._loader[key] = value

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def _data(self) -> pd.DataFrame:
        """Backward-compat access used by engine internals."""
        return self._loader._data
