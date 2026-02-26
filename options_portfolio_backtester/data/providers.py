"""Data providers — ABCs, CSV implementations, and legacy data loaders."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Union

import pandas as pd

from .schema import Schema, Filter


# ---------------------------------------------------------------------------
# Legacy data loaders (moved from backtester.datahandler)
# ---------------------------------------------------------------------------

class TiingoData:
    """Tiingo (stocks & indeces) Data container class."""
    def __init__(self, file: str, schema: Schema | None = None, **params: Any) -> None:
        if schema is None:
            self.schema = TiingoData.default_schema()

        file_extension = os.path.splitext(file)[1]

        if file_extension == '.h5':
            self._data: pd.DataFrame = pd.read_hdf(file, **params)
        elif file_extension == '.csv':
            params['parse_dates'] = [self.schema.date.mapping]
            self._data = pd.read_csv(file, **params)

        columns = self._data.columns
        assert all((col in columns for _key, col in self.schema))

        date_col = self.schema['date']

        self.start_date: pd.Timestamp = self._data[date_col].min()
        self.end_date: pd.Timestamp = self._data[date_col].max()

    def apply_filter(self, f: Filter) -> pd.DataFrame:
        """Apply Filter `f` to the data. Returns a `pd.DataFrame` with the filtered rows."""
        return self._data.query(f.query)

    def iter_dates(self) -> pd.core.groupby.DataFrameGroupBy:
        """Returns `pd.DataFrameGroupBy` that groups stocks by date"""
        return self._data.groupby(self.schema['date'])

    def iter_months(self) -> pd.core.groupby.DataFrameGroupBy:
        """Returns `pd.DataFrameGroupBy` that groups stocks by month"""
        date_col = self.schema['date']
        iterator = self._data.groupby(pd.Grouper(
            key=date_col,
            freq="MS")).apply(lambda g: g[g[date_col] == g[date_col].min()]).reset_index(drop=True).groupby(date_col)
        return iterator

    def __getattr__(self, attr: str) -> Any:
        """Pass method invocation to `self._data`"""

        method = getattr(self._data, attr)
        if hasattr(method, '__call__'):

            def df_method(*args: Any, **kwargs: Any) -> Any:
                return method(*args, **kwargs)

            return df_method
        else:
            return method

    def __getitem__(self, item: Union[str, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(item, pd.Series):
            return self._data[item]
        else:
            key = self.schema[item]
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        if key not in self.schema:
            self.schema.update({key: key})

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return self._data.__repr__()

    @staticmethod
    def default_schema() -> Schema:
        """Returns default schema for Tiingo Data"""
        return Schema.stocks()

    def sma(self, periods: int) -> None:
        sma = self._data.groupby('symbol', as_index=False).rolling(periods)['adjClose'].mean()
        sma = sma.fillna(0)
        sma.index = [index[1] for index in sma.index]
        self._data['sma'] = sma
        self.schema.update({'sma': 'sma'})


class HistoricalOptionsData:
    """Historical Options Data container class."""
    def __init__(self, file: str, schema: Schema | None = None, **params: Any) -> None:
        if schema is None:
            self.schema = HistoricalOptionsData.default_schema()

        file_extension = os.path.splitext(file)[1]

        if file_extension == '.h5':
            self._data: pd.DataFrame = pd.read_hdf(file, **params)
        elif file_extension == '.csv':
            params['parse_dates'] = [self.schema.expiration.mapping, self.schema.date.mapping]
            self._data = pd.read_csv(file, **params)

        columns = self._data.columns
        assert all((col in columns for _key, col in self.schema))

        date_col = self.schema['date']
        expiration_col = self.schema['expiration']

        self._data['dte'] = (self._data[expiration_col] - self._data[date_col]).dt.days
        self.schema.update({'dte': 'dte'})

        self.start_date: pd.Timestamp = self._data[date_col].min()
        self.end_date: pd.Timestamp = self._data[date_col].max()

    def apply_filter(self, f: Filter) -> pd.DataFrame:
        """Apply Filter `f` to the data. Returns a `pd.DataFrame` with the filtered rows."""
        return self._data.query(f.query)

    def iter_dates(self) -> pd.core.groupby.DataFrameGroupBy:
        """Returns `pd.DataFrameGroupBy` that groups contracts by date"""
        return self._data.groupby(self.schema['date'])

    def iter_months(self) -> pd.core.groupby.DataFrameGroupBy:
        """Returns `pd.DataFrameGroupBy` that groups contracts by month"""
        date_col = self.schema['date']
        iterator = self._data.groupby(pd.Grouper(
            key=date_col,
            freq="MS")).apply(lambda g: g[g[date_col] == g[date_col].min()]).reset_index(drop=True).groupby(date_col)
        return iterator

    def __getattr__(self, attr: str) -> Any:
        """Pass method invocation to `self._data`"""

        method = getattr(self._data, attr)
        if hasattr(method, '__call__'):

            def df_method(*args: Any, **kwargs: Any) -> Any:
                return method(*args, **kwargs)

            return df_method
        else:
            return method

    def __getitem__(self, item: Union[str, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(item, pd.Series):
            return self._data[item]
        else:
            key = self.schema[item]
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        if key not in self.schema:
            self.schema.update({key: key})

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return self._data.__repr__()

    @staticmethod
    def default_schema() -> Schema:
        """Returns default schema for Historical Options Data"""
        schema = Schema.options()
        schema.update({
            'contract': 'optionroot',
            'date': 'quotedate',
            'last': 'last',
            'open_interest': 'openinterest',
            'impliedvol': 'impliedvol',
            'delta': 'delta',
            'gamma': 'gamma',
            'theta': 'theta',
            'vega': 'vega'
        })
        return schema


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
