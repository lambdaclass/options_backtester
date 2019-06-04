import pandas as pd
from .schema import Schema


class HistoricalOptionsData:
    """Historical Options Data container class."""

    def __init__(self, file, schema=None, **params):
        if schema:
            assert isinstance(schema, Schema)
        else:
            self.schema = HistoricalOptionsData.default_schema()

        self._data = pd.read_hdf(file, **params)
        columns = self._data.columns
        assert all((col in columns for _key, col in self.schema))

        self._data["dte"] = (self._data["expiration"] -
                             self._data["quotedate"]).dt.days
        self.schema.update({"dte": "dte"})

    def apply_filter(self, f):
        """Apply Filter `f` to the data. Returns a `pd.DataFrame` with the filtered rows."""
        return self._data.query(f.query)

    def iter_dates(self):
        """Returns `pd.DataFrameGroupBy` that groups contracts by date"""
        return self._data.groupby(self.schema["date"])

    def __getattr__(self, attr):
        """Pass method invocation to `self._data`"""

        method = getattr(self._data, attr)
        if hasattr(method, "__call__"):

            def df_method(*args, **kwargs):
                return method(*args, **kwargs)

            return df_method
        else:
            return method

    def __getitem__(self, item):
        key = self.schema[item]
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value
        if key not in self.schema:
            self.schema.update({key: key})

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._data.__repr__()

    def default_schema():
        """Returns default schema for Historical Options Data"""
        schema = Schema.canonical()
        schema.update({
            "contract": "optionroot",
            "date": "quotedate",
            "last": "last",
            "open_interest": "openinterest",
            "impliedvol": "impliedvol",
            "delta": "delta",
            "gamma": "gamma",
            "theta": "theta",
            "vega": "vega"
        })
        return schema
