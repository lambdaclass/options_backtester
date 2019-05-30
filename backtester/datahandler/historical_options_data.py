import pandas as pd
from .schema import Schema


class HistoricalOptionsData:
    """Historical Options Data container class."""

    def __init__(self, file, schema=None, **params):
        if schema:
            assert isinstance(schema, Schema)
        else:
            schema = Schema.canonical()
            schema.update({"contract": "optionroot", "date": "quotedate"})
        self.schema = schema

        self._data = pd.read_hdf(file, **params)
        columns = self._data.columns
        assert all((col in columns for col in schema))

        self._data["dte"] = (self._data["expiration"] -
                             self._data["quotedate"]).dt.days

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value):
        self._data[item] = value

    def __repr__(self):
        return self._data.__repr__()
