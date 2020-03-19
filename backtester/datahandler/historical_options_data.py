import pandas as pd
import os
from .schema import Schema


class HistoricalOptionsData:
    """Historical Options Data container class."""
    def __init__(self, file, schema=None, **params):
        if schema is None:
            self.schema = HistoricalOptionsData.default_schema()

        file_extension = os.path.splitext(file)[1]

        if file_extension == '.h5':
            self._data = pd.read_hdf(file, **params)
        elif file_extension == '.csv':
            params['parse_dates'] = [self.schema.expiration.mapping, self.schema.date.mapping]
            self._data = pd.read_csv(file, **params)

        columns = self._data.columns
        assert all((col in columns for _key, col in self.schema))

        date_col = self.schema['date']
        expiration_col = self.schema['expiration']

        self._data['dte'] = (self._data[expiration_col] - self._data[date_col]).dt.days
        self.schema.update({'dte': 'dte'})

        self.start_date = self._data[date_col].min()
        self.end_date = self._data[date_col].max()

    def apply_filter(self, f):
        """Apply Filter `f` to the data. Returns a `pd.DataFrame` with the filtered rows."""
        return self._data.query(f.query)

    def iter_dates(self):
        """Returns `pd.DataFrameGroupBy` that groups contracts by date"""
        return self._data.groupby(self.schema['date'])

    def iter_months(self):
        """Returns `pd.DataFrameGroupBy` that groups contracts by month"""
        date_col = self.schema['date']
        iterator = self._data.groupby(pd.Grouper(
            key=date_col,
            freq="MS")).apply(lambda g: g[g[date_col] == g[date_col].min()]).reset_index(drop=True).groupby(date_col)
        return iterator

    def __getattr__(self, attr):
        """Pass method invocation to `self._data`"""

        method = getattr(self._data, attr)
        if hasattr(method, '__call__'):

            def df_method(*args, **kwargs):
                return method(*args, **kwargs)

            return df_method
        else:
            return method

    def __getitem__(self, item):
        if isinstance(item, pd.Series):
            return self._data[item]
        else:
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
