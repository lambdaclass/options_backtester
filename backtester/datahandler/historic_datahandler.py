import pandas as pd
from .datahandler import DataHandler
from ..event import MarketEvent


class HistoricDataHandler(DataHandler):
    """Handler for Historical Option Data"""

    def __init__(self, data_path, events):
        self._data = pd.read_csv(
            data_path, parse_dates=["quotedate",
                                    "expiration"]).sort_values(by="date")

        columns = {"quotedate": "date", "optionroot": "symbol"}
        self._data.rename(columns=columns, inplace=True)
        self._data_index = 0
        self.events = events
        self.continue_backtest = True

    def get_latest_bars(self, symbol, N=1):
        """Returns the latest `N` bars for `symbol` if there are at least N
        rows, otherwise returns the all data.
        Returns empty dataframe if `symbol` is not in self._data.
        """
        return self._data[(self._data["symbol"] == symbol)
                          & (self._data["date"] <= self.current_date)][-N:]

    def update_bars(self):
        """Add new data bar to self.data"""
        if self._data_index < len(self._data):
            self.current_date = self._data["date"][self._data_index]
            self.events.put(MarketEvent())
            self._data_index += 1
        else:
            self.continue_backtest = False
