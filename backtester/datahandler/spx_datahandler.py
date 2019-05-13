import pandas as pd
from .datahandler import DataHandler
from ..event import MarketEvent


class SPXDataHandler(DataHandler):
    """Handler for SPX test data"""

    def __init__(self, data_path, events):
        self._data = pd.read_csv(
            data_path, parse_dates=["date"]).sort_values(by="date")

        self._data.rename(columns={"price": "ask"}, inplace=True)
        self._data["bid"] = self._data["ask"]
        self._data_index = 0
        self.events = events
        self.continue_backtest = True

    def get_latest_bars(self, symbol, N=1):
        """Returns the latest `N` bars for `symbol` if there are at least N
        rows, otherwise returns the all data.
        Returns empty dataframe if `symbol` is not in self.data.
        """
        return self._data[self._data["date"] <= self.current_date][-N:]

    def update_bars(self):
        """Add new data bar to self.data"""
        if self._data_index < len(self._data):
            self.current_date = self._data["date"][self._data_index]
            self.events.put(MarketEvent())
            self._data_index += 1
        else:
            self.continue_backtest = False
