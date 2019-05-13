import pandas as pd
from .datahandler import DataHandler
from ..event import MarketEvent


class BalancedDataHandler(DataHandler):
    """Handler for balanced data set"""

    def __init__(self, data_path, events):
        data = pd.read_csv(data_path, parse_dates=["date"])

        # We will assume bid and ask prices = close
        data["bid"] = data["close"]
        data["ask"] = data["close"]

        self._data_generator = self._get_data_generator(data)
        self.events = events
        self.continue_backtest = True

    def get_latest_bars(self, symbol, N=1):
        """Returns the latest `N` bars for `symbol` if there are at least N
        rows, otherwise returns the all data.
        Returns empty dataframe if `symbol` is not in self.data.
        """
        return self._current_bar[self._current_bar["symbol"] == symbol].iloc[0]

    def update_bars(self):
        """Add new data bar to self.data"""
        try:
            self.current_date, self._current_bar = next(self._data_generator)
            self.events.put(MarketEvent())
        except StopIteration:
            self.continue_backtest = False

    def _get_data_generator(self, data):
        """Returns generator that yields daily data bars"""
        grouped = data.groupby("date")
        for date, bars in grouped:
            yield date, bars
