import pandas as pd

from .strategy import Strategy
from .datahandler import HistoricalOptionsData


class Backtest:
    """Processes signals from the Strategy object"""

    def __init__(self, qty=1, capital=1_000_000, shares_per_contract=100):
        self.capital = capital
        self.shares_per_contract = shares_per_contract
        self.qty = qty
        self._strategy = None
        self._data = None
        self.inventory = pd.DataFrame()
        self.stop_if_broke = True

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strat):
        assert isinstance(strat, Strategy)
        self._strategy = strat

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert isinstance(data, HistoricalOptionsData)
        self._data = data

    def run(self):
        """Runs the backtest and returns a `pd.DataFrame` of the orders executed (`self.trade_log`)"""
        assert self._data is not None
        assert self._strategy is not None
        assert self._data.schema == self._strategy.schema

        self.trade_log = pd.DataFrame()

        for date, options in self._data.iter_dates():
            entry_signals = self._strategy.filter_entries(options)
            exit_signals = self._strategy.filter_exits(options, self.inventory)

            self._execute_exit(date, exit_signals)
            self._execute_entry(date, entry_signals)

        return self.trade_log

    def _execute_entry(self, date, entry_signals):
        """Executes entry orders and updates `self.inventory` and `self.trade_log`"""
        entry, total_price = self._process_entry_signals(entry_signals)

        if (not self.stop_if_broke) or (self.capital >= total_price):
            self.inventory = self.inventory.append(entry, ignore_index=True)
            self.trade_log = self.trade_log.append(entry, ignore_index=True)
            self.capital -= total_price

    def _execute_exit(self, date, exit_signals):
        """Executes exits and updates `self.inventory` and `self.trade_log`"""
        if exit_signals is None:
            return
        exits, exits_mask, total_costs = exit_signals

        self.trade_log = self.trade_log.append(exits, ignore_index=True)
        self.inventory.drop(self.inventory[exits_mask].index, inplace=True)
        self.capital -= sum(total_costs)

    def _process_entry_signals(self, entry_signals):
        """Returns a dictionary containing the orders to execute."""

        if not entry_signals.empty:
            costs = entry_signals['totals']['cost']
            return entry_signals.loc[costs.idxmin():costs.idxmin()], costs.min(
            )
        else:
            return entry_signals, 0

    def __repr__(self):
        return "Backtest(capital={}, strategy={})".format(
            self.capital, self._strategy)
