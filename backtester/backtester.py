from functools import reduce
from operator import add

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
        return self

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert isinstance(data, HistoricalOptionsData)
        self._data = data
        return self

    def run(self):
        """Runs the backtest and returns a `pd.DataFrame` of the orders executed."""
        assert self._data is not None
        assert self._strategy is not None

        self.trade_log = pd.DataFrame(
            columns=["date", "contract", "order", "qty", "profit", "capital"])

        for date, entry_signals, exit_signals in self._strategy.signals(
                self._data, self):
            self._execute_exit(date, exit_signals)
            self._execute_entry(date, entry_signals)

        return self.trade_log

    def _execute_entry(self, date, entry_signals):
        """Executes entry orders and updates `self.inventory` and `self.trade_log`"""
        if entry_signals.empty:
            return
        entry, total_price = self._process_entry_signals(entry_signals)
        cost = total_price * self.qty * self.shares_per_contract

        if (not self.stop_if_broke) or (self.capital >= cost):
            self.inventory = self.inventory.append(entry, ignore_index=True)
            for leg in self._strategy.legs:
                row = entry[leg.name]
                contract = row["contract"]
                order = row["order"]
                price = row["cost"] * self.shares_per_contract
                self.capital -= price
                self._update_trade_log(date, contract, order, self.qty, -price)

    def _execute_exit(self, date, exit_signals):
        """Executes exits and updates `self.inventory` and `self.trade_log`"""
        for contracts, price in exit_signals:
            for contract, order, individual_price in contracts:
                profit = individual_price * self.qty * self.shares_per_contract
                self.capital += profit
                self._update_trade_log(date, contract, order, self.qty, profit)
            for leg in self._strategy.legs:
                self.inventory = self.inventory.drop(
                    self.inventory[self.inventory[(
                        leg.name, 'contract')] == contract].index)

    def _process_entry_signals(self, entry_signals):
        """Returns a dictionary containing the orders to execute."""

        if not entry_signals.empty:
            legs = entry_signals.columns.levels[0]
            costs = reduce(add, (entry_signals[leg]["cost"] for leg in legs))
            return entry_signals.loc[costs.idxmin()], costs.min()
        else:
            return entry_signals, 0

    def _update_trade_log(self, date, contract, order, qty, profit):
        """Adds entry for the given order to `self.trade_log`."""
        self.trade_log.loc[len(self.trade_log)] = [
            date, contract, order, qty, profit, self.capital
        ]

    def __repr__(self):
        return "Backtest(capital={}, strategy={})".format(
            self.capital, self._strategy)
