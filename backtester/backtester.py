import pandas as pd
import pyprind

from .strategy import Strategy
from .datahandler import HistoricalOptionsData


class Backtest:
    """Processes signals from the Strategy object"""
    def __init__(self, capital=1_000_000):
        self.initial_capital = self.current_capital = capital
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

    def run(self, monthly=False):
        """Runs the backtest and returns a `pd.DataFrame` of the orders executed (`self.trade_log`)

        Args:
            monthly (bool, optional): Iterates through data monthly rather than daily. Defaults to False.

        Returns:
            pd.DataFrame: Log of the trades executed.
        """

        assert self._data is not None
        assert self._strategy is not None
        assert self._data.schema == self._strategy.schema

        index = pd.MultiIndex.from_product(
            [[l.name for l in self._strategy.legs],
             ['contract', 'underlying', 'expiration', 'type', 'strike', 'cost', 'date', 'order']])
        index_totals = pd.MultiIndex.from_product([['totals'], ['cost']])
        self.inventory = pd.DataFrame(columns=index.append(index_totals))
        self.trade_log = pd.DataFrame()

        data_iterator = self._data.iter_months() if monthly else self._data.iter_dates()
        bar = pyprind.ProgBar(data_iterator.ngroups, bar_char='█')

        for _date, options in data_iterator:
            entry_signals = self._strategy.filter_entries(options, self.inventory)
            exit_signals = self._strategy.filter_exits(options, self.inventory)

            self._execute_exit(exit_signals)
            self._execute_entry(entry_signals)

            bar.update()

        return self.trade_log

    def _execute_entry(self, entry_signals):
        """Executes entry orders and updates `self.inventory` and `self.trade_log`"""
        entry, total_price = self._process_entry_signals(entry_signals)

        if (not self.stop_if_broke) or (self.current_capital >= total_price):
            self.inventory = self.inventory.append(entry, ignore_index=True)
            self.trade_log = self.trade_log.append(entry, ignore_index=True)
            self.current_capital -= total_price

    def _execute_exit(self, exit_signals):
        """Executes exits and updates `self.inventory` and `self.trade_log`"""
        if exit_signals is None:
            return
        exits, exits_mask, total_costs = exit_signals

        self.trade_log = self.trade_log.append(exits, ignore_index=True)
        self.inventory.drop(self.inventory[exits_mask].index, inplace=True)
        self.current_capital -= sum(total_costs)

    def _process_entry_signals(self, entry_signals):
        """Returns a dictionary containing the orders to execute."""

        if not entry_signals.empty:
            # costs = entry_signals['totals']['cost']
            # return entry_signals.loc[costs.idxmin():costs.idxmin()], costs.min()
            return entry_signals.iloc[0], entry_signals.iloc[0]['totals']['cost']
        else:
            return entry_signals, 0

    def summary(self):
        df = self.trade_log

        entries_mask = df.apply(lambda row: row['leg_1']['order'][2] == 'O', axis=1)
        entries = df.loc[entries_mask]
        exits = df.loc[~entries_mask]
        trades = entries.merge(exits,
                               on=[(l.name, 'contract') for l in self._strategy.legs],
                               suffixes=['_entry', '_exit'])

        costs = trades.apply(lambda row: row['totals_entry']['cost'] + row['totals_exit']['cost'], axis=1)
        wins_mask = costs < 0
        total_trades = len(trades)
        win_number = sum(wins_mask)
        loss_number = total_trades - win_number
        win_pct = win_number / total_trades
        largest_loss = costs.max()

        data = [total_trades, win_number, loss_number, win_pct, largest_loss]
        stats = ['Total trades', 'Number of wins', 'Number of losses', 'Win %', 'Largest loss']
        strat = ['Strategy']
        summary = pd.DataFrame(data, stats, strat)
        return summary

    def __repr__(self):
        return "Backtest(capital={}, strategy={})".format(self.current_capital, self._strategy)
