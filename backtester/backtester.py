import pandas as pd
import numpy as np
import pyprind

from .strategy import Strategy, Order
from .datahandler import HistoricalOptionsData


class Backtest:
    """Processes signals from the Strategy object"""
    def __init__(self):
        self._strategy = None
        self._data = None
        self.stop_if_broke = True

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strat):
        assert isinstance(strat, Strategy)
        self._strategy = strat
        self.current_cash = strat.initial_capital

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
            monthly (bool, optional):   Iterates through data monthly rather than daily. Defaults to False.

        Returns:
            pd.DataFrame:               Log of the trades executed.
        """

        assert self._data is not None
        assert self._strategy is not None
        assert self._data.schema == self._strategy.schema

        columns = pd.MultiIndex.from_product(
            [[l.name for l in self._strategy.legs],
             ['contract', 'underlying', 'expiration', 'type', 'strike', 'cost', 'order']])
        totals = pd.MultiIndex.from_product([['totals'], ['cost', 'qty', 'date']])
        self.inventory = pd.DataFrame(columns=columns.append(totals))
        self.trade_log = pd.DataFrame()
        self.balance = pd.DataFrame({
            'capital': self.current_cash,
            'cash': self.current_cash
        },
                                    index=[self.data.start_date - pd.Timedelta(1, unit='day')])

        data_iterator = self._data.iter_months() if monthly else self._data.iter_dates()
        bar = pyprind.ProgBar(data_iterator.ngroups, bar_char='█')

        for date, options in data_iterator:
            entry_signals = self._strategy.filter_entries(options, self.inventory, date)
            exit_signals = self._strategy.filter_exits(options, self.inventory, date)

            self._execute_exit(exit_signals)
            self._execute_entry(entry_signals)
            self._update_balance(date, options)

            bar.update()

        self.balance['% change'] = self.balance['capital'].pct_change()
        self.balance['accumulated return'] = (1.0 + self.balance['% change']).cumprod()

        return self.trade_log

    def _execute_entry(self, entry_signals):
        """Executes entry orders and updates `self.inventory` and `self.trade_log`"""
        entry, total_price = self._process_entry_signals(entry_signals)

        if (not self.stop_if_broke) or (self.current_cash >= total_price):
            self.inventory = self.inventory.append(entry, ignore_index=True)
            self.trade_log = self.trade_log.append(entry, ignore_index=True)
            self.current_cash -= total_price

    def _execute_exit(self, exit_signals):
        """Executes exits and updates `self.inventory` and `self.trade_log`"""
        exits, exits_mask, total_costs = exit_signals

        self.trade_log = self.trade_log.append(exits, ignore_index=True)
        self.inventory.drop(self.inventory[exits_mask].index, inplace=True)
        self.current_cash -= sum(total_costs)

    def _process_entry_signals(self, entry_signals):
        """Returns the entry signals to execute and their cost."""

        if not entry_signals.empty:
            # costs = entry_signals['totals']['cost']
            # return entry_signals.loc[costs.idxmin():costs.idxmin()], costs.min()
            entry = entry_signals.iloc[0]
            return entry, entry['totals']['cost'] * entry['totals']['qty']
        else:
            return entry_signals, 0

    def _update_balance(self, date, options):
        """Updates positions and calculates statistics for the current date.

        Args:
            date (pd.Timestamp):    Current date.
            options (pd.DataFrame): DataFrame of (daily/monthly) options.
        """

        leg_candidates = [
            self._strategy._exit_candidates(l.direction, self.inventory[l.name], options) for l in self._strategy.legs
        ]

        # If a contract is missing we replace the NaN values with those of the inventory
        # except for cost, which we imput as zero.
        for leg in leg_candidates:
            leg['cost'].fillna(0, inplace=True)

        calls_value = -np.sum(
            np.sum(leg['cost'] * self.inventory['totals']['qty']
                   for leg in leg_candidates if (leg['type'] == 'call').any()))
        puts_value = -np.sum(
            np.sum(leg['cost'] * self.inventory['totals']['qty']
                   for leg in leg_candidates if (leg['type'] == 'put').any()))

        capital = calls_value + puts_value + self.current_cash

        row = pd.Series(
            {
                'qty': self.inventory['totals']['qty'].sum(),
                'calls value': calls_value,
                'puts value': puts_value,
                'cash': self.current_cash,
                'capital': capital,
            },
            name=date)
        self.balance = self.balance.append(row)

    def summary(self):
        """Returns a table with summary statistics about the trade log"""
        df = self.trade_log
        balance = self.balance
        df.loc[:,
               ('totals',
                'capital')] = (-df['totals']['cost'] * df['totals']['qty']).cumsum() + self._strategy.initial_capital

        daily_returns = balance['% change'] * 100

        first_leg = self._strategy.legs[0].name

        entry_mask = df[first_leg].eval('(order == @Order.BTO) | (order == @Order.STO)')
        entries = df.loc[entry_mask]
        exits = df.loc[~entry_mask]

        costs = np.array([])
        for contract in entries[first_leg]['contract']:
            entry = entries.loc[entries[first_leg]['contract'] == contract]
            exit_ = exits.loc[exits[first_leg]['contract'] == contract]
            try:
                # Here we assume we are entering only once per contract (i.e both entry and exit_ have only one row)
                costs = np.append(costs, (entry['totals']['cost'] * entry['totals']['qty']).values[0] +
                                  (exit_['totals']['cost'] * exit_['totals']['qty']).values[0])
            except IndexError:
                continue

        # trades = entries.merge(exits,
        #                        on=[(l.name, 'contract') for l in self._strategy.legs],
        #                        suffixes=['_entry', '_exit'])

        # costs = trades.apply(lambda row: row['totals_entry']['cost'] + row['totals_exit']['cost'], axis=1)

        wins = costs < 0
        losses = costs >= 0
        profit_factor = np.sum(wins) / np.sum(losses)
        total_trades = len(exits)
        win_number = np.sum(wins)
        loss_number = total_trades - win_number
        win_pct = (win_number / total_trades) * 100
        largest_loss = np.max(costs)
        avg_profit = np.mean(-costs)
        avg_pl = np.mean(daily_returns)
        total_pl = (df['totals']['capital'].iloc[-1] / self._strategy.initial_capital) * 100

        data = [
            total_trades, win_number, loss_number, win_pct, largest_loss, profit_factor, avg_profit, avg_pl, total_pl
        ]
        stats = [
            'Total trades', 'Number of wins', 'Number of losses', 'Win %', 'Largest loss', 'Profit factor',
            'Average profit', 'Average P&L %', 'Total P&L %'
        ]
        strat = ['Strategy']
        summary = pd.DataFrame(data, stats, strat)

        # Applies formatters to rows
        def format_row_wise(styler, formatters):
            for row, row_formatter in formatters.items():
                row_num = styler.index.get_loc(row)

                for col_num in range(len(styler.columns)):
                    styler._display_funcs[(row_num, col_num)] = row_formatter

            return styler

        formatters = {
            "Total trades": lambda x: f"{x:.0f}",
            "Number of wins": lambda x: f"{x:.0f}",
            "Number of losses": lambda x: f"{x:.0f}",
            "Win %": lambda x: f"{x:.2f}%",
            "Largest loss": lambda x: f"${x:.2f}",
            "Profit factor": lambda x: f"{x:.2f}",
            "Average profit": lambda x: f"${x:.2f}",
            "Average P&L %": lambda x: f"{x:.2f}%",
            "Total P&L %": lambda x: f"{x:.2f}%"
        }

        styler = format_row_wise(summary.style, formatters)

        return styler

    def __repr__(self):
        return "Backtest(capital={}, strategy={})".format(self.current_cash, self._strategy)
