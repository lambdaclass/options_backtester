import numpy as np
import pandas as pd
import pyprind

from .enums import Order, Stock
from .datahandler import HistoricalOptionsData, TiingoData
from .strategy import Strategy


class Backtest:
    """Processes signals from the Strategy object"""
    def __init__(self, allocation, initial_capital=1_000_000, shares_per_contract=100):
        assert isinstance(allocation, dict)

        assets = ('stocks', 'options', 'cash')
        total_allocation = sum(allocation.get(a, 0.0) for a in assets)

        self.allocation = {}
        for asset in assets:
            self.allocation[asset] = allocation.get(asset, 0.0) / total_allocation

        self.current_cash = self.initial_capital = initial_capital
        self.stop_if_broke = True
        self.shares_per_contract = shares_per_contract
        self._stocks = []
        self._options_strategy = None
        self._stock_data = None
        self._options_data = None

    @property
    def stocks(self):
        return self._stocks

    @stocks.setter
    def stocks(self, stocks):
        assert all(isinstance(stock, Stock) for stock in stocks), 'Invalid stocks'
        assert np.isclose(sum(stock.percentage for stock in stocks), 1.0,
                          atol=0.000001), 'Stock percentages must sum to 1.0'
        self._stocks = list(stocks)
        return self

    @property
    def strategy(self):
        return self._options_strategy

    @strategy.setter
    def strategy(self, strat):
        assert isinstance(strat, Strategy)
        self._options_strategy = strat
        self.current_cash = strat.initial_capital

    @property
    def stock_data(self):
        return self._stock_data

    @stock_data.setter
    def stock_data(self, data):
        assert isinstance(data, TiingoData)
        self._stocks_schema = data.schema
        self._stock_data = data
        self._stock_data.first_date = data['date'].min()
        self._stock_data.end_date = data['date'].max()

    @property
    def options_data(self):
        return self._options_data

    @options_data.setter
    def options_data(self, data):
        assert isinstance(data, HistoricalOptionsData)
        self._options_schema = data.schema
        self._options_data = data

    def run(self, rebalance_freq=0, monthly=False, sma_days=None):
        """Runs the backtest and returns a `pd.DataFrame` of the orders executed (`self.trade_log`)

        Args:
            rebalance_freq (int, optional): Determines the frequency of portfolio rebalances. Defaults to 0.
            monthly (bool, optional):       Iterates through data monthly rather than daily. Defaults to False.

        Returns:
            pd.DataFrame:                   Log of the trades executed.
        """

        assert self._stock_data, 'Stock data not set'
        assert self._options_data, 'Options data not set'
        assert self._options_strategy, 'Options Strategy not set'
        assert self._options_data.schema == self._options_strategy.schema

        option_dates = self._options_data['date'].unique()
        stock_dates = self._stock_data['date'].unique()
        assert np.array_equal(stock_dates, option_dates), 'Stock and options dates do not match'

        self._initialize_inventories()
        self.trade_log = pd.DataFrame()
        self.balance = pd.DataFrame({
            'total capital': self.current_cash,
            'cash': self.current_cash
        },
                                    index=[self.stock_data.start_date - pd.Timedelta(1, unit='day')])

        if sma_days:
            self._stock_data.sma(sma_days)

        rebalancing_days = pd.date_range(
            self.stock_data.first_date, self.stock_data.end_date, freq=str(rebalance_freq) +
            'BMS') if rebalance_freq else []

        data_iterator = self._data_iterator(monthly)
        bar = pyprind.ProgBar(len(stock_dates), bar_char='█')

        for date, stocks, options in data_iterator:
            if date in rebalancing_days or date == self.stock_data.start_date:
                self._rebalance_portfolio(date, stocks, options, sma_days)

            self._update_balance(date, stocks, options)
            bar.update()

        self.balance['% change'] = self.balance['total capital'].pct_change()
        self.balance['accumulated return'] = (1.0 + self.balance['% change']).cumprod()

        return self.trade_log

    def _initialize_inventories(self):
        """Initialize empty stocks and options inventories."""
        columns = pd.MultiIndex.from_product(
            [[l.name for l in self._options_strategy.legs],
             ['contract', 'underlying', 'expiration', 'type', 'strike', 'cost', 'order']])
        totals = pd.MultiIndex.from_product([['totals'], ['cost', 'qty', 'date']])
        self._options_inventory = pd.DataFrame(columns=columns.append(totals))

        self._stocks_inventory = pd.DataFrame(columns=['symbol', 'price', 'qty'])

    def _data_iterator(self, monthly):
        """Returns combined iterator for stock and options data.
        Each step, it produces a tuple like the following:
            (date, stocks, options)

        Returns:
            generator: Daily/monthly iterator over `self.stock_data` and `self.options_data`.
        """
        if monthly:
            it = zip(self._stock_data.iter_months(), self._options_data.iter_months())
        else:
            it = zip(self._stock_data.iter_dates(), self._options_data.iter_dates())

        return ((date, stocks, options) for (date, stocks), (_, options) in it)

    def _rebalance_portfolio(self, date, stocks, options, sma_days):
        """Rebalances the portfolio according to `self.allocation`."""
        # Sell all the options currently in the inventory
        self._sell_options(options, date)

        stock_capital = self._current_stock_capital(stocks)
        total_capital = self.current_cash + stock_capital
        options_allocation = self.allocation['options'] * total_capital
        stocks_allocation = self.allocation['stocks'] * total_capital

        # Clear inventories
        self._initialize_inventories()

        self._buy_stocks(stocks, stocks_allocation, sma_days)
        entry_signals = self._options_strategy.filter_entries(options, self._options_inventory, date,
                                                              options_allocation)
        self._execute_entry(entry_signals)

        stocks_value = sum(self._stocks_inventory['price'] * self._stocks_inventory['qty'])
        options_value = sum(self._options_inventory['totals']['cost'] * self._options_inventory['totals']['qty'])

        # Update current cash
        invested_capital = options_value + stocks_value
        self.current_cash = total_capital - invested_capital

    def _current_stock_capital(self, stocks):
        """Return the current value of the stocks inventory.

        Args:
            stocks (pd.DataFrame): Stocks data for the current time step.

        Returns:
            float: Total capital in stocks.
        """
        current_stocks = self._stocks_inventory.merge(stocks,
                                                      how='left',
                                                      left_on='symbol',
                                                      right_on=self._stocks_schema['symbol'])
        return (current_stocks[self._stocks_schema['adjClose']] * current_stocks['qty']).sum()

    def _current_options_capital(self, options):
        # Currently unused method
        total_cost = 0.0
        for leg in self._options_strategy.legs:
            current_options = self._options_inventory[leg.name].merge(options,
                                                                      how='left',
                                                                      left_on='contract',
                                                                      right_on=self._options_schema['contract'])
            price_col = (~leg.direction).value
            try:
                # 100 = shares_per_contract
                cost = current_options[price_col].fillna(
                    0.0).iloc[0] * self._options_inventory['totals']['qty'].values[0] * 100
                if price_col == 'bid':
                    total_cost += cost
                else:
                    total_cost -= cost
            except IndexError:
                total_cost += 0.0

        return total_cost

    def _sell_options(self, options, date):
        # This method essentially recycles most of the code in the filter_exits method in Strategy.
        # The whole thing needs a refactor.

        leg_candidates = [
            self._options_strategy._exit_candidates(l.direction, self._options_inventory[l.name], options,
                                                    self._options_inventory.index) for l in self._options_strategy.legs
        ]

        for i, leg in enumerate(self._options_strategy.legs):
            fields = self._options_strategy._signal_fields((~leg.direction).value)
            leg_candidates[i] = leg_candidates[i].loc[:, fields.values()]
            leg_candidates[i].columns = pd.MultiIndex.from_product([["leg_{}".format(i + 1)],
                                                                    leg_candidates[i].columns])

        candidates = pd.concat(leg_candidates, axis=1)

        # If a contract is missing we replace the NaN values with those of the inventory
        # except for cost, which we imput as zero.
        imputed_inventory = self._options_strategy._imput_missing_data(self._options_inventory)
        candidates = candidates.fillna(imputed_inventory)
        total_costs = sum([candidates[l.name]['cost'] for l in self._options_strategy.legs])

        # Append the 'totals' column to candidates
        qtys = self._options_inventory['totals']['qty']
        dates = [date] * len(self._options_inventory)
        totals = pd.DataFrame.from_dict({"cost": total_costs, "qty": qtys, "date": dates})
        totals.columns = pd.MultiIndex.from_product([["totals"], totals.columns])
        candidates = pd.concat([candidates, totals], axis=1)

        exits_mask = pd.Series([True] * len(self._options_inventory))
        exits_mask.index = self._options_inventory.index

        total_costs *= candidates['totals']['qty']

        self._execute_exit((candidates, exits_mask, total_costs))

    def _buy_stocks(self, stocks, allocation, sma_days):
        for stock in self._stocks:
            query = '{} == "{}"'.format(self._stocks_schema['symbol'], stock.symbol)
            stock_row = stocks.query(query)
            stock_price = stock_row[self._stocks_schema['adjClose']].values[0]

            if sma_days is not None:
                if stock_row['sma'].values[0] < stock_price:
                    qty = (allocation * stock.percentage) // stock_price
                else:
                    qty = 0
            else:
                qty = (allocation * stock.percentage) // stock_price

            stock_entry = pd.Series([stock.symbol, stock_price, qty], index=self._stocks_inventory.columns)
            self._stocks_inventory = self._stocks_inventory.append(stock_entry, ignore_index=True)

    def _execute_entry(self, entry_signals):
        """Executes entry orders and updates `self.inventory` and `self.trade_log`"""
        entry, total_price = self._process_entry_signals(entry_signals)

        self._options_inventory = self._options_inventory.append(entry, ignore_index=True)
        self.trade_log = self.trade_log.append(entry, ignore_index=True)

    def _execute_exit(self, exit_signals):
        """Executes exits and updates `self.inventory` and `self.trade_log`"""
        exits, exits_mask, total_costs = exit_signals

        self.trade_log = self.trade_log.append(exits, ignore_index=True)
        self._options_inventory.drop(self._options_inventory[exits_mask].index, inplace=True)
        self.current_cash -= sum(total_costs)

    def _process_entry_signals(self, entry_signals):
        """Returns the entry signals to execute and their cost."""

        if not entry_signals.empty:
            entry = entry_signals.iloc[0]
            return entry, entry['totals']['cost'] * entry['totals']['qty']
        else:
            return entry_signals, 0

    def _update_balance(self, date, stocks, options):
        """Updates positions and calculates statistics for the current date.

        Args:
            date (pd.Timestamp):    Current date.
            stocks (pd.DataFrame):  DataFrame of stocks
            options (pd.DataFrame): DataFrame of (daily/monthly) options.
        """
        exit_signals = self._options_strategy.filter_exits(options, self._options_inventory, date)
        self._execute_exit(exit_signals)

        # update options
        leg_candidates = [
            self._options_strategy._exit_candidates(l.direction, self._options_inventory[l.name], options,
                                                    self._options_inventory.index) for l in self._options_strategy.legs
        ]

        # If a contract is missing we replace the NaN values with those of the inventory
        # except for cost, which we imput as zero.

        for leg in leg_candidates:
            leg['cost'].fillna(0, inplace=True)

        calls_value = -np.sum(
            sum(leg['cost'] * self._options_inventory['totals']['qty']
                for leg in leg_candidates if (leg['type'] == 'call').any()))
        puts_value = -np.sum(
            sum(leg['cost'] * self._options_inventory['totals']['qty']
                for leg in leg_candidates if (leg['type'] == 'put').any()))

        options_capital = calls_value + puts_value
        self.options_capital = options_capital
        # update stocks portfolio information due to change in price over time
        costs = []
        for stock in self.stocks:
            query = '{} == "{}"'.format(self.stock_data.schema['symbol'], stock.symbol)
            stock_current = stocks.query(query)
            cost = stock_current[self.stock_data.schema['adjClose']].values[0]
            stock_inventory = self._stocks_inventory.query(query)
            try:
                qty = stock_inventory['qty'].values[0]
            except IndexError:
                qty = 0

            costs.append(cost * qty)
        total_value = sum(costs)

        self.stock_capital = total_value
        self.total_capital = self.stock_capital + self.options_capital + self.current_cash

        row = pd.Series(
            {
                'total capital': self.stock_capital + self.options_capital,
                'cash': self.current_cash,
                'stocks capital': self.stock_capital,
                'stocks qty': self._stocks_inventory['qty'].sum(),
                'options capital': options_capital,
                'options qty': self._options_inventory['totals']['qty'].sum(),
                'calls capital': calls_value,
                'puts capital': puts_value
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
        largest_loss = max(0, np.max(costs))
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
        return "Backtest(capital={}, allocation={}, stocks={}, strategy={})".format(
            self.current_cash, self.allocation, self._stocks, self._options_strategy)
