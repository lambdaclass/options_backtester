from functools import reduce

import numpy as np
import pandas as pd
import pyprind

from .enums import Order, Stock, Signal, Direction, get_order
from .datahandler import HistoricalOptionsData, TiingoData
from .strategy import Strategy


class Backtest:
    """Backtest runner class."""
    def __init__(self, allocation, initial_capital=1_000_000, shares_per_contract=100):
        assert isinstance(allocation, dict)

        assets = ('stocks', 'options', 'cash')
        total_allocation = sum(allocation.get(a, 0.0) for a in assets)

        self.allocation = {}
        for asset in assets:
            self.allocation[asset] = allocation.get(asset, 0.0) / total_allocation

        self.initial_capital = initial_capital
        self.stop_if_broke = True
        self.shares_per_contract = shares_per_contract
        self._stocks = []
        self._options_strategy = None
        self._stocks_data = None
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

    @property
    def stocks_data(self):
        return self._stocks_data

    @stocks_data.setter
    def stocks_data(self, data):
        assert isinstance(data, TiingoData)
        self._stocks_schema = data.schema
        self._stocks_data = data

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

        assert self._stocks_data, 'Stock data not set'
        assert self._options_data, 'Options data not set'
        assert self._options_strategy, 'Options Strategy not set'
        assert self._options_data.schema == self._options_strategy.schema

        option_dates = self._options_data['date'].unique()
        stock_dates = self._stocks_data['date'].unique()
        assert np.array_equal(stock_dates,
                              option_dates), 'Stock and options dates do not match (check that TZ are equal)'

        self._initialize_inventories()
        self.current_cash = self.initial_capital
        self.trade_log = pd.DataFrame()
        self.balance = pd.DataFrame({
            'total capital': self.current_cash,
            'cash': self.current_cash
        },
                                    index=[self._stocks_data.start_date - pd.Timedelta(1, unit='day')])

        if sma_days:
            self._stocks_data.sma(sma_days)

        rebalancing_days = pd.date_range(
            self._stocks_data.start_date, self._stocks_data.end_date, freq=str(rebalance_freq) +
            'BMS') if rebalance_freq else []

        data_iterator = self._data_iterator(monthly)
        bar = pyprind.ProgBar(len(stock_dates), bar_char='█')

        for date, stocks, options in data_iterator:
            if date in rebalancing_days or date == self._stocks_data.start_date:
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
            generator: Daily/monthly iterator over `self._stocks_data` and `self.options_data`.
        """

        if monthly:
            it = zip(self._stocks_data.iter_months(), self._options_data.iter_months())
        else:
            it = zip(self._stocks_data.iter_dates(), self._options_data.iter_dates())

        return ((date, stocks, options) for (date, stocks), (_, options) in it)

    def _rebalance_portfolio(self, date, stocks, options, sma_days):
        """Reabalances the portfolio according to `self.allocation` weights.

        Args:
            date (pd.Timestamp):    Current date.
            stocks (pd.DataFrame):  Stocks data for the current date.
            options (pd.DataFrame): Options data for the current date.
            sma_days (int):         SMA window size
        """

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
        self._execute_entry(entry_signals, options_allocation)

        stocks_value = sum(self._stocks_inventory['price'] * self._stocks_inventory['qty'])
        options_value = sum(self._options_inventory['totals']['cost'] * self._options_inventory['totals']['qty'])

        # Update current cash
        self.current_cash = total_capital - options_value - stocks_value

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

    def _buy_stocks(self, stocks, allocation, sma_days):
        """Buys stocks according to their given weight, optionally using an SMA entry filter.

        Args:
            stocks (pd.DataFrame):  Stocks data for the current time step.
            allocation (float):     Total capital allocation for stocks.
            sma_days (int):         SMA window.
        """

        stock_symbols = [stock.symbol for stock in self.stocks]
        query = '{} in {}'.format(self._stocks_schema['symbol'], stock_symbols)
        inventory_stocks = stocks.query(query)
        stock_percentages = np.array([stock.percentage for stock in self.stocks])
        stock_prices = inventory_stocks[self._stocks_schema['adjClose']]

        if sma_days:
            qty = np.where(inventory_stocks['sma'] < stock_prices, (allocation * stock_percentages) // stock_prices, 0)
        else:
            qty = (allocation * stock_percentages) // stock_prices

        self._stocks_inventory = pd.DataFrame({'symbol': stock_symbols, 'price': stock_prices, 'qty': qty})

    def _update_balance(self, date, stocks, options):
        """Updates positions and calculates statistics for the current date.

        Args:
            date (pd.Timestamp):    Current date.
            stocks (pd.DataFrame):  DataFrame of stocks.
            options (pd.DataFrame): DataFrame of (daily/monthly) options.
        """
        exit_signals = self.filter_exits(options, self._options_inventory, date)
        self._execute_exit(exit_signals)

        # update options
        leg_candidates = [
            self._exit_candidates(l.direction, self._options_inventory[l.name], options, self._options_inventory.index)
            for l in self._options_strategy.legs
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
            query = '{} == "{}"'.format(self._stocks_data.schema['symbol'], stock.symbol)
            stock_current = stocks.query(query)
            cost = stock_current[self._stocks_data.schema['adjClose']].values[0]
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

    def _execute_option_entries(self, date, options, options_allocation):
        """Enters option positions according to `self._options_strategy`.
        Calls `self._pick_entry_signals` to select from the entry signals given by the strategy.

        Args:
            date (pd.Timestamp):        Current date.
            options (pd.DataFrame):     Options data for the current time step.
            options_allocation (float): Capital amount allocated to options.
        """

        # Remove contracts already in inventory
        inventory_contracts = pd.concat(
            [self._options_inventory[leg.name]['contract'] for leg in self._options_strategy.legs])
        subset_options = options[~options[self.schema['contract']].isin(inventory_contracts)]

        entry_signals = []
        for leg in self._options_strategy.legs:
            flt = leg.entry_filter
            cost_field = leg.direction.value

            leg_entries = subset_options[flt(subset_options)]
            # Exit if no entry signals for the current leg
            if leg_entries.empty:
                return pd.DataFrame()

            fields = self._signal_fields(cost_field)
            leg_entries = leg_entries.reindex(columns=fields.keys())
            leg_entries.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, Signal.ENTRY)
            leg_entries['order'] = order

            # Change sign of cost for SELL orders
            if leg.direction == Direction.SELL:
                leg_entries['cost'] = -leg_entries['cost']

            leg_entries['cost'] *= self._shares_per_contract
            leg_entries.columns = pd.MultiIndex.from_product([[leg.name], leg_entries.columns])
            entry_signals.append(leg_entries.reset_index(drop=True))

        # Append the 'totals' column to entry_signals
        total_costs = sum(leg_entries['cost'] for leg_entries in entry_signals)
        qty = np.abs(options_allocation // total_costs)
        totals = pd.DataFrame.from_dict({'cost': total_costs, 'qty': qty, 'date': date})
        totals.columns = pd.MultiIndex.from_product([['totals'], totals.columns])
        entry_signals.append(totals)
        entry_signals = pd.concat(entry_signals, axis=1)

        entries = self._pick_entry_signals(entry_signals)

        # Update options inventory, trade log and current cash
        self._options_inventory = self._options_inventory.append(entries, ignore_index=True)
        self.trade_log = self.trade_log.append(entries, ignore_index=True)
        self.current_cash -= sum(total_costs)

    def _execute_option_exits(self, date, options):
        """Exits option positions according to `self._options_strategy`.
        Option positions are closed whenever the strategy signals an exit, when the profit/loss thresholds
        are exceeded or whenever the contracts in `self._options_inventory` are not found in `options`.

        Args:
            date (pd.Timestamp):        Current date.
            options (pd.DataFrame):     Options data for the current time step.
        """

        strategy = self._options_strategy
        current_options_quotes = self._get_current_option_quotes(options)

        filter_masks = []
        for i, leg in enumerate(strategy.legs):
            flt = leg.exit_filter

            # This mask is to ensure that legs with missing contracts exit.
            missing_contracts_mask = current_options_quotes[i]['cost'].isna()

            filter_masks.append(flt(current_options_quotes[i]) | missing_contracts_mask)
            fields = self._signal_fields((~leg.direction).value)
            current_options_quotes[i] = current_options_quotes[i].reindex(columns=fields.keys())
            current_options_quotes[i].rename(columns=fields, inplace=True)
            current_options_quotes[i].columns = pd.MultiIndex.from_product([[leg.name],
                                                                            current_options_quotes[i].columns])

        exit_candidates = pd.concat(current_options_quotes, axis=1)

        # If a contract is missing we replace the NaN values with those of the inventory
        # except for cost, which we imput as zero.
        exit_candidates = self._impute_missing_option_values(exit_candidates)

        # Append the 'totals' column to exit_candidates
        qtys = self._options_inventory['totals']['qty']
        total_costs = sum([exit_candidates[l.name]['cost'] for l in self.legs])
        totals = pd.DataFrame.from_dict({'cost': total_costs, 'qty': qtys, 'date': date})
        totals.columns = pd.MultiIndex.from_product([['totals'], totals.columns])
        exit_candidates = pd.concat([exit_candidates, totals], axis=1)

        # Compute which contracts need to exit, either because of price thresholds or user exit filters
        threshold_exits = strategy.filter_thresholds(self._options_inventory['totals']['cost'], total_costs)
        filter_mask = reduce(lambda x, y: x | y, filter_masks)
        exits_mask = threshold_exits | filter_mask

        exits = exit_candidates[exits_mask]
        total_costs = total_costs[exits_mask] * exits['totals']['qty']

        # Update options inventory, trade log and current cash
        self._options_inventory.drop(self._options_inventory[exits_mask].index, inplace=True)
        self.trade_log = self.trade_log.append(exits, ignore_index=True)
        self.current_cash -= sum(total_costs)

    def _pick_entry_signals(self, entry_signals):
        """Returns the entry signals to execute.

        Args:
            entry_signals (pd.DataFrame):   DataFrame of option entry signals chosen by the strategy.

        Returns:
            pd.DataFrame:                   DataFrame of entries to execute.
        """

        if not entry_signals.empty:
            # FIXME: This is a naive signal selection criterion, it simply picks the first one in `entry_singals`
            return entry_signals.iloc[0]
        else:
            return entry_signals

    def _signal_fields(self, cost_field):
        fields = {
            self.schema['contract']: 'contract',
            self.schema['underlying']: 'underlying',
            self.schema['expiration']: 'expiration',
            self.schema['type']: 'type',
            self.schema['strike']: 'strike',
            self.schema[cost_field]: 'cost',
            'order': 'order'
        }

        return fields

    def _get_current_option_quotes(self, options):
        """Returns the current quotes for all the options in `self._options_inventory` as a list of DataFrames.
        It also adds a `cost` column with the cost of closing the position in each contract and an `order` 
        column with the corresponding exit order type.

        Args:
            options (pd.DataFrame): Options data in the current time step.

        Returns:
            [pd.DataFrame]:         List of DataFrames, one for each leg in `self._options_inventory`,
                                    with the exit cost for the contracts.
        """

        current_options_quotes = []
        for leg in self._options_strategy.legs:
            inventory_leg = self._options_inventory[leg.name]

            # This is a left join to ensure that the result has the same length as the inventory. If the contract
            # isn't in the daily data the values will all be NaN and the filters should all yield False.
            leg_options = inventory_leg[['contract']].merge(options,
                                                            how='left',
                                                            left_on='contract',
                                                            right_on=leg.schema['contract'])

            # leg_options.index needs to be the same as the inventory's so that the exit masks that are constructed
            # from it can be correctly applied to the inventory.
            leg_options.index = self._options_inventory.index
            leg_options['order'] = get_order(leg.direction, Signal.EXIT)

            # Change sign of cost for SELL orders
            if ~leg.direction == Direction.SELL:
                leg_options['cost'] = -leg_options['cost']
            leg_options['cost'] *= self._shares_per_contract

            current_options_quotes.append(leg_options)

        return current_options_quotes

    def _impute_missing_option_values(self, exit_candidates):
        """Returns a copy of the inventory with the cost of all its contracts set to zero.

        Args:
            exit_candidates (pd.DataFrame): DataFrame of exit candidates with possible missing values.

        Returns:
            pd.DataFrame:                   Exit candidates with imputed values.
        """
        df = self._options_inventory.copy()
        for leg in self._options_strategy.legs:
            df.at[:, (leg.name, 'cost')] = 0

        return exit_candidates.fillna(df)

    def __repr__(self):
        return "Backtest(capital={}, allocation={}, stocks={}, strategy={})".format(
            self.current_cash, self.allocation, self._stocks, self._options_strategy)
