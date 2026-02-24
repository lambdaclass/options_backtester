from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Generator, Union

import numpy as np
import pandas as pd
import pyprind

from .enums import *


class Backtest:
    """Backtest runner class."""
    def __init__(self, allocation: dict[str, float], initial_capital: int = 1_000_000, shares_per_contract: int = 100) -> None:
        assets = ('stocks', 'options', 'cash')
        total_allocation = sum(allocation.get(a, 0.0) for a in assets)

        self.allocation: dict[str, float] = {}
        for asset in assets:
            self.allocation[asset] = allocation.get(asset, 0.0) / total_allocation

        self.initial_capital = initial_capital
        self.stop_if_broke = True
        self.shares_per_contract = shares_per_contract
        self.options_budget: Union[Callable[[pd.Timestamp, float], float], float, None] = None
        self._stocks: list[Stock] = []
        self._options_strategy: Strategy | None = None
        self._stocks_data: TiingoData | None = None
        self._options_data: HistoricalOptionsData | None = None
        self.run_metadata: dict[str, Any] = {}

    @property
    def stocks(self) -> list[Stock]:
        return self._stocks

    @stocks.setter
    def stocks(self, stocks: list[Stock]) -> None:
        assert np.isclose(sum(stock.percentage for stock in stocks), 1.0,
                          atol=0.000001), 'Stock percentages must sum to 1.0'
        self._stocks = list(stocks)

    @property
    def options_strategy(self) -> Strategy | None:
        return self._options_strategy

    @options_strategy.setter
    def options_strategy(self, strat: Strategy) -> None:
        self._options_strategy = strat

    @property
    def stocks_data(self) -> TiingoData | None:
        return self._stocks_data

    @stocks_data.setter
    def stocks_data(self, data: TiingoData) -> None:
        self._stocks_schema = data.schema
        self._stocks_data = data

    @property
    def options_data(self) -> HistoricalOptionsData | None:
        return self._options_data

    @options_data.setter
    def options_data(self, data: HistoricalOptionsData) -> None:
        self._options_schema = data.schema
        self._options_data = data

    def run(self, rebalance_freq: int = 0, monthly: bool = False, sma_days: int | None = None,
            rebalance_unit: str = 'BMS') -> pd.DataFrame:
        """Runs the backtest and returns a `pd.DataFrame` of the orders executed (`self.trade_log`)

        Args:
            rebalance_freq (int, optional): Determines the frequency of portfolio rebalances. Defaults to 0.
            monthly (bool, optional):       Iterates through data monthly rather than daily. Defaults to False.
            rebalance_unit (str, optional): Pandas frequency unit. 'BMS' = business month start (default),
                                            'B' = every business day, 'W-MON' = weekly on Monday,
                                            '2W-MON' = biweekly. Combined with rebalance_freq as
                                            f'{rebalance_freq}{rebalance_unit}'.

        Returns:
            pd.DataFrame:                   Log of the trades executed.
        """

        assert self._stocks_data, 'Stock data not set'
        assert all(stock.symbol in self._stocks_data['symbol'].values
                   for stock in self._stocks), 'Ensure all stocks in portfolio are present in the data'
        assert self._options_data, 'Options data not set'
        assert self._options_strategy, 'Options Strategy not set'
        assert self._options_data.schema == self._options_strategy.schema

        option_dates = self._options_data['date'].unique()
        stock_dates = self.stocks_data['date'].unique()
        assert np.array_equal(stock_dates,
                              option_dates), 'Stock and options dates do not match (check that TZ are equal)'

        self._initialize_inventories()
        self.current_cash: float = self.initial_capital
        self._trade_log_parts: list[pd.DataFrame] = []
        initial_balance = pd.DataFrame({
            'total capital': self.current_cash,
            'cash': self.current_cash
        },
                                       index=[self.stocks_data.start_date - pd.Timedelta(1, unit='day')])
        self._balance_parts: list[pd.DataFrame] = [initial_balance]

        if sma_days:
            self.stocks_data.sma(sma_days)

        dates = pd.DataFrame(self.options_data._data[['quotedate',
                                                      'volume']]).drop_duplicates('quotedate').set_index('quotedate')
        if rebalance_freq:
            freq_str = str(rebalance_freq) + rebalance_unit
            rebalancing_days: pd.DatetimeIndex = pd.to_datetime(
                dates.groupby(pd.Grouper(freq=freq_str)).apply(lambda x: x.index.min()).values)
        else:
            rebalancing_days = []

        data_iterator = self._data_iterator(monthly)
        bar = pyprind.ProgBar(len(stock_dates), bar_char='█')

        for date, stocks, options in data_iterator:
            if (date in rebalancing_days):
                previous_rb_date = rebalancing_days[rebalancing_days.get_loc(date) -
                                                    1] if rebalancing_days.get_loc(date) != 0 else date
                self._update_balance(previous_rb_date, date)
                self._rebalance_portfolio(date, stocks, options, sma_days)

            bar.update()

        # Update balance for the period between the last rebalancing day and the last day
        self._update_balance(rebalancing_days[-1], self.stocks_data.end_date)

        # Assemble trade_log and balance from accumulated parts
        self.trade_log: pd.DataFrame = pd.concat(self._trade_log_parts, ignore_index=True) if self._trade_log_parts else pd.DataFrame()
        self.balance: pd.DataFrame = pd.concat(self._balance_parts, sort=False)
        # Ensure numeric dtypes after concat (mixed int/float parts can produce object dtype)
        for col in self.balance.columns:
            self.balance[col] = pd.to_numeric(self.balance[col], errors='coerce')

        self.balance['options capital'] = self.balance['calls capital'] + self.balance['puts capital']
        self.balance['stocks capital'] = sum(self.balance[stock.symbol] for stock in self._stocks)
        first_idx = self.balance.index[0]
        self.balance.loc[first_idx, 'stocks capital'] = 0
        self.balance.loc[first_idx, 'options capital'] = 0
        self.balance[
            'total capital'] = self.balance['cash'] + self.balance['stocks capital'] + self.balance['options capital']
        self.balance['% change'] = self.balance['total capital'].pct_change()
        self.balance['accumulated return'] = (1.0 + self.balance['% change']).cumprod()
        self._attach_run_metadata(
            rebalance_freq=rebalance_freq,
            monthly=monthly,
            sma_days=sma_days,
            rebalance_unit=rebalance_unit,
        )

        return self.trade_log

    def _attach_run_metadata(
        self,
        rebalance_freq: int,
        monthly: bool,
        sma_days: int | None,
        rebalance_unit: str,
    ) -> None:
        metadata = self._build_run_metadata(
            rebalance_freq=rebalance_freq,
            monthly=monthly,
            sma_days=sma_days,
            rebalance_unit=rebalance_unit,
        )
        self.run_metadata = metadata
        self.balance.attrs['run_metadata'] = metadata
        self.trade_log.attrs['run_metadata'] = metadata

    def _build_run_metadata(
        self,
        rebalance_freq: int,
        monthly: bool,
        sma_days: int | None,
        rebalance_unit: str,
    ) -> dict[str, Any]:
        stocks = [{'symbol': stock.symbol, 'percentage': float(stock.percentage)} for stock in self._stocks]
        run_config = {
            'allocation': {k: float(v) for k, v in self.allocation.items()},
            'initial_capital': float(self.initial_capital),
            'shares_per_contract': int(self.shares_per_contract),
            'rebalance_freq': int(rebalance_freq),
            'rebalance_unit': rebalance_unit,
            'monthly': bool(monthly),
            'sma_days': int(sma_days) if sma_days is not None else None,
            'stocks': stocks,
        }
        data_snapshot = self._data_snapshot()
        return {
            'framework': 'backtester.Backtest',
            'dispatch_mode': 'python-legacy',
            'rust_available': False,
            'git_sha': self._git_sha(),
            'run_at_utc': datetime.now(timezone.utc).isoformat(),
            'config_hash': self._sha256_json(run_config),
            'data_snapshot_hash': self._sha256_json(data_snapshot),
            'data_snapshot': data_snapshot,
        }

    def _data_snapshot(self) -> dict[str, Any]:
        options_dates = self._options_data['date']
        stocks_dates = self._stocks_data['date']
        return {
            'options_rows': int(len(self._options_data._data)),
            'stocks_rows': int(len(self._stocks_data._data)),
            'options_date_start': pd.Timestamp(options_dates.min()).isoformat(),
            'options_date_end': pd.Timestamp(options_dates.max()).isoformat(),
            'stocks_date_start': pd.Timestamp(stocks_dates.min()).isoformat(),
            'stocks_date_end': pd.Timestamp(stocks_dates.max()).isoformat(),
            'options_columns': list(self._options_data._data.columns),
            'stocks_columns': list(self._stocks_data._data.columns),
        }

    @staticmethod
    def _sha256_json(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    @staticmethod
    def _git_sha() -> str:
        repo_root = Path(__file__).resolve().parents[1]
        try:
            proc = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip()
        except Exception:
            return 'unknown'

    def _initialize_inventories(self) -> None:
        """Initialize empty stocks and options inventories."""
        columns = pd.MultiIndex.from_product(
            [[l.name for l in self._options_strategy.legs],
             ['contract', 'underlying', 'expiration', 'type', 'strike', 'cost', 'order']])
        totals = pd.MultiIndex.from_product([['totals'], ['cost', 'qty', 'date']])
        self._options_inventory: pd.DataFrame = pd.DataFrame(columns=pd.Index(columns.tolist() + totals.tolist()))

        self._stocks_inventory: pd.DataFrame = pd.DataFrame(columns=['symbol', 'price', 'qty'])

    def _data_iterator(self, monthly: bool) -> Generator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame], None, None]:
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

    def _rebalance_portfolio(self, date: pd.Timestamp, stocks: pd.DataFrame, options: pd.DataFrame, sma_days: int | None) -> None:
        """Reabalances the portfolio according to `self.allocation` weights.

        Args:
            date (pd.Timestamp):    Current date.
            stocks (pd.DataFrame):  Stocks data for the current date.
            options (pd.DataFrame): Options data for the current date.
            sma_days (int):         SMA window size
        """

        self._execute_option_exits(date, options)

        stock_capital = self._current_stock_capital(stocks)
        options_capital = self._current_options_capital(options)
        total_capital = self.current_cash + stock_capital + options_capital

        # buy stocks
        stocks_allocation = self.allocation['stocks'] * total_capital
        self._stocks_inventory = pd.DataFrame(columns=['symbol', 'price', 'qty'])

        # We simulate a sell of the stock positions and then a rebuy.
        # This would **not** work if we added transaction fees.
        self.current_cash = stocks_allocation + total_capital * self.allocation['cash']
        self._buy_stocks(stocks, stocks_allocation, sma_days)

        # exit/enter contracts
        if self.options_budget is not None:
            budget = self.options_budget
            options_allocation: float = budget(date, total_capital) if callable(budget) else budget
        else:
            options_allocation = self.allocation['options'] * total_capital
        if options_allocation >= options_capital:
            self._execute_option_entries(date, options, options_allocation - options_capital)
        else:
            to_sell = options_capital - options_allocation
            current_options = self._get_current_option_quotes(options)
            self._sell_some_options(date, to_sell, current_options)

    def _sell_some_options(self, date: pd.Timestamp, to_sell: float, current_options: list[pd.DataFrame]) -> None:
        sold: float = 0
        total_costs = sum([current_options[i]['cost'] for i in range(len(current_options))])
        trade_rows: list[pd.Series] = []
        for (exit_cost, (row_index, inventory_row)) in zip(total_costs, self._options_inventory.iterrows()):
            if exit_cost == 0:
                continue
            if (to_sell - sold > -exit_cost) and (to_sell - sold) > 0:
                qty_to_sell = (to_sell - sold) // exit_cost
                if -qty_to_sell <= inventory_row['totals']['qty']:
                    qty_to_sell = (to_sell - sold) // exit_cost
                else:
                    if qty_to_sell != 0:
                        qty_to_sell = -inventory_row['totals']['qty']
                if qty_to_sell != 0:
                    trade_log_append = self._options_inventory.loc[row_index].copy()
                    trade_log_append['totals', 'qty'] = -qty_to_sell
                    trade_log_append['totals', 'date'] = date
                    trade_log_append['totals', 'cost'] = exit_cost
                    for i, leg in enumerate(self._options_strategy.legs):
                        trade_log_append[leg.name, 'order'] = ~trade_log_append[leg.name, 'order']
                        trade_log_append[leg.name, 'cost'] = current_options[i].loc[row_index]['cost']

                    trade_rows.append(trade_log_append)
                    self._options_inventory.at[row_index, ('totals', 'date')] = date
                    self._options_inventory.at[row_index, ('totals', 'qty')] += qty_to_sell

                sold += (qty_to_sell * exit_cost)

        if trade_rows:
            self._trade_log_parts.append(pd.DataFrame(trade_rows))
        self.current_cash += sold - to_sell

    def _current_stock_capital(self, stocks: pd.DataFrame) -> float:
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

    def _current_options_capital(self, options: pd.DataFrame) -> float:
        options_value = self._get_current_option_quotes(options)
        values_by_row: Any = [0] * len(options_value[0])
        if len(options_value[0]) != 0:
            for i in range(len(self._options_strategy.legs)):
                values_by_row += options_value[i]['cost'].values
            total: float = -sum(values_by_row * self._options_inventory['totals']['qty'].values)
        else:
            total = 0
        return total

    def _buy_stocks(self, stocks: pd.DataFrame, allocation: float, sma_days: int | None) -> None:
        """Buys stocks according to their given weight, optionally using an SMA entry filter.
        Updates `self._stocks_inventory` and `self.current_cash`.

        Args:
            stocks (pd.DataFrame):  Stocks data for the current time step.
            allocation (float):     Total capital allocation for stocks.
            sma_days (int):         SMA window.
        """

        stock_symbols = [stock.symbol for stock in self.stocks]
        sym_col = self._stocks_schema['symbol']
        inventory_stocks = stocks[stocks[sym_col].isin(stock_symbols)]
        stock_percentages = np.array([stock.percentage for stock in self.stocks])
        stock_prices = inventory_stocks[self._stocks_schema['adjClose']].values

        if sma_days:
            sma_values = inventory_stocks['sma'].values
            qty = np.where(sma_values < stock_prices, (allocation * stock_percentages) // stock_prices, 0)
        else:
            qty = (allocation * stock_percentages) // stock_prices

        self.current_cash -= np.sum(stock_prices * qty)
        self._stocks_inventory = pd.DataFrame({'symbol': stock_symbols, 'price': stock_prices, 'qty': qty})

    def _update_balance(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
        """Updates self.balance in batch in a certain period between rebalancing days"""
        stocks_date_col = self._stocks_schema['date']
        sd = self._stocks_data._data
        stocks_data = sd[(sd[stocks_date_col] >= start_date) & (sd[stocks_date_col] < end_date)]

        options_date_col = self._options_schema['date']
        od = self._options_data._data
        options_data = od[(od[options_date_col] >= start_date) & (od[options_date_col] < end_date)]

        calls_value = pd.Series(0.0, index=options_data[options_date_col].unique())
        puts_value = pd.Series(0.0, index=options_data[options_date_col].unique())

        options_contract_col = self._options_schema['contract']
        for leg in self._options_strategy.legs:
            leg_inventory = self._options_inventory[leg.name]
            if leg_inventory.empty or leg_inventory['contract'].isna().all():
                continue
            cost_field = (~leg.direction).value

            # Build per-contract info from inventory
            inv_info = pd.DataFrame({
                '_contract': leg_inventory['contract'].values,
                '_qty': self._options_inventory['totals']['qty'].values,
                '_type': leg_inventory['type'].values,
            })

            # Single merge for ALL contracts in this leg
            all_current = inv_info.merge(options_data, how='left',
                                         left_on='_contract', right_on=options_contract_col)

            sign = -1 if cost_field == Direction.BUY.value else 1
            all_current['_value'] = sign * all_current[cost_field] * all_current['_qty'] * self.shares_per_contract

            # Split into calls and puts, group by date
            calls_mask = all_current['_type'] == Type.CALL.value
            if calls_mask.any():
                calls_data = all_current.loc[calls_mask].groupby(options_date_col)['_value'].sum()
                calls_value = calls_value.add(calls_data, fill_value=0)
            puts_mask = ~calls_mask
            if puts_mask.any():
                puts_data = all_current.loc[puts_mask].groupby(options_date_col)['_value'].sum()
                puts_value = puts_value.add(puts_data, fill_value=0)

        stocks_current = self._stocks_inventory[['symbol', 'qty']].merge(stocks_data[['date', 'symbol', 'adjClose']],
                                                                         on='symbol')
        stocks_current['cost'] = stocks_current['qty'] * stocks_current['adjClose']

        add = stocks_current.pivot_table(index=stocks_date_col, columns='symbol', values='cost', aggfunc='sum')
        add = add.reindex(columns=[stock.symbol for stock in self._stocks])

        add['cash'] = self.current_cash
        add['options qty'] = self._options_inventory['totals']['qty'].sum()
        add['calls capital'] = calls_value
        add['puts capital'] = puts_value
        add['stocks qty'] = self._stocks_inventory['qty'].sum()

        for symbol, qty in zip(self._stocks_inventory['symbol'], self._stocks_inventory['qty']):
            add[symbol + ' qty'] = qty

        self._balance_parts.append(pd.DataFrame(add))

    def _execute_option_entries(self, date: pd.Timestamp, options: pd.DataFrame, options_allocation: float) -> None:
        """Enters option positions according to `self._options_strategy`.
        Calls `self._pick_entry_signals` to select from the entry signals given by the strategy.
        Updates `self._options_inventory` and `self.current_cash`.

        Args:
            date (pd.Timestamp):        Current date.
            options (pd.DataFrame):     Options data for the current time step.
            options_allocation (float): Capital amount allocated to options.
        """
        self.current_cash += options_allocation

        # Remove contracts already in inventory
        inventory_contracts = pd.concat(
            [self._options_inventory[leg.name]['contract'] for leg in self._options_strategy.legs])
        subset_options = options[~options[self._options_schema['contract']].isin(inventory_contracts)]

        entry_signals: list[pd.DataFrame] = []
        for leg in self._options_strategy.legs:
            flt = leg.entry_filter
            cost_field = leg.direction.value

            leg_entries = subset_options[flt(subset_options)]
            # Exit if no entry signals for the current leg
            if leg_entries.empty:
                return

            if leg.entry_sort:
                col, asc = leg.entry_sort
                leg_entries = leg_entries.sort_values(col, ascending=asc)

            fields = self._signal_fields(cost_field)
            leg_entries = leg_entries.reindex(columns=fields.keys())
            leg_entries.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, Signal.ENTRY)
            leg_entries['order'] = order

            # Change sign of cost for SELL orders
            if leg.direction == Direction.SELL:
                leg_entries['cost'] = -leg_entries['cost']

            leg_entries['cost'] *= self.shares_per_contract
            leg_entries.columns = pd.MultiIndex.from_product([[leg.name], leg_entries.columns])
            entry_signals.append(leg_entries.reset_index(drop=True))

        # Append the 'totals' column to entry_signals
        total_costs = sum([leg_entry.droplevel(0, axis=1)['cost'] for leg_entry in entry_signals])
        qty = options_allocation // abs(total_costs)
        totals = pd.DataFrame.from_dict({'cost': total_costs, 'qty': qty, 'date': date})
        totals.columns = pd.MultiIndex.from_product([['totals'], totals.columns])
        entry_signals.append(totals)
        entry_signals_df = pd.concat(entry_signals, axis=1)

        # Remove signals where qty == 0
        entry_signals_df = entry_signals_df[entry_signals_df['totals']['qty'] > 0]

        entries = self._pick_entry_signals(entry_signals_df)

        # Update options inventory, trade log and current cash
        entries_df = pd.DataFrame(entries) if entries.empty else pd.DataFrame([entries])
        self._options_inventory = pd.concat([self._options_inventory, entries_df], ignore_index=True)
        self._trade_log_parts.append(entries_df)
        self.current_cash -= np.sum(entries['totals']['cost'] * entries['totals']['qty'])

    def _execute_option_exits(self, date: pd.Timestamp, options: pd.DataFrame) -> None:
        """Exits option positions according to `self._options_strategy`.
        Option positions are closed whenever the strategy signals an exit, when the profit/loss thresholds
        are exceeded or whenever the contracts in `self._options_inventory` are not found in `options`.
        Updates `self._options_inventory` and `self.current_cash`.

        Args:
            date (pd.Timestamp):        Current date.
            options (pd.DataFrame):     Options data for the current time step.
        """

        strategy = self._options_strategy
        current_options_quotes = self._get_current_option_quotes(options)

        filter_masks: list[pd.Series] = []
        for i, leg in enumerate(strategy.legs):
            flt = leg.exit_filter

            # This mask is to ensure that legs with missing contracts exit.
            missing_contracts_mask = current_options_quotes[i]['cost'].isna()

            filter_masks.append(flt(current_options_quotes[i]) | missing_contracts_mask)
            fields = self._signal_fields((~leg.direction).value)
            current_options_quotes[i] = current_options_quotes[i].reindex(columns=fields.values())
            current_options_quotes[i].rename(columns=fields, inplace=True)
            current_options_quotes[i].columns = pd.MultiIndex.from_product([[leg.name],
                                                                            current_options_quotes[i].columns])

        exit_candidates = pd.concat(current_options_quotes, axis=1)

        # If a contract is missing we replace the NaN values with those of the inventory
        # except for cost, which we imput as zero.
        exit_candidates = self._impute_missing_option_values(exit_candidates)

        # Append the 'totals' column to exit_candidates
        qtys = self._options_inventory['totals']['qty']
        total_costs = sum([exit_candidates[l.name]['cost'] for l in self._options_strategy.legs])
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
        if not exits.empty:
            self._trade_log_parts.append(pd.DataFrame(exits))
        self.current_cash -= sum(total_costs)

    def _pick_entry_signals(self, entry_signals: pd.DataFrame) -> pd.Series:
        """Returns the entry signals to execute.

        Args:
            entry_signals (pd.DataFrame):   DataFrame of option entry signals chosen by the strategy.

        Returns:
            pd.Series:                      First entry signal to execute.
        """

        if not entry_signals.empty:
            # FIXME: This is a naive signal selection criterion, it simply picks the first one in `entry_singals`
            return entry_signals.iloc[0]
        else:
            return entry_signals

    def _signal_fields(self, cost_field: str) -> dict[str, str]:
        fields = {
            self._options_schema['contract']: 'contract',
            self._options_schema['underlying']: 'underlying',
            self._options_schema['expiration']: 'expiration',
            self._options_schema['type']: 'type',
            self._options_schema['strike']: 'strike',
            self._options_schema[cost_field]: 'cost',
            'order': 'order'
        }

        return fields

    def _get_current_option_quotes(self, options: pd.DataFrame) -> list[pd.DataFrame]:
        """Returns the current quotes for all the options in `self._options_inventory` as a list of DataFrames.
        It also adds a `cost` column with the cost of closing the position in each contract and an `order`
        column with the corresponding exit order type.

        Args:
            options (pd.DataFrame): Options data in the current time step.

        Returns:
            list[pd.DataFrame]:     List of DataFrames, one for each leg in `self._options_inventory`,
                                    with the exit cost for the contracts.
        """

        current_options_quotes: list[pd.DataFrame] = []
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
            leg_options['cost'] = leg_options[self._options_schema[(~leg.direction).value]]

            # Change sign of cost for SELL orders
            if ~leg.direction == Direction.SELL:
                leg_options['cost'] = -leg_options['cost']
            leg_options['cost'] *= self.shares_per_contract

            current_options_quotes.append(leg_options)

        return current_options_quotes

    def _impute_missing_option_values(self, exit_candidates: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of the inventory with the cost of all its contracts set to zero.

        Args:
            exit_candidates (pd.DataFrame): DataFrame of exit candidates with possible missing values.

        Returns:
            pd.DataFrame:                   Exit candidates with imputed values.
        """
        df = self._options_inventory.copy()
        if not df.empty:
            for leg in self._options_strategy.legs:
                df.loc[(leg.name, 'cost')] = 0

        return exit_candidates.fillna(df)

    def __repr__(self) -> str:
        return "Backtest(capital={}, allocation={}, stocks={}, strategy={})".format(
            self.current_cash, self.allocation, self._stocks, self._options_strategy)
