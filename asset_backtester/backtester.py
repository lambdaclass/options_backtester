import pandas as pd
import numpy as np
import pyprind
from strategy.strategy import Strategy


class Backtest:
    """Processes signals from the Strategy object"""
    def __init__(self, schema):
        self.schema = schema
        self._strategy = None
        self._data = None

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
        self._data = data

    def run(self, initial_capital=1_000_000, periods='1'):
        assert self._data is not None
        assert self._strategy is not None

        self.current_capital = 0
        self.current_cash = initial_capital

        self.inventory = pd.DataFrame(columns=['symbol', 'cost', 'qty'])
        self.balance = pd.DataFrame()

        data_iterator = self._data.iter_dates()
        monthly_iterator = self._data.iter_months()
    
        rebalancing_days = pd.date_range(self._data['date'].iloc[0], self._data['date'].iloc[-1], freq=periods + 'BMS').to_pydatetime()
       
        bar = pyprind.ProgBar(data_iterator.ngroups, bar_char='█')

        self.balance = pd.DataFrame(
            {
                'capital': self.current_cash,
                'cash': self.current_cash
            },
            index=[self.data.start_date - pd.Timedelta(1, unit='day')])

        for date, stocks in data_iterator:
            if date == self._data._data['date'][0]:
                self.rebalance_portfolio(stocks)
            self._update_balance(date, stocks)
           
            if date in rebalancing_days:
                self.rebalance_portfolio(stocks)
            bar.update()

        self.balance['% change'] = self.balance['capital'].pct_change()
        self.balance['accumulated return'] = (
            1.0 + self.balance['% change']).cumprod()

        return self.balance

    def rebalance_portfolio(self, stocks):
        money_total = self.current_cash + self.current_capital
        for asset in self._strategy.assets:
            stock = stocks[stocks['symbol'] == asset.symbol]
            stock_price = stock[self.schema['Adj Close']].values[0]
            qty = (money_total * asset.percentage) // stock_price
            inventory_entry = self.inventory[self.inventory['symbol'] ==
                                             asset.symbol]
            self.inventory.drop(inventory_entry.index, inplace=True)
            update = pd.Series([asset.symbol, stock_price, qty])
            update.index = self.inventory.columns
            self.inventory = self.inventory.append(update, ignore_index=True)

        # Update current cash
        invested_capital = sum(self.inventory['cost'] * self.inventory['qty'])
        self.current_cash = money_total - invested_capital

    def _update_balance(self, date, stocks):
        """Updates positions and calculates statistics for the current date.

        Args:
            date (pd.Timestamp):    Current date.
            stocks (pd.DataFrame): DataFrame of (daily/monthly) stocks.
        """

        costs = []

        for asset in self._strategy.assets:
            asset_entry = stocks[stocks['symbol'] == asset.symbol]
            inventory_asset_entry = self.inventory[self.inventory['symbol'] ==
                                                   asset.symbol]
            cost = asset_entry[self.schema['Adj Close']].values[0]
            qty = inventory_asset_entry['qty'].values[0]
            costs.append(cost * qty)

        total_value = sum(costs)
        self.current_capital = total_value
        money_total = total_value + self.current_cash

        row = pd.Series(
            {
                'total_value': total_value,
                'cash': self.current_cash,
                'capital': money_total,
            },
            name=date)
        self.balance = self.balance.append(row)
