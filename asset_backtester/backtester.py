import pandas as pd
import pyprind
import numpy as np
from .portfolio import Portfolio


class Backtest:
    def __init__(self, schema):
        self.schema = schema
        self._portfolio = None
        self._data = None

    @property
    def portfolio(self):
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio):
        assert isinstance(portfolio, Portfolio)
        self._portfolio = portfolio

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def run(self, initial_capital=1_000_000, periods=1, sma_days=None):
        """Runs a backtest and returns a dataframe with the daily balance"""
        assert self._data is not None
        assert self._portfolio is not None

        self.current_capital = 0
        self.current_cash = initial_capital
        self.inventory = pd.DataFrame(columns=['symbol', 'cost', 'qty'])
        self.balance = pd.DataFrame()
        if sma_days:
            self._data.sma(sma_days)

        data_iterator = self._data.iter_dates()

        first_day = self._data['date'].min()
        last_day = self._data['date'].max()
        rebalancing_days = pd.date_range(first_day, last_day, freq=str(periods) +
                                         'BMS').to_pydatetime() if periods is not None else []

        bar = pyprind.ProgBar(data_iterator.ngroups, bar_char='█')

        self.balance = pd.DataFrame({
            'capital': self.current_cash,
            'cash': self.current_cash
        },
                                    index=[self._data.start_date - pd.Timedelta(1, unit='day')])

        for date, data in data_iterator:

            if date == first_day:
                self._rebalance_portfolio(data, sma_days)
            self._update_balance(date, data)
            if date in rebalancing_days:
                self._rebalance_portfolio(data, sma_days)

            bar.update()

        self.balance['% change'] = self.balance['capital'].pct_change()
        self.balance['accumulated return'] = (1.0 + self.balance['% change']).cumprod()

        return self.balance

    def _rebalance_portfolio(self, data, sma_days):
        """Rebalances the portfolio so that the total money is allocated according to the given percentages"""
        money_total = self.current_cash + self.current_capital
        for asset in self._portfolio.assets:
            query = '{} == "{}"'.format(self.schema['symbol'], asset.symbol)
            asset_current = data.query(query)

            asset_price = asset_current[self.schema['adjClose']].values[0]

            if sma_days is not None:
                if asset_current['sma'].values[0] < asset_price:
                    qty = (money_total * asset.percentage) // asset_price
                else:
                    qty = 0

            else:
                qty = (money_total * asset.percentage) // asset_price

            inventory_entry = self.inventory.query(query)
            self.inventory.drop(inventory_entry.index, inplace=True)
            updated_asset = pd.Series([asset.symbol, asset_price, qty])
            updated_asset.index = self.inventory.columns
            self.inventory = self.inventory.append(updated_asset, ignore_index=True)
        # Update current cash
        invested_capital = sum(self.inventory['cost'] * self.inventory['qty'])
        self.current_cash = money_total - invested_capital

    def _update_balance(self, date, data):
        """Updates self.balance for the given date"""
        costs = []

        for asset in self._portfolio.assets:
            query = '{} == "{}"'.format(self.schema['symbol'], asset.symbol)
            asset_current = data.query(query)
            inventory_asset = self.inventory.query(query)

            cost = asset_current[self.schema['adjClose']].values[0]
            qty = inventory_asset['qty'].values[0]
            costs.append(cost * qty)

        total_value = sum(costs)
        self.current_capital = total_value
        money_total = total_value + self.current_cash

        row = pd.Series({
            'total_value': total_value,
            'cash': self.current_cash,
            'capital': money_total,
        }, name=date)
        self.balance = self.balance.append(row)
