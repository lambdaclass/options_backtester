import pandas as pd

from .strategy import Strategy
from .strategy.signal import Order
from .datahandler import HistoricalOptionsData


class Backtest:
    """Processes signals from the Strategy object"""

    def __init__(self, capital=1_000_000, shares_per_contract=100):
        self.capital = capital
        self.shares_per_contract = shares_per_contract
        self._strategy = None
        self._data = None
        self._inventory = set()

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
                self._data):
            self._execute_exit(date, exit_signals)
            self._execute_entry(date, entry_signals)

        return self.trade_log

    def _execute_entry(self, date, orders, entry_signals):
        """Executes entry orders and updates `self.inventory` and `self.trade_log`"""

        orders = self._process_entry_signals(entry_signals)

        for leg, (idx, qty) in orders.items():
            row = entry_signals[leg].loc[idx, :]
            contract = row["contract"]
            order = row["order"]
            price = row["price"]
            expiration = row["expiration"]
            cost = price * qty * self.shares_per_contract
            cost *= -1 if order == Order.STO.name else 1
            if self.capital >= cost:
                self.capital -= cost
                self._inventory.add((contract, leg, qty, expiration))
                self.strategy.register_entry(contract, price)
                self._update_trade_log(date, contract, order, qty, -cost)

    def _process_entry_signals(self, entry_signals):
        """Returns a dictionary containing the orders to execute."""
        # Pass `qty` of contracts to buy/sell to `Backtest.__init__`

        orders = {}

        if not entry_signals.empty:
            for leg in entry_signals.legs:
                leg_signals = entry_signals[leg]
                # Filter out zero priced options
                leg_signals = leg_signals.query("price > 0.0")
                if leg_signals.empty:
                    return {}
                if (leg_signals["order"] == Order.BTO.name).any():
                    orders[leg] = (leg_signals["price"].idxmin(), 1)
                else:
                    orders[leg] = (leg_signals["price"].idxmax(), 1)
        return orders

    def _execute_exit(self, date, exit_signals):
        """Executes exits and updates `self.inventory` and `self.trade_log`"""
        remove_set = set()

        for contract, leg, qty, expiration in self._inventory:
            if contract in exit_signals[leg]["contract"].values:
                row = exit_signals[leg].query("contract == @contract")
                price = row["price"].values[0]
                order = row["order"].values[0]
                profit = price * qty * self.shares_per_contract
                profit *= 1 if order == Order.STC.name else -1
                self.capital += profit
                self._update_trade_log(date, contract, order, qty, profit)
                remove_set.add((contract, leg, qty, expiration))
            elif expiration <= date:
                remove_set.add((contract, leg, qty, expiration))

        self._inventory.difference_update(remove_set)

    def _update_trade_log(self, date, contract, order, qty, profit):
        """Adds entry for the given order to `self.trade_log`."""
        self.trade_log.loc[len(self.trade_log)] = [
            date, contract, order, qty, profit, self.capital
        ]

    def __repr__(self):
        return "Backtest(capital={}, strategy={})".format(
            self.capital, self._strategy)
