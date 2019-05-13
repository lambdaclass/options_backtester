from abc import ABCMeta, abstractmethod
import pandas as pd


class Portfolio(metaclass=ABCMeta):
    """Processes signals from the Strategy object"""

    @abstractmethod
    def __init__(self, data_handler, events, capital=1000000):
        self.data_handler = data_handler
        self.events = events
        self.initial_capital = capital
        self.current_position = {"Cash": self.initial_capital}
        self.all_positions = {}
        self.current_balance = {"Cash": self.initial_capital}
        self.all_balances = {}

    @abstractmethod
    def _get_allocation(self, strength, price):
        """Calculates symbol allocation"""
        raise NotImplementedError("Portfolio must implement _get_allocation()")

    def update_signal(self, signal):
        """Processes signal event and updates the current position"""
        date = self.data_handler.current_date
        if date not in self.all_positions:
            self.all_positions[date] = self.current_position.copy()
        self.current_position = self.all_positions[date]

        (price, direction) = self._get_price(signal)
        qty = self._get_allocation(signal, price)
        (current_amount, current_open_price) = self.current_position.get(
            signal.symbol, (0, 0))
        new_open_price = (current_open_price * current_amount +
                          direction * price * qty) / (current_amount + qty)
        self.current_position[signal.symbol] = (
            current_amount + direction * qty, new_open_price)
        self.current_position["Cash"] -= direction * price * qty

    def update_timeindex(self, event):
        """Calculates new balance for the current timeindex.
        Appends current position to all_positions list."""
        date = self.data_handler.current_date
        self.all_balances[date] = self.current_balance.copy()
        self.current_balance = self.all_balances[date]
        self.current_balance["Total Exposure"] = 0

        for symbol, values in self.current_position.items():
            if symbol == "Cash":
                self.current_balance["Cash"] = values
                continue

            (amount, open_price) = values
            current_bar = self.data_handler.get_latest_bars(symbol)
            if amount < 0:
                price = current_bar["ask"]
            else:
                price = current_bar["bid"]
            market_value = amount * price
            self.current_balance[symbol + " Amount"] = amount
            self.current_balance[symbol + " Open"] = open_price
            self.current_balance[symbol + " Exposure"] = market_value
            self.current_balance["Total Exposure"] += market_value

        self.all_positions[date] = self.current_position
        self.all_balances[date] = self.current_balance

    def _get_price(self, signal):
        """Returns price and direction for given symbol.
        Ask price if signal.type == BUY, bid price if signal.type == SELL.
        Also returns 1 or -1 for types BUY, SELL respectively"""
        current_bar = self.data_handler.get_latest_bars(signal.symbol)
        if signal.direction == "BUY":
            direction = 1
            price = current_bar["ask"]
        else:
            direction = -1
            price = current_bar["bid"]
        return (price, direction)

    def create_report(self):
        """Creates a pandas DataFrame from all_balances."""
        curve = pd.DataFrame(self.all_balances)
        curve = curve.transpose()
        curve["Total Portfolio"] = curve["Total Exposure"] + curve["Cash"]
        curve["Interval Change"] = curve["Total Portfolio"].pct_change()
        curve["% Price"] = (1.0 + curve["Interval Change"]).cumprod() - 1
        return curve
