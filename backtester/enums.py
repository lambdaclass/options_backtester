# Enums
from collections import namedtuple
from enum import Enum

# Stocks
Stock = namedtuple('Stock', 'symbol percentage')


# Options
class Type(Enum):
    CALL = 'call'
    PUT = 'put'

    def __invert__(self):
        flip = Type.PUT if self == Type.CALL else Type.CALL
        return flip


class Direction(Enum):
    BUY = 'ask'  # Schema field for BUY price
    SELL = 'bid'  # Schema field for SELL price

    def __invert__(self):
        flip = Direction.SELL if self == Direction.BUY else Direction.BUY
        return flip


# Signals
Signal = Enum("Signal", "ENTRY EXIT")


class Order(Enum):
    BTO = 'BTO'  # Buy to Open
    BTC = 'BTC'  # Buy to Close
    STO = 'STO'  # Sell to Open
    STC = 'STC'  # Sell to Close

    def __invert__(self):
        if self == Order.BTO:
            return Order.STC
        elif self == Order.BTC:
            return Order.STO
        elif self == Order.STO:
            return Order.BTC
        elif self == Order.STC:
            return Order.BTO


def get_order(direction, signal):
    """Returns Order type given direction (BUY | SELL) and
    signal (ENTRY | EXIT).
    """
    if direction == Direction.BUY:
        return Order.BTO if signal == Signal.ENTRY else Order.STC
    else:
        return Order.STO if signal == Signal.ENTRY else Order.BTC
