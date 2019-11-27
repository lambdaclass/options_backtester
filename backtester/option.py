# Option Enum types
from enum import Enum


class Type(Enum):
    CALL = 'call'
    PUT = 'put'

    def __invert__(self):
        flip = Type.PUT if self == Type.CALL else Type.CALL
        return flip


class Direction(Enum):
    BUY = 'ask'
    SELL = 'bid'

    def __invert__(self):
        flip = Direction.SELL if self == Direction.BUY else Direction.SELL
        return flip
