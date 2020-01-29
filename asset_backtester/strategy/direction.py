from enum import Enum


class Direction(Enum):
    BUY = 'ask'  # Schema field for BUY price
    SELL = 'bid'  # Schema field for SELL price

    def __invert__(self):
        flip = Direction.SELL if self == Direction.BUY else Direction.BUY
        return flip