from .direction import Direction
from datahandler.schema import Schema


class Asset:
    """Strategy Leg data class"""
    def __init__(self, symbol, percentage, direction=Direction.BUY):
        assert isinstance(direction, Direction)

        self.symbol = symbol
        self.percentage = percentage
        self.direction = direction

    def __repr__(self):
        return "Asset(symbol={}, percentage={}, direction={})".format(
            self.symbol, self.percentage, self.direction)
