import math
from .portfolio import Portfolio


class SimplePortfolio(Portfolio):
    """Allocates all capital to the first signal processed"""

    def __init__(self, *args):
        super().__init__(*args)

    def _get_allocation(self, strength, price):
        """Allocates all capital to the given signal"""
        return math.floor(self.current_position["Cash"] / price)
