import math
from .portfolio import Portfolio


class KellyPortfolio(Portfolio):
    """Allocates signals using Kelly's criterion"""

    def __init__(self, *args):
        super().__init__(*args)

    def _get_allocation(self, strength, price):
        """Calculates allocation using Kelly's criterion"""
        (win_percent, win_loss_ratio) = strength
        kelly = max(0, win_percent - (1 - win_percent) / win_loss_ratio)
        total_allocation = self.current_position["Cash"] * kelly
        return math.floor(total_allocation / price)
