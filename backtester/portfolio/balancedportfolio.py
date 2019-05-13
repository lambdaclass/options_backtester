from .portfolio import Portfolio


class BalancedPortfolio(Portfolio):
    """Buys and holds a basket of securities, and allocates them
    according to given weights.
    """

    def __init__(self, *args, weights={}):
        self.weights = weights
        super().__init__(*args)

    def _get_allocation(self, signal, price):
        """Allocates capital in porportion to given weight"""
        weight = self.weights.get(signal.symbol, 0)
        cash_proportion = self.initial_capital * weight
        return cash_proportion / price
