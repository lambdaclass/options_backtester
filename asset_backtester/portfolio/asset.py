class Asset:
    """Asset data class"""
    def __init__(self, symbol, percentage):
        self.symbol = symbol
        self.percentage = percentage

    def __repr__(self):
        return "Asset(symbol={}, percentage={}, direction={})".format(self.symbol, self.percentage, self.direction)
