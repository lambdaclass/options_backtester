class Asset:
    """Asset data class"""
    def __init__(self, symbol, percentage):
        self.symbol = symbol
        self.percentage = percentage

    def __repr__(self):
        return "Asset(symbol={}, percentage={})".format(self.symbol, self.percentage)
