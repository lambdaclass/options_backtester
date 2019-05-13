class Event():
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with
    corresponding bars.
    """

    def __init__(self):
        self.type = "MARKET"


class SignalEvent(Event):
    """
    Handles the event of receiving a signal form the Strategy
    object.
    Portfolio object processes buy/sell orders.
    """

    def __init__(self, symbol, direction, strength):
        """symbol: ticker symbol
        direction: BUY | SELL
        strength: (%Win chance, Win/Loss ratio)"""
        self.type = "SIGNAL"
        self.symbol = symbol
        self.direction = direction
        self.strength = strength
