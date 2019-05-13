from .strategy import Strategy
from ..event import SignalEvent


class Balanced(Strategy):
    """Balanced portfolio strategy.
    Inspired by Ray Dalio's all weather portfolio.
    """

    def __init__(self,
                 data_handler,
                 events,
                 symbols=[
                     "VOO", "GLD", "VNQ", "VNQI", "TLT", "TIP", "BNDX", "EEM",
                     "RJI"
                 ]):
        self.data_handler = data_handler
        self.symbols = symbols
        self.events = events
        self._bought = False

    def generate_signals(self, event):
        if not self._bought:
            for symbol in self.symbols:
                buy_signal = SignalEvent(
                    symbol=symbol, direction="BUY", strength=(1.0, 100))
                self.events.put(buy_signal)

            self._bought = True
