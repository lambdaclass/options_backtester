from .strategy import Strategy
from ..event import SignalEvent


class Benchmark(Strategy):
    """Simple buy and hold SPX strategy"""

    def __init__(self, data_handler, events):
        self.data_handler = data_handler
        self.events = events
        self._bought = False

    def generate_signals(self, event):
        if not self._bought:
            buy_signal = SignalEvent(
                symbol="SPX", direction="BUY", strength=(1.0, 100))
            self.events.put(buy_signal)
            self._bought = True
