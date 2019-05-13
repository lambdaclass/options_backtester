"""Event based backtester"""

from queue import Queue
from .datahandler import BalancedDataHandler
from .strategy import Balanced
from .portfolio import BalancedPortfolio


def run(data_path,
        data_handler=BalancedDataHandler,
        port_class=BalancedPortfolio,
        strat_class=Balanced,
        **strat_args):
    events = Queue()
    bars = data_handler(data_path, events)

    weights = {
        "VOO": 0.3,
        "GLD": 0.1,
        "VNQ": 0.05,
        "VNQI": 0.05,
        "TLT": 0.2,
        "TIP": 0.1,
        "BNDX": 0.1,
        "RJI": 0.1
    }
    port = port_class(bars, events, weights=weights)
    strat = strat_class(bars, events, **strat_args)

    while True:
        bars.update_bars()
        if not bars.continue_backtest:
            break

        while True:
            if events.empty():
                break
            event = events.get()
            if event.type == "MARKET":
                strat.generate_signals(event)
                port.update_timeindex(event)
            elif event.type == "SIGNAL":
                port.update_signal(event)

    return port
