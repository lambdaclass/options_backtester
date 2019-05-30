from enum import Enum


class OptionContract:
    """Option contract data class"""

    Type = Enum("Type", {"CALL": "call", "PUT": "put"})
    Direction = Enum("Direction", "BUY SELL")

    # Orders:
    # BTO: Buy to Open
    # BTC: Buy to Close
    # STO: Sell to Open
    # STC: Sell to Close
    Order = Enum("Order", "BTO BTC STO STC")

    def __init__(self,
                 option_type=Type.CALL,
                 direction=Direction.BUY,
                 order=Order.BTO):
        assert isinstance(option_type, OptionContract.Type)
        assert isinstance(direction, OptionContract.Direction)
        assert isinstance(order, OptionContract.Order)

        self._store = {}
        self._store["type"] = option_type
        self._store["direction"] = direction
        self._store["order"] = order

    def __repr__(self):
        return "Option({})".format(str(self._store))
