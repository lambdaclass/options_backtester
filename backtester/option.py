# Option Enum types
from enum import Enum

Type = Enum("Type", {"CALL": "call", "PUT": "put"})
Direction = Enum("Direction", {"BUY": "ask", "SELL": "bid"})

# Orders:
# BTO: Buy to Open
# BTC: Buy to Close
# STO: Sell to Open
# STC: Sell to Close
Order = Enum("Order", "BTO BTC STO STC")
