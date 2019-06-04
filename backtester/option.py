# Option Enum types
from enum import Enum

Type = Enum("Type", {"CALL": "call", "PUT": "put"})
Direction = Enum("Direction", {"BUY": "ask", "SELL": "bid"})
