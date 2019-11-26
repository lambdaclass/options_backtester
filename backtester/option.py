# Option Enum types
from enum import Enum

Type = Enum("Type", {"CALL": "call", "PUT": "put"})
Direction = Enum("Direction", {"BUY": "ask", "SELL": "bid"})


def flip(enum_type):
    """Returns the other value of `Type` or `Direction`"""
    assert isinstance(enum_type, Type) or isinstance(enum_type, Direction)

    for e in type(enum_type):
        if e != enum_type:
            return e
