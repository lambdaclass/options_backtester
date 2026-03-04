"""Fill models — determine the execution price for trades."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from options_portfolio_backtester.core.types import Direction
from options_portfolio_backtester.execution._rust_bridge import rust_fill_price


class FillModel(ABC):
    """Determines the price at which a trade is filled."""

    @abstractmethod
    def get_fill_price(self, row: pd.Series, direction: Direction) -> float:
        """Return the execution price for a given option quote row and direction."""
        ...


class MarketAtBidAsk(FillModel):
    """Fill at the bid (sell) or ask (buy) — matches original behavior."""

    def get_fill_price(self, row: pd.Series, direction: Direction) -> float:
        return float(row[direction.price_column])

    def to_rust_config(self) -> dict:
        return {"type": "MarketAtBidAsk"}


class MidPrice(FillModel):
    """Fill at the midpoint of bid and ask."""

    def get_fill_price(self, row: pd.Series, direction: Direction) -> float:
        bid = float(row["bid"])
        ask = float(row["ask"])
        return (bid + ask) / 2.0

    def to_rust_config(self) -> dict:
        return {"type": "MidPrice"}


class VolumeAwareFill(FillModel):
    """Fill price that adjusts for volume impact.

    For low-volume contracts, the fill is pushed toward the less favorable
    price. Above `full_volume_threshold`, the fill is at bid/ask.
    """

    def __init__(self, full_volume_threshold: int = 100) -> None:
        self.full_volume_threshold = full_volume_threshold

    def get_fill_price(self, row: pd.Series, direction: Direction) -> float:
        bid = float(row["bid"])
        ask = float(row["ask"])
        is_buy = direction == Direction.BUY
        vol_raw = row.get("volume")
        volume = None if vol_raw is None or (isinstance(vol_raw, float) and vol_raw != vol_raw) else float(vol_raw)
        return rust_fill_price("VolumeAware", self.full_volume_threshold, bid, ask, volume, is_buy)

    def to_rust_config(self) -> dict:
        return {"type": "VolumeAware", "full_volume_threshold": self.full_volume_threshold}
