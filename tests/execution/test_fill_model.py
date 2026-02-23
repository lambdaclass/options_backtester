"""Tests for fill models."""

import pandas as pd

from options_backtester.core.types import Direction
from options_backtester.execution.fill_model import (
    MarketAtBidAsk, MidPrice, VolumeAwareFill,
)


def _make_row(bid: float = 1.00, ask: float = 1.10, volume: int = 100) -> pd.Series:
    return pd.Series({"bid": bid, "ask": ask, "volume": volume})


class TestMarketAtBidAsk:
    def test_buy_fills_at_ask(self):
        m = MarketAtBidAsk()
        assert m.get_fill_price(_make_row(), Direction.BUY) == 1.10

    def test_sell_fills_at_bid(self):
        m = MarketAtBidAsk()
        assert m.get_fill_price(_make_row(), Direction.SELL) == 1.00


class TestMidPrice:
    def test_mid(self):
        m = MidPrice()
        assert m.get_fill_price(_make_row(), Direction.BUY) == 1.05

    def test_mid_sell(self):
        m = MidPrice()
        assert m.get_fill_price(_make_row(), Direction.SELL) == 1.05


class TestVolumeAwareFill:
    def test_high_volume_fills_at_target(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        assert m.get_fill_price(_make_row(volume=200), Direction.BUY) == 1.10

    def test_zero_volume_fills_at_mid(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        assert m.get_fill_price(_make_row(volume=0), Direction.BUY) == 1.05

    def test_half_volume_interpolates(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        price = m.get_fill_price(_make_row(volume=50), Direction.BUY)
        # mid=1.05, target=1.10, ratio=0.5 -> 1.05 + 0.5*0.05 = 1.075
        assert abs(price - 1.075) < 1e-10
