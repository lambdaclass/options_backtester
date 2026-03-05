"""Deep execution model tests — fill models, cost models, signal selectors, sizers.

Tests edge cases, boundary conditions, and composition of execution components.
"""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.execution.cost_model import (
    NoCosts,
    PerContractCommission,
    TieredCommission,
    SpreadSlippage,
)
from options_portfolio_backtester.execution.fill_model import (
    MarketAtBidAsk,
    MidPrice,
    VolumeAwareFill,
)
from options_portfolio_backtester.execution.signal_selector import (
    FirstMatch,
    NearestDelta,
    MaxOpenInterest,
)
from options_portfolio_backtester.execution.sizer import (
    CapitalBased,
    FixedQuantity,
    FixedDollar,
    PercentOfPortfolio,
)
from options_portfolio_backtester.core.types import Direction


# ---------------------------------------------------------------------------
# Cost Models — deep tests
# ---------------------------------------------------------------------------


class TestNoCosts:
    def test_option_cost_always_zero(self):
        m = NoCosts()
        assert m.option_cost(100.0, 50, 100) == 0.0
        assert m.option_cost(0.0, 0, 100) == 0.0

    def test_stock_cost_always_zero(self):
        m = NoCosts()
        assert m.stock_cost(50.0, 1000.0) == 0.0

    def test_rust_config(self):
        assert NoCosts().to_rust_config() == {"type": "NoCosts"}


class TestPerContractCommission:
    def test_basic_cost(self):
        m = PerContractCommission(rate=0.65)
        assert m.option_cost(10.0, 10, 100) == 6.5

    def test_negative_qty_uses_abs(self):
        m = PerContractCommission(rate=1.0)
        assert m.option_cost(5.0, -20, 100) == 20.0

    def test_zero_qty(self):
        m = PerContractCommission(rate=0.65)
        assert m.option_cost(10.0, 0, 100) == 0.0

    def test_stock_cost_per_share(self):
        m = PerContractCommission(rate=0.65, stock_rate=0.01)
        assert m.stock_cost(50.0, 100) == 1.0

    def test_stock_cost_negative_qty(self):
        m = PerContractCommission(stock_rate=0.005)
        assert m.stock_cost(50.0, -200) == 1.0

    def test_rust_config_roundtrip(self):
        m = PerContractCommission(rate=0.65, stock_rate=0.005)
        cfg = m.to_rust_config()
        assert cfg["type"] == "PerContract"
        assert cfg["rate"] == 0.65
        assert cfg["stock_rate"] == 0.005


class TestTieredCommission:
    def test_default_tiers(self):
        m = TieredCommission()
        # First 10000 at $0.65
        assert m.option_cost(1.0, 100, 100) == 100 * 0.65

    def test_tier_boundary(self):
        """Exactly at first tier boundary."""
        m = TieredCommission()
        cost = m.option_cost(1.0, 10_000, 100)
        assert cost == 10_000 * 0.65

    def test_crosses_first_tier(self):
        """15000 contracts: 10000 at $0.65, 5000 at $0.50."""
        m = TieredCommission()
        cost = m.option_cost(1.0, 15_000, 100)
        expected = 10_000 * 0.65 + 5_000 * 0.50
        assert abs(cost - expected) < 0.01

    def test_crosses_all_tiers(self):
        """200000 contracts: 10000*0.65 + 40000*0.50 + 50000*0.25 + 100000*0.25."""
        m = TieredCommission()
        cost = m.option_cost(1.0, 200_000, 100)
        expected = 10_000 * 0.65 + 40_000 * 0.50 + 50_000 * 0.25 + 100_000 * 0.25
        assert abs(cost - expected) < 0.01

    def test_custom_tiers(self):
        m = TieredCommission(tiers=[(5, 1.0), (10, 0.5)])
        # 7 contracts: 5*1.0 + 2*0.5
        cost = m.option_cost(1.0, 7, 100)
        assert abs(cost - 6.0) < 0.01

    def test_negative_qty(self):
        m = TieredCommission()
        cost = m.option_cost(1.0, -100, 100)
        assert cost == 100 * 0.65

    def test_zero_qty(self):
        m = TieredCommission()
        assert m.option_cost(1.0, 0, 100) == 0.0

    def test_rust_config(self):
        m = TieredCommission()
        cfg = m.to_rust_config()
        assert cfg["type"] == "Tiered"
        assert len(cfg["tiers"]) == 3


class TestSpreadSlippage:
    def test_option_cost_is_zero(self):
        m = SpreadSlippage(pct=0.5)
        assert m.option_cost(1.0, 10, 100) == 0.0

    def test_slippage_computation(self):
        m = SpreadSlippage(pct=0.5)
        # spread = 1.0, qty=10, spc=100
        slippage = m.slippage(bid=9.0, ask=10.0, quantity=10, shares_per_contract=100)
        assert slippage == 0.5 * 1.0 * 10 * 100

    def test_slippage_zero_spread(self):
        m = SpreadSlippage(pct=0.5)
        assert m.slippage(10.0, 10.0, 10, 100) == 0.0

    def test_slippage_full_pct(self):
        m = SpreadSlippage(pct=1.0)
        slippage = m.slippage(9.0, 10.0, 1, 100)
        assert slippage == 100.0

    def test_pct_bounds(self):
        with pytest.raises(AssertionError):
            SpreadSlippage(pct=-0.1)
        with pytest.raises(AssertionError):
            SpreadSlippage(pct=1.1)


# ---------------------------------------------------------------------------
# Fill Models — deep tests
# ---------------------------------------------------------------------------


def _make_option_row(bid=9.0, ask=10.0, volume=100):
    return pd.Series({"bid": bid, "ask": ask, "volume": volume})


class TestMarketAtBidAsk:
    def test_buy_fills_at_ask(self):
        m = MarketAtBidAsk()
        assert m.get_fill_price(_make_option_row(bid=9, ask=10), Direction.BUY) == 10.0

    def test_sell_fills_at_bid(self):
        m = MarketAtBidAsk()
        assert m.get_fill_price(_make_option_row(bid=9, ask=10), Direction.SELL) == 9.0

    def test_zero_spread(self):
        m = MarketAtBidAsk()
        assert m.get_fill_price(_make_option_row(bid=10, ask=10), Direction.BUY) == 10.0
        assert m.get_fill_price(_make_option_row(bid=10, ask=10), Direction.SELL) == 10.0


class TestMidPrice:
    def test_midpoint(self):
        m = MidPrice()
        assert m.get_fill_price(_make_option_row(bid=9, ask=11), Direction.BUY) == 10.0

    def test_same_bid_ask(self):
        m = MidPrice()
        assert m.get_fill_price(_make_option_row(bid=10, ask=10), Direction.BUY) == 10.0

    def test_direction_doesnt_matter(self):
        m = MidPrice()
        row = _make_option_row(bid=8, ask=12)
        assert m.get_fill_price(row, Direction.BUY) == m.get_fill_price(row, Direction.SELL)

    def test_wide_spread(self):
        m = MidPrice()
        assert m.get_fill_price(_make_option_row(bid=1, ask=100), Direction.BUY) == 50.5


class TestVolumeAwareFill:
    def test_high_volume_fills_at_target(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        row = _make_option_row(bid=9, ask=10, volume=100)
        assert m.get_fill_price(row, Direction.BUY) == 10.0
        assert m.get_fill_price(row, Direction.SELL) == 9.0

    def test_zero_volume_fills_at_mid(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        row = _make_option_row(bid=9, ask=11, volume=0)
        assert m.get_fill_price(row, Direction.BUY) == 10.0
        assert m.get_fill_price(row, Direction.SELL) == 10.0

    def test_half_volume_interpolates(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        row = _make_option_row(bid=8, ask=12, volume=50)
        # mid=10, target_buy=12, ratio=0.5 → 10 + 0.5*(12-10) = 11
        assert m.get_fill_price(row, Direction.BUY) == 11.0
        # mid=10, target_sell=8, ratio=0.5 → 10 + 0.5*(8-10) = 9
        assert m.get_fill_price(row, Direction.SELL) == 9.0

    def test_above_threshold_same_as_market(self):
        m = VolumeAwareFill(full_volume_threshold=100)
        row = _make_option_row(bid=9, ask=10, volume=500)
        assert m.get_fill_price(row, Direction.BUY) == 10.0

    def test_rust_config(self):
        m = VolumeAwareFill(full_volume_threshold=200)
        cfg = m.to_rust_config()
        assert cfg["type"] == "VolumeAware"
        assert cfg["full_volume_threshold"] == 200


# ---------------------------------------------------------------------------
# Signal Selectors — deep tests
# ---------------------------------------------------------------------------


def _make_candidates(n=5, with_delta=True, with_oi=True):
    data = {
        "contract": [f"SPX{i}" for i in range(n)],
        "strike": [100 + i * 5 for i in range(n)],
        "bid": [1.0 + i * 0.1 for i in range(n)],
        "ask": [1.5 + i * 0.1 for i in range(n)],
    }
    if with_delta:
        data["delta"] = [-0.05 - i * 0.05 for i in range(n)]  # -0.05, -0.10, ...
    if with_oi:
        data["openinterest"] = [100 * (i + 1) for i in range(n)]
    return pd.DataFrame(data)


class TestFirstMatch:
    def test_selects_first_row(self):
        s = FirstMatch()
        df = _make_candidates()
        selected = s.select(df)
        assert selected["contract"] == "SPX0"

    def test_single_row(self):
        s = FirstMatch()
        df = _make_candidates(n=1)
        selected = s.select(df)
        assert selected["contract"] == "SPX0"


class TestNearestDelta:
    def test_selects_closest_delta(self):
        s = NearestDelta(target_delta=-0.15)
        df = _make_candidates()
        # deltas: -0.05, -0.10, -0.15, -0.20, -0.25
        selected = s.select(df)
        assert selected["contract"] == "SPX2"  # delta=-0.15

    def test_boundary_delta(self):
        s = NearestDelta(target_delta=-0.075)
        df = _make_candidates()
        # Closest to -0.075 is -0.05 (diff=0.025) or -0.10 (diff=0.025)
        selected = s.select(df)
        assert selected["contract"] in {"SPX0", "SPX1"}

    def test_missing_delta_column_fallback(self):
        s = NearestDelta(target_delta=-0.30)
        df = _make_candidates(with_delta=False)
        selected = s.select(df)
        # Falls back to iloc[0]
        assert selected["contract"] == "SPX0"

    def test_column_requirements(self):
        s = NearestDelta(delta_column="my_delta")
        assert "my_delta" in s.column_requirements

    def test_rust_config(self):
        s = NearestDelta(target_delta=-0.25, delta_column="delta")
        cfg = s.to_rust_config()
        assert cfg["target"] == -0.25


class TestMaxOpenInterest:
    def test_selects_highest_oi(self):
        s = MaxOpenInterest()
        df = _make_candidates()
        selected = s.select(df)
        assert selected["contract"] == "SPX4"  # oi=500

    def test_missing_oi_column_fallback(self):
        s = MaxOpenInterest()
        df = _make_candidates(with_oi=False)
        selected = s.select(df)
        assert selected["contract"] == "SPX0"  # fallback

    def test_custom_oi_column(self):
        s = MaxOpenInterest(oi_column="my_oi")
        df = _make_candidates(with_oi=False)
        df["my_oi"] = [10, 50, 30, 20, 40]
        selected = s.select(df)
        assert selected["contract"] == "SPX1"  # oi=50


# ---------------------------------------------------------------------------
# Position Sizers — deep tests
# ---------------------------------------------------------------------------


class TestCapitalBasedSizer:
    def test_basic_sizing(self):
        s = CapitalBased()
        assert s.size(cost_per_contract=100, available_capital=1000, total_capital=10000) == 10

    def test_fractional_truncated(self):
        s = CapitalBased()
        assert s.size(150, 1000, 10000) == 6  # 1000/150 = 6.66 → 6

    def test_zero_cost(self):
        s = CapitalBased()
        assert s.size(0, 1000, 10000) == 0

    def test_cost_exceeds_capital(self):
        s = CapitalBased()
        assert s.size(2000, 1000, 10000) == 0

    def test_negative_cost_uses_abs(self):
        s = CapitalBased()
        assert s.size(-100, 1000, 10000) == 10


class TestFixedQuantitySizer:
    def test_within_budget(self):
        s = FixedQuantity(quantity=5)
        assert s.size(100, 1000, 10000) == 5

    def test_exceeds_budget_scales_down(self):
        s = FixedQuantity(quantity=20)
        assert s.size(100, 1000, 10000) == 10  # 1000/100 = 10

    def test_zero_cost_returns_fixed_qty(self):
        """Zero cost doesn't trigger the budget check, returns fixed qty."""
        s = FixedQuantity(quantity=5)
        assert s.size(0, 1000, 10000) == 5

    def test_one_contract(self):
        s = FixedQuantity(quantity=1)
        assert s.size(100, 1000, 10000) == 1


class TestFixedDollarSizer:
    def test_within_budget(self):
        s = FixedDollar(amount=500)
        assert s.size(100, 1000, 10000) == 5

    def test_amount_exceeds_available(self):
        s = FixedDollar(amount=2000)
        assert s.size(100, 500, 10000) == 5  # uses min(2000, 500)

    def test_zero_cost(self):
        s = FixedDollar(amount=500)
        assert s.size(0, 1000, 10000) == 0


class TestPercentOfPortfolioSizer:
    def test_basic(self):
        s = PercentOfPortfolio(pct=0.01)
        # 0.01 * 10000 = 100; 100/50 = 2
        assert s.size(50, 1000, 10000) == 2

    def test_pct_exceeds_available(self):
        s = PercentOfPortfolio(pct=0.5)
        # 0.5 * 10000 = 5000, but available = 1000 → min(5000,1000) = 1000
        assert s.size(100, 1000, 10000) == 10

    def test_invalid_pct(self):
        with pytest.raises(AssertionError):
            PercentOfPortfolio(pct=0.0)
        with pytest.raises(AssertionError):
            PercentOfPortfolio(pct=1.5)

    def test_zero_cost(self):
        s = PercentOfPortfolio(pct=0.01)
        assert s.size(0, 1000, 10000) == 0
