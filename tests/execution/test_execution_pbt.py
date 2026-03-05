"""Property-based tests for execution models — cost, fill, sizer, selector.

Uses Hypothesis to fuzz all execution components with random inputs and verify
mathematical invariants hold across the entire input space.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

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
# Hypothesis strategies
# ---------------------------------------------------------------------------

price = st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False)
quantity_int = st.integers(min_value=0, max_value=100_000)
signed_qty = st.integers(min_value=-100_000, max_value=100_000)
spc = st.sampled_from([1, 10, 100, 1000])
rate = st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False)
pct_01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
capital = st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False)
direction = st.sampled_from([Direction.BUY, Direction.SELL])
volume = st.integers(min_value=0, max_value=10_000)


# ---------------------------------------------------------------------------
# Cost models — property-based
# ---------------------------------------------------------------------------


class TestNoCostsPBT:
    @given(price, signed_qty, spc)
    @settings(max_examples=100)
    def test_always_zero(self, p, q, s):
        m = NoCosts()
        assert m.option_cost(p, q, s) == 0.0
        assert m.stock_cost(p, q) == 0.0


class TestPerContractPBT:
    @given(rate, price, signed_qty, spc)
    @settings(max_examples=200)
    def test_non_negative(self, r, p, q, s):
        m = PerContractCommission(rate=r)
        assert m.option_cost(p, q, s) >= 0.0

    @given(rate, price, quantity_int, spc)
    @settings(max_examples=200)
    def test_symmetric_buy_sell(self, r, p, q, s):
        """Commission is the same for +q and -q (direction-independent)."""
        m = PerContractCommission(rate=r)
        assert m.option_cost(p, q, s) == m.option_cost(p, -q, s)

    @given(rate, price, quantity_int, spc)
    @settings(max_examples=100)
    def test_linear_in_quantity(self, r, p, q, s):
        """Doubling quantity doubles cost."""
        assume(q <= 50_000)
        m = PerContractCommission(rate=r)
        c1 = m.option_cost(p, q, s)
        c2 = m.option_cost(p, 2 * q, s)
        assert abs(c2 - 2 * c1) < 1e-8

    @given(rate, price, quantity_int)
    @settings(max_examples=100)
    def test_independent_of_price_and_spc(self, r, p, q):
        """Per-contract commission doesn't depend on price or spc."""
        m = PerContractCommission(rate=r)
        c1 = m.option_cost(p, q, 100)
        c2 = m.option_cost(p * 2, q, 1000)
        assert abs(c1 - c2) < 1e-8

    @given(st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
           price, signed_qty)
    @settings(max_examples=100)
    def test_stock_cost_non_negative(self, sr, p, q):
        m = PerContractCommission(stock_rate=sr)
        assert m.stock_cost(p, q) >= 0.0


class TestTieredCommissionPBT:
    @given(price, quantity_int, spc)
    @settings(max_examples=200)
    def test_non_negative(self, p, q, s):
        m = TieredCommission()
        assert m.option_cost(p, q, s) >= 0.0

    @given(price, quantity_int, spc)
    @settings(max_examples=200)
    def test_symmetric(self, p, q, s):
        m = TieredCommission()
        assert m.option_cost(p, q, s) == m.option_cost(p, -q, s)

    @given(st.integers(min_value=1, max_value=50_000))
    @settings(max_examples=200)
    def test_monotone_in_quantity(self, q):
        """More contracts never costs less total."""
        m = TieredCommission()
        c1 = m.option_cost(1.0, q, 100)
        c2 = m.option_cost(1.0, q + 1, 100)
        assert c2 >= c1 - 1e-10

    @given(st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=100)
    def test_bounded_by_flat_rate(self, q):
        """Tiered cost <= flat rate at highest tier rate * quantity."""
        m = TieredCommission()
        cost = m.option_cost(1.0, q, 100)
        max_rate = max(r for _, r in m.tiers)
        assert cost <= max_rate * q + 1e-8

    @given(st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=100)
    def test_bounded_below_by_lowest_rate(self, q):
        """Tiered cost >= min_rate * quantity."""
        m = TieredCommission()
        cost = m.option_cost(1.0, q, 100)
        min_rate = min(r for _, r in m.tiers)
        assert cost >= min_rate * q - 1e-8

    @given(st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=100)
    def test_average_rate_decreasing(self, q):
        """Average cost per contract decreases (or stays same) with volume."""
        assume(q < 100_000)
        m = TieredCommission()
        c1 = m.option_cost(1.0, q, 100)
        c2 = m.option_cost(1.0, q + 1000, 100)
        avg1 = c1 / q
        avg2 = c2 / (q + 1000)
        assert avg2 <= avg1 + 1e-10


class TestSpreadSlippagePBT:
    @given(pct_01, price, signed_qty, spc)
    @settings(max_examples=200)
    def test_option_cost_always_zero(self, pct, p, q, s):
        m = SpreadSlippage(pct=pct)
        assert m.option_cost(p, q, s) == 0.0

    @given(pct_01, st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
           quantity_int, spc)
    @settings(max_examples=200)
    def test_slippage_non_negative(self, pct, bid, ask, q, s):
        m = SpreadSlippage(pct=pct)
        slip = m.slippage(bid, ask, q, s)
        assert slip >= -1e-10

    @given(pct_01, price, quantity_int, spc)
    @settings(max_examples=100)
    def test_zero_spread_zero_slippage(self, pct, p, q, s):
        m = SpreadSlippage(pct=pct)
        assert m.slippage(p, p, q, s) == 0.0

    @given(st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
           price, quantity_int, spc)
    @settings(max_examples=100)
    def test_slippage_monotone_in_pct(self, pct, p, q, s):
        """Higher pct means more slippage."""
        assume(q > 0)
        bid = p * 0.95
        ask = p * 1.05
        m1 = SpreadSlippage(pct=pct)
        m2 = SpreadSlippage(pct=min(pct + 0.01, 1.0))
        assert m2.slippage(bid, ask, q, s) >= m1.slippage(bid, ask, q, s) - 1e-10

    @given(st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
           price, spc)
    @settings(max_examples=100)
    def test_slippage_linear_in_quantity(self, pct, p, s):
        """Double quantity → double slippage."""
        m = SpreadSlippage(pct=pct)
        bid, ask = p * 0.95, p * 1.05
        s1 = m.slippage(bid, ask, 10, s)
        s2 = m.slippage(bid, ask, 20, s)
        assert abs(s2 - 2 * s1) < 1e-6


# ---------------------------------------------------------------------------
# Fill models — property-based
# ---------------------------------------------------------------------------


class TestMarketAtBidAskPBT:
    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_buy_at_ask_sell_at_bid(self, bid, ask):
        assume(bid <= ask)
        m = MarketAtBidAsk()
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        assert m.get_fill_price(row, Direction.BUY) == ask
        assert m.get_fill_price(row, Direction.SELL) == bid

    @given(price)
    @settings(max_examples=100)
    def test_zero_spread_both_equal(self, p):
        m = MarketAtBidAsk()
        row = pd.Series({"bid": p, "ask": p, "volume": 100})
        assert m.get_fill_price(row, Direction.BUY) == m.get_fill_price(row, Direction.SELL)

    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_buy_never_cheaper_than_sell(self, bid, ask):
        """Buy fill >= sell fill (you always pay more to buy)."""
        assume(bid <= ask)
        m = MarketAtBidAsk()
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        assert m.get_fill_price(row, Direction.BUY) >= m.get_fill_price(row, Direction.SELL)


class TestMidPricePBT:
    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           direction)
    @settings(max_examples=200)
    def test_between_bid_and_ask(self, bid, ask, d):
        assume(bid <= ask)
        m = MidPrice()
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        mid = m.get_fill_price(row, d)
        assert bid - 1e-10 <= mid <= ask + 1e-10

    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_direction_independent(self, bid, ask):
        assume(bid <= ask)
        m = MidPrice()
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        assert m.get_fill_price(row, Direction.BUY) == m.get_fill_price(row, Direction.SELL)

    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_midpoint_formula(self, bid, ask):
        assume(bid <= ask)
        m = MidPrice()
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        expected = (bid + ask) / 2.0
        assert abs(m.get_fill_price(row, Direction.BUY) - expected) < 1e-10


class TestVolumeAwareFillPBT:
    @given(st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           volume, st.integers(min_value=1, max_value=10_000), direction)
    @settings(max_examples=300)
    def test_fill_between_mid_and_edge(self, bid, ask, vol, threshold, d):
        """Fill price is always between mid and bid/ask."""
        assume(bid <= ask)
        m = VolumeAwareFill(full_volume_threshold=threshold)
        row = pd.Series({"bid": bid, "ask": ask, "volume": vol})
        fill = m.get_fill_price(row, d)
        mid = (bid + ask) / 2.0
        if d == Direction.BUY:
            assert mid - 1e-10 <= fill <= ask + 1e-10
        else:
            assert bid - 1e-10 <= fill <= mid + 1e-10

    @given(st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=200)
    def test_zero_volume_fills_at_mid(self, bid, ask, threshold):
        assume(bid <= ask)
        m = VolumeAwareFill(full_volume_threshold=threshold)
        row = pd.Series({"bid": bid, "ask": ask, "volume": 0})
        mid = (bid + ask) / 2.0
        assert abs(m.get_fill_price(row, Direction.BUY) - mid) < 1e-10
        assert abs(m.get_fill_price(row, Direction.SELL) - mid) < 1e-10

    @given(st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=500), direction)
    @settings(max_examples=200)
    def test_higher_volume_moves_toward_edge(self, bid, ask, threshold, d):
        """More volume pushes fill toward bid/ask (worse for trader)."""
        assume(bid < ask)
        assume(threshold >= 4)
        m = VolumeAwareFill(full_volume_threshold=threshold)
        low_vol = pd.Series({"bid": bid, "ask": ask, "volume": threshold // 4})
        high_vol = pd.Series({"bid": bid, "ask": ask, "volume": threshold // 2})
        fill_low = m.get_fill_price(low_vol, d)
        fill_high = m.get_fill_price(high_vol, d)
        if d == Direction.BUY:
            assert fill_high >= fill_low - 1e-10
        else:
            assert fill_high <= fill_low + 1e-10

    @given(st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=500, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=500))
    @settings(max_examples=200)
    def test_above_threshold_equals_market(self, bid, ask, threshold):
        assume(bid <= ask)
        m = VolumeAwareFill(full_volume_threshold=threshold)
        row = pd.Series({"bid": bid, "ask": ask, "volume": threshold * 2})
        market = MarketAtBidAsk()
        assert abs(m.get_fill_price(row, Direction.BUY) - market.get_fill_price(row, Direction.BUY)) < 1e-10
        assert abs(m.get_fill_price(row, Direction.SELL) - market.get_fill_price(row, Direction.SELL)) < 1e-10


# ---------------------------------------------------------------------------
# Signal selectors — property-based
# ---------------------------------------------------------------------------

def _make_candidates(n, deltas=None, ois=None):
    data = {
        "contract": [f"SPX{i}" for i in range(n)],
        "strike": [100 + i * 5 for i in range(n)],
        "bid": [1.0 + i * 0.1 for i in range(n)],
        "ask": [1.5 + i * 0.1 for i in range(n)],
    }
    if deltas is not None:
        data["delta"] = deltas
    if ois is not None:
        data["openinterest"] = ois
    return pd.DataFrame(data)


class TestFirstMatchPBT:
    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_always_selects_row_from_dataframe(self, n):
        s = FirstMatch()
        df = _make_candidates(n)
        result = s.select(df)
        assert result["contract"] in df["contract"].values

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_always_selects_first(self, n):
        s = FirstMatch()
        df = _make_candidates(n)
        assert s.select(df)["contract"] == "SPX0"


class TestNearestDeltaPBT:
    @given(st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
           st.integers(min_value=2, max_value=30))
    @settings(max_examples=200)
    def test_selected_is_closest_to_target(self, target, n):
        """Selected row has smallest |delta - target| of all rows."""
        deltas = [-0.05 * (i + 1) for i in range(n)]
        s = NearestDelta(target_delta=target)
        df = _make_candidates(n, deltas=deltas)
        result = s.select(df)
        result_diff = abs(result["delta"] - target)
        min_diff = (df["delta"] - target).abs().min()
        assert result_diff <= min_diff + 1e-10

    @given(st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_result_is_valid_row(self, target, n):
        deltas = [-0.05 * (i + 1) for i in range(n)]
        s = NearestDelta(target_delta=target)
        df = _make_candidates(n, deltas=deltas)
        result = s.select(df)
        assert result["contract"] in df["contract"].values

    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_missing_delta_falls_back_to_first(self, n):
        s = NearestDelta(target_delta=-0.30)
        df = _make_candidates(n)  # no delta column
        result = s.select(df)
        assert result["contract"] == "SPX0"


class TestMaxOpenInterestPBT:
    @given(st.lists(st.integers(min_value=0, max_value=100_000), min_size=2, max_size=30))
    @settings(max_examples=200)
    def test_selects_max_oi(self, ois):
        s = MaxOpenInterest()
        df = _make_candidates(len(ois), ois=ois)
        result = s.select(df)
        assert result["openinterest"] == max(ois)

    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_missing_oi_falls_back(self, n):
        s = MaxOpenInterest()
        df = _make_candidates(n)  # no OI column
        result = s.select(df)
        assert result["contract"] == "SPX0"

    @given(st.integers(min_value=2, max_value=20), st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=100)
    def test_uniform_oi_selects_some_row(self, n, oi):
        """When all OI equal, still returns a valid row."""
        s = MaxOpenInterest()
        df = _make_candidates(n, ois=[oi] * n)
        result = s.select(df)
        assert result["contract"] in df["contract"].values


# ---------------------------------------------------------------------------
# Sizers — property-based
# ---------------------------------------------------------------------------


class TestCapitalBasedPBT:
    @given(st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=200)
    def test_non_negative_integer(self, cost, avail, total):
        s = CapitalBased()
        qty = s.size(cost, avail, total)
        assert isinstance(qty, int)
        assert qty >= 0

    @given(st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=200)
    def test_total_cost_within_budget(self, cost, avail, total):
        """qty * cost <= available_capital."""
        s = CapitalBased()
        qty = s.size(cost, avail, total)
        assert qty * cost <= avail + 1e-6

    @given(capital, capital)
    @settings(max_examples=50)
    def test_zero_cost_returns_zero(self, avail, total):
        s = CapitalBased()
        assert s.size(0, avail, total) == 0

    @given(st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
           capital)
    @settings(max_examples=200)
    def test_more_capital_more_contracts(self, cost, avail, total):
        """Doubling available capital never decreases contracts."""
        assume(avail * 2 <= 1e9)
        s = CapitalBased()
        q1 = s.size(cost, avail, total)
        q2 = s.size(cost, avail * 2, total)
        assert q2 >= q1

    @given(st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=100)
    def test_negative_cost_same_as_positive(self, cost, avail, total):
        s = CapitalBased()
        assert s.size(cost, avail, total) == s.size(-cost, avail, total)


class TestFixedQuantityPBT:
    @given(st.integers(min_value=1, max_value=1000),
           st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=200)
    def test_never_exceeds_budget(self, qty, cost, avail, total):
        s = FixedQuantity(quantity=qty)
        result = s.size(cost, avail, total)
        assert result * cost <= avail + 1e-6

    @given(st.integers(min_value=1, max_value=100),
           st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=1e6, max_value=1e9, allow_nan=False, allow_infinity=False),
           capital)
    @settings(max_examples=100)
    def test_large_budget_returns_fixed(self, qty, cost, avail, total):
        """With huge available capital, always returns the fixed quantity."""
        s = FixedQuantity(quantity=qty)
        assert s.size(cost, avail, total) == qty

    @given(st.integers(min_value=1, max_value=1000),
           st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=100)
    def test_bounded_by_capital_based(self, qty, cost, avail, total):
        """FixedQuantity result <= CapitalBased result or == qty."""
        s = FixedQuantity(quantity=qty)
        result = s.size(cost, avail, total)
        cap_result = CapitalBased().size(cost, avail, total)
        assert result <= max(qty, cap_result) + 1


class TestFixedDollarPBT:
    @given(st.floats(min_value=100, max_value=1e6, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=200)
    def test_non_negative_integer(self, amount, cost, avail, total):
        s = FixedDollar(amount=amount)
        qty = s.size(cost, avail, total)
        assert isinstance(qty, int)
        assert qty >= 0

    @given(st.floats(min_value=100, max_value=1e6, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=200)
    def test_total_cost_within_min_amount_avail(self, amount, cost, avail, total):
        """qty * cost <= min(amount, avail)."""
        s = FixedDollar(amount=amount)
        qty = s.size(cost, avail, total)
        assert qty * cost <= min(amount, avail) + 1e-6

    @given(st.floats(min_value=100, max_value=1e6, allow_nan=False, allow_infinity=False),
           capital, capital)
    @settings(max_examples=50)
    def test_zero_cost_returns_zero(self, amount, avail, total):
        s = FixedDollar(amount=amount)
        assert s.size(0, avail, total) == 0


class TestPercentOfPortfolioPBT:
    @given(st.floats(min_value=0.001, max_value=0.99, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital,
           st.floats(min_value=1000, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_non_negative_integer(self, pct, cost, avail, total):
        s = PercentOfPortfolio(pct=pct)
        qty = s.size(cost, avail, total)
        assert isinstance(qty, int)
        assert qty >= 0

    @given(st.floats(min_value=0.001, max_value=0.99, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           capital,
           st.floats(min_value=1000, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_cost_bounded_by_pct_of_total(self, pct, cost, avail, total):
        """qty * cost <= pct * total (or available, whichever is smaller)."""
        s = PercentOfPortfolio(pct=pct)
        qty = s.size(cost, avail, total)
        assert qty * cost <= min(pct * total, avail) + 1e-6

    @given(st.floats(min_value=0.001, max_value=0.99, allow_nan=False, allow_infinity=False),
           capital,
           st.floats(min_value=1000, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_zero_cost_returns_zero(self, pct, avail, total):
        s = PercentOfPortfolio(pct=pct)
        assert s.size(0, avail, total) == 0

    @given(st.floats(min_value=0.01, max_value=0.49, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=1e6, max_value=1e9, allow_nan=False, allow_infinity=False),
           st.floats(min_value=1e6, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_higher_pct_more_contracts(self, pct, cost, avail, total):
        """Higher pct never gives fewer contracts."""
        s1 = PercentOfPortfolio(pct=pct)
        s2 = PercentOfPortfolio(pct=min(pct + 0.01, 1.0))
        assert s2.size(cost, avail, total) >= s1.size(cost, avail, total)
