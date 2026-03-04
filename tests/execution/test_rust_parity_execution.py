"""Rust execution model tests: edge cases, invariants, PBT, integration.

Covers:
- Edge-case fuzzing (NaN, Inf, empty, zero, extreme values)
- Property-based testing (Hypothesis)
- Invariant tests (cost >= 0, fill between bid/ask, index in range, monotonicity)
- Integration tests (Python classes delegating to Rust)
"""

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from options_portfolio_backtester.core.types import Direction, Greeks
from options_portfolio_backtester._ob_rust import (
    rust_option_cost,
    rust_stock_cost,
    rust_fill_price,
    rust_nearest_delta_index,
    rust_max_value_index,
    rust_risk_check,
)


# ===================================================================
# EDGE-CASE / FUZZING TESTS
# ===================================================================

class TestCostModelEdgeCases:

    def test_per_contract_basic(self):
        assert rust_option_cost("PerContract", 0.65, 0.005, [], 10.0, 10.0, 100) == pytest.approx(6.5)

    def test_per_contract_negative_quantity(self):
        assert rust_option_cost("PerContract", 0.65, 0.005, [], 10.0, -10.0, 100) == pytest.approx(6.5)

    def test_per_contract_zero_quantity(self):
        assert rust_option_cost("PerContract", 0.65, 0.005, [], 10.0, 0.0, 100) == 0.0

    def test_per_contract_stock(self):
        assert rust_stock_cost("PerContract", 0.65, 0.005, [], 150.0, 100.0) == pytest.approx(0.5)

    def test_zero_rate(self):
        assert rust_option_cost("PerContract", 0.0, 0.0, [], 10.0, 100.0, 100) == 0.0
        assert rust_stock_cost("PerContract", 0.65, 0.0, [], 150.0, 100.0) == 0.0

    def test_very_small_quantity(self):
        assert rust_option_cost("PerContract", 0.65, 0.005, [], 10.0, 0.001, 100) == pytest.approx(0.65 * 0.001)

    def test_tiered_within_first_tier(self):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, 100.0, 100) == pytest.approx(65.0)

    def test_tiered_spanning_tiers(self):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        expected = 10_000 * 0.65 + 5_000 * 0.50
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, 15_000.0, 100) == pytest.approx(expected)

    def test_tiered_beyond_all(self):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        expected = 10_000 * 0.65 + 40_000 * 0.50 + 50_000 * 0.25 + 20_000 * 0.25
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, 120_000.0, 100) == pytest.approx(expected)

    def test_tiered_exactly_at_boundary(self):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, 10_000.0, 100) == pytest.approx(6500.0)

    def test_tiered_negative_quantity(self):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        expected = 10_000 * 0.65 + 5_000 * 0.50
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, -15_000.0, 100) == pytest.approx(expected)

    def test_very_large_quantity(self):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, 1e9, 100) > 0

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown cost model type"):
            rust_option_cost("Bogus", 0.65, 0.005, [], 10.0, 10.0, 100)


class TestFillModelEdgeCases:

    def test_full_volume_buy(self):
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, 100.0, True) == pytest.approx(10.0)

    def test_full_volume_sell(self):
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, 200.0, False) == pytest.approx(9.0)

    def test_zero_volume(self):
        mid = (9.0 + 10.0) / 2.0
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, 0.0, True) == pytest.approx(mid)
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, 0.0, False) == pytest.approx(mid)

    def test_half_volume(self):
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, 50.0, True) == pytest.approx(9.75)
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, 50.0, False) == pytest.approx(9.25)

    def test_missing_volume(self):
        assert rust_fill_price("VolumeAware", 100, 9.0, 10.0, None, True) == pytest.approx(10.0)

    def test_zero_bid_ask(self):
        assert rust_fill_price("VolumeAware", 100, 0.0, 0.0, 50.0, True) == 0.0

    def test_bid_equals_ask(self):
        assert rust_fill_price("VolumeAware", 100, 5.0, 5.0, 50.0, True) == pytest.approx(5.0)

    def test_invalid_fill_type_raises(self):
        with pytest.raises(ValueError, match="Unknown fill model type"):
            rust_fill_price("Bogus", 100, 9.0, 10.0, 50.0, True)


class TestSignalSelectorEdgeCases:

    def test_nearest_delta_exact(self):
        assert rust_nearest_delta_index([-0.20, -0.30, -0.45], -0.30) == 1

    def test_nearest_delta_between(self):
        assert rust_nearest_delta_index([-0.20, -0.30, -0.45], -0.35) == 1

    def test_empty_list(self):
        assert rust_nearest_delta_index([], -0.30) == 0
        assert rust_max_value_index([]) == 0

    def test_all_nan_nearest(self):
        assert rust_nearest_delta_index([float('nan'), float('nan')], -0.30) == 0

    def test_all_nan_max(self):
        assert rust_max_value_index([float('nan'), float('nan')]) == 0

    def test_single_element(self):
        assert rust_nearest_delta_index([0.5], 0.0) == 0
        assert rust_max_value_index([42.0]) == 0

    def test_max_value_basic(self):
        assert rust_max_value_index([500.0, 1200.0, 800.0]) == 1

    def test_max_value_negative(self):
        assert rust_max_value_index([-10.0, -5.0, -20.0]) == 1

    def test_max_value_ties_first_wins(self):
        assert rust_max_value_index([100.0, 100.0, 50.0]) == 0

    def test_large_list(self):
        assert rust_max_value_index([float(v) for v in range(10_000)]) == 9_999


class TestRiskCheckEdgeCases:

    def test_max_delta_allows(self):
        assert rust_risk_check("MaxDelta", 100.0, [50, 0, 0, 0], [30, 0, 0, 0], 1e6, 1e6) is True

    def test_max_delta_rejects(self):
        assert rust_risk_check("MaxDelta", 100.0, [80, 0, 0, 0], [30, 0, 0, 0], 1e6, 1e6) is False

    def test_max_delta_exactly_at_limit(self):
        assert rust_risk_check("MaxDelta", 100.0, [50, 0, 0, 0], [50, 0, 0, 0], 1e6, 1e6) is True

    def test_max_delta_negative(self):
        assert rust_risk_check("MaxDelta", 100.0, [-80, 0, 0, 0], [-30, 0, 0, 0], 1e6, 1e6) is False

    def test_max_vega_allows(self):
        assert rust_risk_check("MaxVega", 50.0, [0, 0, 0, 20], [0, 0, 0, 10], 1e6, 1e6) is True

    def test_max_vega_rejects(self):
        assert rust_risk_check("MaxVega", 50.0, [0, 0, 0, 40], [0, 0, 0, 20], 1e6, 1e6) is False

    def test_max_drawdown_allows(self):
        g = [0, 0, 0, 0]
        assert rust_risk_check("MaxDrawdown", 0.20, g, g, 900_000, 1_000_000) is True

    def test_max_drawdown_rejects(self):
        g = [0, 0, 0, 0]
        assert rust_risk_check("MaxDrawdown", 0.20, g, g, 750_000, 1_000_000) is False

    def test_max_drawdown_zero_peak(self):
        g = [0, 0, 0, 0]
        assert rust_risk_check("MaxDrawdown", 0.20, g, g, 100, 0.0) is True

    def test_zero_greeks(self):
        g = [0, 0, 0, 0]
        assert rust_risk_check("MaxDelta", 100.0, g, g, 1e6, 1e6) is True
        assert rust_risk_check("MaxVega", 50.0, g, g, 1e6, 1e6) is True

    def test_negative_peak_value(self):
        g = [0, 0, 0, 0]
        assert rust_risk_check("MaxDrawdown", 0.20, g, g, 100, -1.0) is True

    def test_invalid_constraint_raises(self):
        with pytest.raises(ValueError, match="Unknown risk constraint type"):
            rust_risk_check("Bogus", 100.0, [0]*4, [0]*4, 1e6, 1e6)


# ===================================================================
# PROPERTY-BASED TESTS
# ===================================================================

safe_float = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
safe_qty = st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
delta_float = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.01, max_value=1e8, allow_nan=False, allow_infinity=False)


class TestCostInvariants:

    @given(rate=safe_float, qty=safe_qty)
    @settings(max_examples=200)
    def test_cost_always_non_negative(self, rate, qty):
        assert rust_option_cost("PerContract", rate, 0.005, [], 10.0, qty, 100) >= 0.0

    @given(qty=st.floats(min_value=0.0, max_value=200_000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_tiered_cost_non_negative(self, qty):
        tiers = [(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)]
        assert rust_option_cost("Tiered", 0.0, 0.005, tiers, 10.0, qty, 100) >= 0.0

    @given(rate=safe_float, qty=safe_qty)
    @settings(max_examples=200)
    def test_sign_symmetry(self, rate, qty):
        pos = rust_option_cost("PerContract", rate, 0.005, [], 10.0, qty, 100)
        neg = rust_option_cost("PerContract", rate, 0.005, [], 10.0, -qty, 100)
        assert pos == pytest.approx(neg, rel=1e-12)


class TestFillInvariants:

    @given(
        bid=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
        spread=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        vol=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        is_buy=st.booleans(),
    )
    @settings(max_examples=200)
    def test_fill_between_bid_ask(self, bid, spread, vol, is_buy):
        ask = bid + spread
        price = rust_fill_price("VolumeAware", 100, bid, ask, vol, is_buy)
        assert price >= bid - 1e-10
        assert price <= ask + 1e-10


class TestSelectorInvariants:

    @given(
        values=st.lists(delta_float, min_size=1, max_size=50),
        target=delta_float,
    )
    @settings(max_examples=200)
    def test_index_in_range(self, values, target):
        idx = rust_nearest_delta_index(values, target)
        assert 0 <= idx < len(values)

    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=50,
        ),
    )
    @settings(max_examples=200)
    def test_max_index_in_range(self, values):
        idx = rust_max_value_index(values)
        assert 0 <= idx < len(values)


class TestRiskInvariants:

    @given(
        limit_low=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        limit_high=st.floats(min_value=500.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        delta=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_higher_limit_more_permissive(self, limit_low, limit_high, delta):
        cur = [delta, 0.0, 0.0, 0.0]
        prop = [0.0, 0.0, 0.0, 0.0]
        low_ok = rust_risk_check("MaxDelta", limit_low, cur, prop, 1e6, 1e6)
        high_ok = rust_risk_check("MaxDelta", limit_high, cur, prop, 1e6, 1e6)
        if low_ok:
            assert high_ok


# ===================================================================
# INTEGRATION: Python classes delegating to Rust
# ===================================================================

class TestPythonClassDelegation:

    def test_per_contract_via_class(self):
        from options_portfolio_backtester.execution.cost_model import PerContractCommission
        pc = PerContractCommission(0.65, 0.005)
        assert pc.option_cost(10.0, 10, 100) == pytest.approx(6.5)
        assert pc.stock_cost(150.0, 100) == pytest.approx(0.5)

    def test_tiered_via_class(self):
        from options_portfolio_backtester.execution.cost_model import TieredCommission
        tc = TieredCommission()
        assert tc.option_cost(10.0, 100, 100) == pytest.approx(65.0)
        assert tc.option_cost(10.0, 15_000, 100) == pytest.approx(10_000 * 0.65 + 5_000 * 0.50)

    def test_volume_aware_via_class(self):
        from options_portfolio_backtester.execution.fill_model import VolumeAwareFill
        vf = VolumeAwareFill(100)
        row = pd.Series({"bid": 9.0, "ask": 10.0, "volume": 50.0})
        assert vf.get_fill_price(row, Direction.BUY) == pytest.approx(9.75)
        assert vf.get_fill_price(row, Direction.SELL) == pytest.approx(9.25)

    def test_nearest_delta_via_class(self):
        from options_portfolio_backtester.execution.signal_selector import NearestDelta
        df = pd.DataFrame({"delta": [-0.20, -0.30, -0.45], "price": [1.0, 2.0, 3.0]})
        nd = NearestDelta(-0.30)
        assert nd.select(df)["delta"] == pytest.approx(-0.30)

    def test_max_oi_via_class(self):
        from options_portfolio_backtester.execution.signal_selector import MaxOpenInterest
        df = pd.DataFrame({"openinterest": [500, 1200, 800], "price": [1.0, 2.0, 3.0]})
        assert MaxOpenInterest().select(df)["openinterest"] == 1200

    def test_max_delta_via_class(self):
        from options_portfolio_backtester.portfolio.risk import MaxDelta
        md = MaxDelta(100.0)
        assert md.check(Greeks(50, 0, 0, 0), Greeks(30, 0, 0, 0), 1e6, 1e6) is True
        assert md.check(Greeks(80, 0, 0, 0), Greeks(30, 0, 0, 0), 1e6, 1e6) is False

    def test_max_vega_via_class(self):
        from options_portfolio_backtester.portfolio.risk import MaxVega
        mv = MaxVega(50.0)
        assert mv.check(Greeks(0, 0, 0, 20), Greeks(0, 0, 0, 10), 1e6, 1e6) is True
        assert mv.check(Greeks(0, 0, 0, 40), Greeks(0, 0, 0, 20), 1e6, 1e6) is False

    def test_max_drawdown_via_class(self):
        from options_portfolio_backtester.portfolio.risk import MaxDrawdown
        mdd = MaxDrawdown(0.20)
        assert mdd.check(Greeks(), Greeks(), 900_000, 1_000_000) is True
        assert mdd.check(Greeks(), Greeks(), 750_000, 1_000_000) is False
        assert mdd.check(Greeks(), Greeks(), 100, 0.0) is True
