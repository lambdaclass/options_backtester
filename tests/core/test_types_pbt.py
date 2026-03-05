"""Property-based tests for core domain types.

Fuzzes Greeks algebra, Fill notional, Direction/Order/OptionType enum inversions,
and get_order mapping with Hypothesis.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from options_portfolio_backtester.core.types import (
    Direction, OptionType, Order, Signal, Greeks, Fill, get_order,
    StockAllocation,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

greek_float = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
scalar = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
price = st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False)
quantity = st.integers(min_value=1, max_value=10_000)
spc = st.sampled_from([1, 10, 100, 1000])
commission = st.floats(min_value=0.0, max_value=1000, allow_nan=False, allow_infinity=False)
slippage = st.floats(min_value=0.0, max_value=1000, allow_nan=False, allow_infinity=False)
direction = st.sampled_from([Direction.BUY, Direction.SELL])
option_type = st.sampled_from([OptionType.CALL, OptionType.PUT])
signal = st.sampled_from([Signal.ENTRY, Signal.EXIT])
order = st.sampled_from([Order.BTO, Order.BTC, Order.STO, Order.STC])
pct = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

greeks = st.builds(
    Greeks,
    delta=greek_float,
    gamma=greek_float,
    theta=greek_float,
    vega=greek_float,
)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


class TestDirectionPBT:
    @given(direction)
    def test_price_column_is_bid_or_ask(self, d):
        assert d.price_column in {"bid", "ask"}

    @given(direction)
    def test_buy_maps_to_ask(self, d):
        if d == Direction.BUY:
            assert d.price_column == "ask"
        else:
            assert d.price_column == "bid"

    @given(direction)
    def test_invert_changes_price_column(self, d):
        assert d.price_column != (~d).price_column

    @given(direction)
    def test_double_invert_identity(self, d):
        assert ~~d == d

    @given(direction)
    def test_invert_is_different(self, d):
        assert ~d != d


# ---------------------------------------------------------------------------
# OptionType
# ---------------------------------------------------------------------------


class TestOptionTypePBT:
    @given(option_type)
    def test_double_invert_identity(self, ot):
        assert ~~ot == ot

    @given(option_type)
    def test_invert_is_different(self, ot):
        assert ~ot != ot

    @given(option_type)
    def test_call_inverts_to_put(self, ot):
        if ot == OptionType.CALL:
            assert ~ot == OptionType.PUT
        else:
            assert ~ot == OptionType.CALL


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


class TestOrderPBT:
    @given(order)
    def test_double_invert_identity(self, o):
        assert ~~o == o

    @given(order)
    def test_invert_changes_buy_sell(self, o):
        """BTO↔STC, STO↔BTC: invert swaps buy/sell side."""
        inv = ~o
        assert inv != o

    @given(direction, signal)
    def test_get_order_exhaustive(self, d, s):
        """get_order always returns a valid Order."""
        o = get_order(d, s)
        assert isinstance(o, Order)

    @given(direction, signal)
    def test_get_order_entry_exit_paired(self, d, s):
        """Entry and exit orders for same direction are inverses."""
        entry = get_order(d, Signal.ENTRY)
        exit_ = get_order(d, Signal.EXIT)
        assert ~entry == exit_

    @given(direction)
    def test_buy_entry_is_bto(self, d):
        if d == Direction.BUY:
            assert get_order(d, Signal.ENTRY) == Order.BTO
        else:
            assert get_order(d, Signal.ENTRY) == Order.STO


# ---------------------------------------------------------------------------
# Greeks — extensive algebra properties
# ---------------------------------------------------------------------------


class TestGreeksFieldsPBT:
    @given(greeks)
    def test_has_four_fields(self, g):
        d = g.as_dict
        assert set(d.keys()) == {"delta", "gamma", "theta", "vega"}

    @given(greeks)
    def test_frozen(self, g):
        with pytest.raises(AttributeError):
            g.delta = 999.0

    @given(greek_float, greek_float, greek_float, greek_float)
    def test_construction(self, d, ga, th, v):
        g = Greeks(delta=d, gamma=ga, theta=th, vega=v)
        assert g.delta == d
        assert g.gamma == ga
        assert g.theta == th
        assert g.vega == v


class TestGreeksAdditionPBT:
    @given(greeks, greeks)
    @settings(max_examples=200)
    def test_commutative(self, a, b):
        assert _greeks_close(a + b, b + a)

    @given(greeks, greeks, greeks)
    @settings(max_examples=200)
    def test_associative(self, a, b, c):
        assert _greeks_close((a + b) + c, a + (b + c), tol=1e-8)

    @given(greeks)
    @settings(max_examples=100)
    def test_zero_identity(self, g):
        assert _greeks_close(g + Greeks(), g)

    @given(greeks)
    @settings(max_examples=100)
    def test_inverse(self, g):
        assert _greeks_close(g + (-g), Greeks(), tol=1e-10)


class TestGreeksScalarMulPBT:
    @given(greeks, scalar)
    @settings(max_examples=200)
    def test_componentwise(self, g, s):
        r = g * s
        assert abs(r.delta - g.delta * s) < 1e-6
        assert abs(r.gamma - g.gamma * s) < 1e-6
        assert abs(r.theta - g.theta * s) < 1e-6
        assert abs(r.vega - g.vega * s) < 1e-6

    @given(greeks)
    @settings(max_examples=50)
    def test_identity(self, g):
        assert _greeks_close(g * 1.0, g)

    @given(greeks)
    @settings(max_examples=50)
    def test_zero(self, g):
        assert _greeks_close(g * 0.0, Greeks(), tol=1e-10)

    @given(greeks, scalar)
    @settings(max_examples=200)
    def test_rmul(self, g, s):
        assert _greeks_close(s * g, g * s)

    @given(greeks, greeks, scalar)
    @settings(max_examples=200)
    def test_distributes_over_addition(self, a, b, s):
        r1 = (a + b) * s
        r2 = (a * s) + (b * s)
        tol = max(abs(r1.delta), abs(r2.delta), 1) * 1e-6 + 1e-10
        assert abs(r1.delta - r2.delta) < tol
        assert abs(r1.vega - r2.vega) < tol


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------


class TestFillPBT:
    @given(price, quantity, direction, spc)
    @settings(max_examples=200)
    def test_direction_sign_matches(self, p, q, d, s):
        f = Fill(price=p, quantity=q, direction=d, shares_per_contract=s)
        expected = -1 if d == Direction.BUY else 1
        assert f.direction_sign == expected

    @given(price, quantity, spc, commission, slippage)
    @settings(max_examples=200)
    def test_sell_notional_exceeds_buy(self, p, q, s, comm, slip):
        """SELL notional > BUY notional for same price/qty (costs reduce both)."""
        buy = Fill(price=p, quantity=q, direction=Direction.BUY,
                   shares_per_contract=s, commission=comm, slippage=slip)
        sell = Fill(price=p, quantity=q, direction=Direction.SELL,
                    shares_per_contract=s, commission=comm, slippage=slip)
        assert sell.notional > buy.notional

    @given(price, quantity, direction, spc)
    @settings(max_examples=100)
    def test_zero_costs_notional(self, p, q, d, s):
        f = Fill(price=p, quantity=q, direction=d, shares_per_contract=s)
        expected = f.direction_sign * p * q * s
        assert abs(f.notional - expected) < 1e-6

    @given(price, quantity, direction, spc, commission, slippage)
    @settings(max_examples=200)
    def test_costs_reduce_notional(self, p, q, d, s, comm, slip):
        f_clean = Fill(price=p, quantity=q, direction=d, shares_per_contract=s)
        f_dirty = Fill(price=p, quantity=q, direction=d, shares_per_contract=s,
                       commission=comm, slippage=slip)
        assert f_dirty.notional <= f_clean.notional + 1e-10

    @given(price, quantity, direction, spc,
           st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_higher_commission_lower_notional(self, p, q, d, s, c1, c2):
        assume(c2 > c1)
        f1 = Fill(price=p, quantity=q, direction=d, shares_per_contract=s, commission=c1)
        f2 = Fill(price=p, quantity=q, direction=d, shares_per_contract=s, commission=c2)
        assert f2.notional <= f1.notional + 1e-10


# ---------------------------------------------------------------------------
# StockAllocation
# ---------------------------------------------------------------------------


class TestStockAllocationPBT:
    @given(st.text(min_size=1, max_size=10), pct)
    @settings(max_examples=50)
    def test_named_tuple_fields(self, sym, p):
        sa = StockAllocation(symbol=sym, percentage=p)
        assert sa.symbol == sym
        assert sa.percentage == p
        assert sa[0] == sym
        assert sa[1] == p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _greeks_close(a: Greeks, b: Greeks, tol: float = 1e-10) -> bool:
    return (
        abs(a.delta - b.delta) < tol
        and abs(a.gamma - b.gamma) < tol
        and abs(a.theta - b.theta) < tol
        and abs(a.vega - b.vega) < tol
    )
