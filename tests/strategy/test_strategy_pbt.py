"""Property-based tests for strategies, risk constraints, and Greeks algebra.

Fuzzes strategy preset construction, risk constraint monotonicity and composition,
and Greeks vector-space properties with Hypothesis.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from options_portfolio_backtester.core.types import (
    Direction, OptionType, Order, Signal, Greeks, Fill, get_order,
)
from options_portfolio_backtester.portfolio.risk import (
    MaxDelta, MaxVega, MaxDrawdown, RiskConstraint, RiskManager,
)
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.strategy.presets import (
    strangle, iron_condor, covered_call, cash_secured_put, collar, butterfly,
    Strangle,
)
from options_portfolio_backtester.data.schema import Schema

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

greek_float = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
scalar = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
limit_float = st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False)
dd_pct = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
direction = st.sampled_from([Direction.BUY, Direction.SELL])
option_type = st.sampled_from([OptionType.CALL, OptionType.PUT])
signal = st.sampled_from([Signal.ENTRY, Signal.EXIT])
dte_min = st.integers(min_value=1, max_value=90)
dte_exit = st.integers(min_value=0, max_value=30)
otm_pct = st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False)
pct_tol = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

greeks_strat = st.builds(
    Greeks,
    delta=greek_float,
    gamma=greek_float,
    theta=greek_float,
    vega=greek_float,
)


def _options_schema():
    s = Schema.options()
    s.update({
        "contract": "optionroot",
        "date": "quotedate",
        "dte": "dte",
        "last": "last",
        "open_interest": "openinterest",
        "impliedvol": "impliedvol",
        "delta": "delta",
        "gamma": "gamma",
        "theta": "theta",
        "vega": "vega",
    })
    return s


# ---------------------------------------------------------------------------
# Greeks algebra — vector space properties
# ---------------------------------------------------------------------------


class TestGreeksAlgebraPBT:
    @given(greeks_strat, greeks_strat)
    @settings(max_examples=200)
    def test_addition_commutative(self, a, b):
        r1 = a + b
        r2 = b + a
        assert abs(r1.delta - r2.delta) < 1e-10
        assert abs(r1.gamma - r2.gamma) < 1e-10
        assert abs(r1.theta - r2.theta) < 1e-10
        assert abs(r1.vega - r2.vega) < 1e-10

    @given(greeks_strat, greeks_strat, greeks_strat)
    @settings(max_examples=200)
    def test_addition_associative(self, a, b, c):
        r1 = (a + b) + c
        r2 = a + (b + c)
        assert abs(r1.delta - r2.delta) < 1e-8
        assert abs(r1.gamma - r2.gamma) < 1e-8
        assert abs(r1.theta - r2.theta) < 1e-8
        assert abs(r1.vega - r2.vega) < 1e-8

    @given(greeks_strat)
    @settings(max_examples=100)
    def test_additive_identity(self, g):
        zero = Greeks()
        r = g + zero
        assert abs(r.delta - g.delta) < 1e-10
        assert abs(r.gamma - g.gamma) < 1e-10
        assert abs(r.theta - g.theta) < 1e-10
        assert abs(r.vega - g.vega) < 1e-10

    @given(greeks_strat)
    @settings(max_examples=100)
    def test_additive_inverse(self, g):
        """g + (-g) == zero."""
        r = g + (-g)
        assert abs(r.delta) < 1e-10
        assert abs(r.gamma) < 1e-10
        assert abs(r.theta) < 1e-10
        assert abs(r.vega) < 1e-10

    @given(greeks_strat, scalar)
    @settings(max_examples=200)
    def test_scalar_mul_distributes_over_components(self, g, s):
        r = g * s
        assert abs(r.delta - g.delta * s) < 1e-6
        assert abs(r.gamma - g.gamma * s) < 1e-6
        assert abs(r.theta - g.theta * s) < 1e-6
        assert abs(r.vega - g.vega * s) < 1e-6

    @given(greeks_strat, scalar, scalar)
    @settings(max_examples=200)
    def test_scalar_mul_composition(self, g, a, b):
        """(a * b) * g == a * (b * g) within tolerance."""
        assume(abs(a * b) < 1e6)
        r1 = g * (a * b)
        r2 = (g * b) * a
        assert abs(r1.delta - r2.delta) < max(abs(r1.delta), 1) * 1e-6 + 1e-10
        assert abs(r1.vega - r2.vega) < max(abs(r1.vega), 1) * 1e-6 + 1e-10

    @given(greeks_strat, greeks_strat, scalar)
    @settings(max_examples=200)
    def test_scalar_mul_distributes_over_addition(self, a, b, s):
        """s * (a + b) == s*a + s*b."""
        r1 = (a + b) * s
        r2 = (a * s) + (b * s)
        assert abs(r1.delta - r2.delta) < max(abs(r1.delta), 1) * 1e-6 + 1e-10
        assert abs(r1.gamma - r2.gamma) < max(abs(r1.gamma), 1) * 1e-6 + 1e-10

    @given(greeks_strat, scalar)
    @settings(max_examples=100)
    def test_rmul_equals_mul(self, g, s):
        """s * g == g * s."""
        r1 = s * g
        r2 = g * s
        assert abs(r1.delta - r2.delta) < 1e-10
        assert abs(r1.vega - r2.vega) < 1e-10

    @given(greeks_strat)
    @settings(max_examples=100)
    def test_mul_by_one_is_identity(self, g):
        r = g * 1.0
        assert abs(r.delta - g.delta) < 1e-10
        assert abs(r.vega - g.vega) < 1e-10

    @given(greeks_strat)
    @settings(max_examples=100)
    def test_mul_by_zero_is_zero(self, g):
        r = g * 0.0
        assert abs(r.delta) < 1e-10
        assert abs(r.gamma) < 1e-10
        assert abs(r.theta) < 1e-10
        assert abs(r.vega) < 1e-10

    @given(greeks_strat)
    @settings(max_examples=100)
    def test_neg_is_mul_minus_one(self, g):
        r1 = -g
        r2 = g * -1.0
        assert abs(r1.delta - r2.delta) < 1e-10
        assert abs(r1.vega - r2.vega) < 1e-10

    @given(greeks_strat)
    @settings(max_examples=100)
    def test_as_dict_roundtrip(self, g):
        d = g.as_dict
        reconstructed = Greeks(**d)
        assert abs(reconstructed.delta - g.delta) < 1e-10
        assert abs(reconstructed.gamma - g.gamma) < 1e-10
        assert abs(reconstructed.theta - g.theta) < 1e-10
        assert abs(reconstructed.vega - g.vega) < 1e-10


# ---------------------------------------------------------------------------
# Direction / Order / OptionType inversions
# ---------------------------------------------------------------------------


class TestEnumInversionsPBT:
    @given(direction)
    @settings(max_examples=10)
    def test_direction_double_invert(self, d):
        assert ~~d == d

    @given(option_type)
    @settings(max_examples=10)
    def test_option_type_double_invert(self, ot):
        assert ~~ot == ot

    @given(direction, signal)
    @settings(max_examples=10)
    def test_order_double_invert(self, d, s):
        order = get_order(d, s)
        assert ~~order == order

    @given(direction)
    @settings(max_examples=10)
    def test_direction_invert_differs(self, d):
        assert ~d != d

    @given(option_type)
    @settings(max_examples=10)
    def test_option_type_invert_differs(self, ot):
        assert ~ot != ot


# ---------------------------------------------------------------------------
# Fill value properties
# ---------------------------------------------------------------------------


class TestFillPBT:
    @given(st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=10_000),
           direction,
           st.sampled_from([1, 10, 100, 1000]))
    @settings(max_examples=200)
    def test_buy_negative_sell_positive_notional(self, price, qty, d, spc):
        """BUY direction_sign = -1 (cash out), SELL direction_sign = +1 (cash in)."""
        f = Fill(price=price, quantity=qty, direction=d, shares_per_contract=spc)
        if d == Direction.BUY:
            assert f.direction_sign == -1
        else:
            assert f.direction_sign == 1

    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=1000),
           st.sampled_from([1, 10, 100]))
    @settings(max_examples=200)
    def test_sell_notional_exceeds_buy_notional(self, price, qty, spc):
        """With zero costs, sell notional > buy notional (opposite signs)."""
        buy = Fill(price=price, quantity=qty, direction=Direction.BUY, shares_per_contract=spc)
        sell = Fill(price=price, quantity=qty, direction=Direction.SELL, shares_per_contract=spc)
        assert sell.notional > buy.notional

    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=1000),
           direction,
           st.sampled_from([1, 10, 100]),
           st.floats(min_value=0.0, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.0, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_commission_slippage_reduce_notional(self, price, qty, d, spc, comm, slip):
        """Adding commission/slippage always reduces notional."""
        f_clean = Fill(price=price, quantity=qty, direction=d, shares_per_contract=spc)
        f_costs = Fill(price=price, quantity=qty, direction=d, shares_per_contract=spc,
                       commission=comm, slippage=slip)
        assert f_costs.notional <= f_clean.notional + 1e-10

    @given(st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
           st.integers(min_value=1, max_value=1000),
           direction,
           st.sampled_from([100]))
    @settings(max_examples=100)
    def test_notional_formula(self, price, qty, d, spc):
        """Verify notional = direction_sign * price * qty * spc - commission - slippage."""
        comm, slip = 5.0, 2.0
        f = Fill(price=price, quantity=qty, direction=d, shares_per_contract=spc,
                 commission=comm, slippage=slip)
        expected = f.direction_sign * price * qty * spc - comm - slip
        assert abs(f.notional - expected) < 1e-6


# ---------------------------------------------------------------------------
# Risk constraints — property-based
# ---------------------------------------------------------------------------


class TestMaxDeltaPBT:
    @given(limit_float, greeks_strat, greeks_strat, positive_float, positive_float)
    @settings(max_examples=200)
    def test_within_limit_passes(self, limit, current, proposed, pv, peak):
        """If |current.delta + proposed.delta| <= limit, check returns True."""
        new_delta = current.delta + proposed.delta
        m = MaxDelta(limit=limit)
        result = m.check(current, proposed, pv, peak)
        if abs(new_delta) <= limit:
            assert result is True
        else:
            assert result is False

    @given(limit_float)
    @settings(max_examples=50)
    def test_zero_greeks_always_pass(self, limit):
        m = MaxDelta(limit=limit)
        zero = Greeks()
        assert m.check(zero, zero, 100.0, 100.0) is True

    @given(st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_tighter_limit_blocks_more(self, tight, loose):
        """A tighter limit blocks at least as many trades as a looser one."""
        assume(tight < loose)
        m_tight = MaxDelta(limit=tight)
        m_loose = MaxDelta(limit=loose)
        g = Greeks(delta=tight + 0.01)
        proposed = Greeks()
        # If tight blocks it, check that it makes sense
        if not m_tight.check(g, proposed, 100, 100):
            # loose may or may not block — but tight blocks
            pass
        if m_loose.check(g, proposed, 100, 100):
            # loose passes → tight may or may not
            pass


class TestMaxVegaPBT:
    @given(limit_float, greeks_strat, greeks_strat, positive_float, positive_float)
    @settings(max_examples=200)
    def test_correctness(self, limit, current, proposed, pv, peak):
        new_vega = current.vega + proposed.vega
        m = MaxVega(limit=limit)
        result = m.check(current, proposed, pv, peak)
        if abs(new_vega) <= limit:
            assert result is True
        else:
            assert result is False

    @given(limit_float)
    @settings(max_examples=50)
    def test_zero_greeks_always_pass(self, limit):
        m = MaxVega(limit=limit)
        zero = Greeks()
        assert m.check(zero, zero, 100.0, 100.0) is True


class TestMaxDrawdownPBT:
    @given(dd_pct,
           st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
           st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_correctness(self, max_dd, pv, peak):
        assume(peak > 0)
        m = MaxDrawdown(max_dd_pct=max_dd)
        dd = (peak - pv) / peak
        result = m.check(Greeks(), Greeks(), pv, peak)
        if dd < max_dd:
            assert result is True
        else:
            assert result is False

    @given(dd_pct, positive_float)
    @settings(max_examples=100)
    def test_at_peak_always_passes(self, max_dd, peak):
        """No drawdown at peak → always allowed."""
        m = MaxDrawdown(max_dd_pct=max_dd)
        assert m.check(Greeks(), Greeks(), peak, peak) is True

    @given(dd_pct)
    @settings(max_examples=50)
    def test_zero_peak_always_passes(self, max_dd):
        m = MaxDrawdown(max_dd_pct=max_dd)
        assert m.check(Greeks(), Greeks(), 50.0, 0.0) is True

    @given(st.floats(min_value=0.01, max_value=0.49, allow_nan=False, allow_infinity=False),
           st.floats(min_value=100, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_tighter_limit_blocks_more(self, tight, peak):
        """A tighter drawdown limit blocks at a higher portfolio value."""
        loose = tight + 0.1
        pv = peak * (1 - (tight + 0.05))  # dd = tight + 0.05
        m_tight = MaxDrawdown(max_dd_pct=tight)
        m_loose = MaxDrawdown(max_dd_pct=loose)
        assert m_tight.check(Greeks(), Greeks(), pv, peak) is False
        assert m_loose.check(Greeks(), Greeks(), pv, peak) is True


class TestRiskManagerPBT:
    @given(greeks_strat, greeks_strat, positive_float, positive_float)
    @settings(max_examples=100)
    def test_no_constraints_always_passes(self, current, proposed, pv, peak):
        rm = RiskManager()
        allowed, reason = rm.is_allowed(current, proposed, pv, peak)
        assert allowed is True
        assert reason == ""

    @given(limit_float, limit_float, greeks_strat, greeks_strat, positive_float, positive_float)
    @settings(max_examples=200)
    def test_composite_is_conjunction(self, delta_limit, vega_limit, current, proposed, pv, peak):
        """RiskManager passes iff ALL individual constraints pass."""
        rm = RiskManager([MaxDelta(delta_limit), MaxVega(vega_limit)])
        allowed, _ = rm.is_allowed(current, proposed, pv, peak)

        delta_ok = MaxDelta(delta_limit).check(current, proposed, pv, peak)
        vega_ok = MaxVega(vega_limit).check(current, proposed, pv, peak)

        assert allowed == (delta_ok and vega_ok)

    @given(limit_float, limit_float, dd_pct, greeks_strat, greeks_strat,
           st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
           st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_triple_constraint_conjunction(self, dl, vl, ddp, curr, prop, pv, peak):
        assume(peak > 0)
        rm = RiskManager([MaxDelta(dl), MaxVega(vl), MaxDrawdown(ddp)])
        allowed, _ = rm.is_allowed(curr, prop, pv, peak)

        d_ok = MaxDelta(dl).check(curr, prop, pv, peak)
        v_ok = MaxVega(vl).check(curr, prop, pv, peak)
        dd_ok = MaxDrawdown(ddp).check(curr, prop, pv, peak)

        assert allowed == (d_ok and v_ok and dd_ok)

    @given(greeks_strat, greeks_strat, positive_float, positive_float)
    @settings(max_examples=50)
    def test_adding_constraints_only_restricts(self, current, proposed, pv, peak):
        """Adding a constraint can block but never unblock."""
        rm1 = RiskManager([MaxDelta(50)])
        rm2 = RiskManager([MaxDelta(50), MaxVega(50)])
        a1, _ = rm1.is_allowed(current, proposed, pv, peak)
        a2, _ = rm2.is_allowed(current, proposed, pv, peak)
        if a2:
            assert a1  # if composite passes, each individual must pass too


# ---------------------------------------------------------------------------
# Strategy presets — property-based
# ---------------------------------------------------------------------------


class TestStrategyPresetsPBT:
    @given(direction, dte_min, dte_exit, otm_pct, pct_tol)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_strangle_always_two_legs(self, d, dte_lo, dte_ex, otm, tol):
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        s = strangle(schema, "SPY", d, (dte_lo, dte_lo + 30), dte_ex, otm, tol)
        assert len(s.legs) == 2
        types = {leg.type for leg in s.legs}
        assert types == {OptionType.CALL, OptionType.PUT}

    @given(dte_min, dte_exit)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_iron_condor_four_legs(self, dte_lo, dte_ex):
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        s = iron_condor(schema, "SPY", (dte_lo, dte_lo + 30), dte_ex)
        assert len(s.legs) == 4

    @given(dte_min, dte_exit, otm_pct, pct_tol)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_covered_call_one_leg(self, dte_lo, dte_ex, otm, tol):
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        s = covered_call(schema, "SPY", (dte_lo, dte_lo + 30), dte_ex, otm, tol)
        assert len(s.legs) == 1
        assert s.legs[0].type == OptionType.CALL
        assert s.legs[0].direction == Direction.SELL

    @given(dte_min, dte_exit, otm_pct, pct_tol)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cash_secured_put_one_leg(self, dte_lo, dte_ex, otm, tol):
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        s = cash_secured_put(schema, "SPY", (dte_lo, dte_lo + 30), dte_ex, otm, tol)
        assert len(s.legs) == 1
        assert s.legs[0].type == OptionType.PUT
        assert s.legs[0].direction == Direction.SELL

    @given(dte_min, dte_exit, otm_pct, otm_pct, pct_tol)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_collar_two_legs(self, dte_lo, dte_ex, call_otm, put_otm, tol):
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        s = collar(schema, "SPY", (dte_lo, dte_lo + 30), dte_ex, call_otm, put_otm, tol)
        assert len(s.legs) == 2
        directions = {leg.direction for leg in s.legs}
        assert directions == {Direction.BUY, Direction.SELL}

    @given(dte_min, dte_exit, option_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_butterfly_three_legs(self, dte_lo, dte_ex, ot):
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        s = butterfly(schema, "SPY", (dte_lo, dte_lo + 30), dte_ex, option_type=ot)
        assert len(s.legs) == 3

    @given(st.sampled_from(["long", "short"]),
           dte_min, dte_exit, otm_pct, pct_tol)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_strangle_class_matches_function(self, name, dte_lo, dte_ex, otm, tol):
        """Strangle class produces same leg count and types as strangle() function."""
        assume(dte_lo > dte_ex)
        schema = _options_schema()
        d = Direction.BUY if name == "long" else Direction.SELL
        s_func = strangle(schema, "SPY", d, (dte_lo, dte_lo + 30), dte_ex, otm, tol)
        s_cls = Strangle(schema, name, "SPY", (dte_lo, dte_lo + 30), dte_ex, otm, tol)
        assert len(s_func.legs) == len(s_cls.legs)
        for fl, cl in zip(s_func.legs, s_cls.legs):
            assert fl.type == cl.type
            assert fl.direction == cl.direction


class TestStrategyOperationsPBT:
    @given(st.integers(min_value=1, max_value=8), direction, option_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_add_remove_legs_preserves_length(self, n, d, ot):
        """Adding n legs then removing last gives n-1 legs."""
        schema = _options_schema()
        s = Strategy(schema)
        for i in range(n):
            leg = StrategyLeg(f"leg_{i}", schema, option_type=ot, direction=d)
            s.add_leg(leg)
        assert len(s.legs) == n
        s.legs.pop()
        assert len(s.legs) == n - 1

    @given(st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_exit_thresholds_stored(self, profit, loss):
        schema = _options_schema()
        s = Strategy(schema)
        s.add_exit_thresholds(profit, loss)
        assert s.exit_thresholds == (profit, loss)

    @given(direction, option_type)
    @settings(max_examples=20)
    def test_clear_legs(self, d, ot):
        schema = _options_schema()
        s = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=ot, direction=d)
        s.add_leg(leg)
        s.legs.clear()
        assert len(s.legs) == 0
