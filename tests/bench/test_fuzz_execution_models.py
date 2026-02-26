"""Hypothesis-based fuzzing across execution model combinations."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    buy_put_strategy,
    run_rust,
    run_python,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)

# ── Custom strategies ──────────────────────────────────────────────────

positive_rate = st.floats(min_value=0.01, max_value=10.0,
                          allow_nan=False, allow_infinity=False)
stock_rate = st.floats(min_value=0.001, max_value=0.05,
                       allow_nan=False, allow_infinity=False)
pct_floats = st.floats(min_value=0.01, max_value=5.0,
                        allow_nan=False, allow_infinity=False)
delta_target = st.floats(min_value=-0.90, max_value=-0.05,
                         allow_nan=False, allow_infinity=False)
volume_threshold = st.integers(min_value=1, max_value=1000)
alloc_pct = st.floats(min_value=0.05, max_value=0.90,
                      allow_nan=False, allow_infinity=False)
capital_st = st.integers(min_value=10_000, max_value=10_000_000)

DEFAULT_ALLOC = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
DEFAULT_CAPITAL = 1_000_000


# ── Cost Model Fuzz ────────────────────────────────────────────────────

class TestCostModelFuzz:
    """Random commission rates must produce identical results."""

    @given(rate=positive_rate, sr=stock_rate)
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_per_contract_random_rates(self, rate, sr):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=rate, stock_rate=sr)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        assert_parity(py, rs, label=f"PerContract({rate:.2f},{sr:.4f})")

    @given(
        tier1_rate=positive_rate,
        tier2_rate=positive_rate,
        sr=stock_rate,
    )
    @settings(max_examples=20, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_tiered_random_rates(self, tier1_rate, tier2_rate, sr):
        from options_portfolio_backtester.execution.cost_model import (
            TieredCommission,
        )
        tiers = [(10_000, tier1_rate), (100_000, tier2_rate)]
        cm = TieredCommission(tiers=tiers, stock_rate=sr)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        assert_parity(py, rs, label=f"Tiered({tier1_rate:.2f},{tier2_rate:.2f})")


# ── Fill Model Fuzz ────────────────────────────────────────────────────

class TestFillModelFuzz:
    """Random volume thresholds must produce identical results."""

    @given(threshold=volume_threshold)
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_volume_aware_random_threshold(self, threshold):
        from options_portfolio_backtester.execution.fill_model import (
            VolumeAwareFill,
        )
        fm = VolumeAwareFill(full_volume_threshold=threshold)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      fill_model=fm)
        assert_parity(py, rs, label=f"VolumeAware({threshold})")


# ── Signal Selector Fuzz ───────────────────────────────────────────────

class TestSignalSelectorFuzz:
    """Random delta targets must produce identical results."""

    @given(target=delta_target)
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_nearest_delta_random_target(self, target):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        ss = NearestDelta(target_delta=target)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      signal_selector=ss)
        assert_parity(py, rs, label=f"NearestDelta({target:.3f})")


# ── Exit Threshold Fuzz ────────────────────────────────────────────────

class TestExitThresholdFuzz:
    """Random exit thresholds must produce identical results."""

    @given(
        profit_pct=st.one_of(st.none(), pct_floats),
        loss_pct=st.one_of(st.none(), pct_floats),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_random_thresholds(self, profit_pct, loss_pct):
        import math
        assume(profit_pct is not None or loss_pct is not None)

        def strat(schema):
            s = buy_put_strategy(schema)
            s.add_exit_thresholds(
                profit_pct if profit_pct is not None else math.inf,
                loss_pct if loss_pct is not None else math.inf,
            )
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs,
                      label=f"thresholds(p={profit_pct},l={loss_pct})")


# ── Combined Model Fuzz (the big one) ─────────────────────────────────

cost_model_choice = st.sampled_from(["NoCosts", "PerContract", "Tiered"])
fill_model_choice = st.sampled_from(["MarketAtBidAsk", "MidPrice", "VolumeAware"])
selector_choice = st.sampled_from(["FirstMatch", "NearestDelta"])


def _make_cost_model(choice, data):
    from options_portfolio_backtester.execution.cost_model import (
        NoCosts, PerContractCommission, TieredCommission,
    )
    if choice == "NoCosts":
        return NoCosts()
    elif choice == "PerContract":
        rate = data.draw(positive_rate)
        return PerContractCommission(rate=rate)
    else:
        r1 = data.draw(positive_rate)
        r2 = data.draw(positive_rate)
        return TieredCommission(tiers=[(10_000, r1), (100_000, r2)])


def _make_fill_model(choice, data):
    from options_portfolio_backtester.execution.fill_model import (
        MarketAtBidAsk, MidPrice, VolumeAwareFill,
    )
    if choice == "MarketAtBidAsk":
        return MarketAtBidAsk()
    elif choice == "MidPrice":
        return MidPrice()
    else:
        threshold = data.draw(volume_threshold)
        return VolumeAwareFill(full_volume_threshold=threshold)


def _make_selector(choice, data):
    from options_portfolio_backtester.execution.signal_selector import (
        FirstMatch, NearestDelta,
    )
    if choice == "FirstMatch":
        return FirstMatch()
    else:
        target = data.draw(delta_target)
        return NearestDelta(target_delta=target)


class TestAllocationModelFuzz:
    """Full combinatorial fuzz: random allocations + random model combos."""

    @given(
        stocks_pct=alloc_pct,
        options_pct=st.floats(min_value=0.05, max_value=0.50,
                              allow_nan=False, allow_infinity=False),
        capital=capital_st,
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        ss_choice=selector_choice,
        data=st.data(),
    )
    @settings(max_examples=50, deadline=60000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_combined_random(self, stocks_pct, options_pct, capital,
                             cm_choice, fm_choice, ss_choice, data):
        assume(stocks_pct + options_pct <= 0.98)
        cash_pct = 1.0 - stocks_pct - options_pct
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}

        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        ss = _make_selector(ss_choice, data)

        py = run_python(alloc, capital, buy_put_strategy,
                        cost_model=cm, fill_model=fm, signal_selector=ss)
        rs = run_rust(alloc, capital, buy_put_strategy,
                      cost_model=cm, fill_model=fm, signal_selector=ss)
        assert_parity(
            py, rs,
            label=f"combined({cm_choice},{fm_choice},{ss_choice},"
                  f"cap={capital})",
        )
