"""Heavy Hypothesis fuzz tests — maximum parity coverage.

Every parameter randomized: risk constraints, multi-leg combos, sell direction,
per-leg overrides, rebalance frequency, SMA days, fixed budgets, exit thresholds,
and the kitchen sink (everything at once).

All tests use assert_parity which includes full balance timeseries + trade log
row-level content verification.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    buy_put_strategy,
    buy_call_strategy,
    sell_put_strategy,
    sell_call_strategy,
    buy_put_spread_strategy,
    sell_call_spread_strategy,
    strangle_strategy,
    straddle_strategy,
    two_leg_strategy,
    run_rust,
    run_python,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)

# ── Shared Hypothesis strategies ──────────────────────────────────────

alloc_pct = st.floats(min_value=0.05, max_value=0.90,
                      allow_nan=False, allow_infinity=False)
capital_st = st.integers(min_value=10_000, max_value=10_000_000)
positive_rate = st.floats(min_value=0.01, max_value=10.0,
                          allow_nan=False, allow_infinity=False)
stock_rate = st.floats(min_value=0.001, max_value=0.05,
                       allow_nan=False, allow_infinity=False)
pct_floats = st.floats(min_value=0.01, max_value=5.0,
                       allow_nan=False, allow_infinity=False)
delta_target = st.floats(min_value=-0.90, max_value=-0.05,
                         allow_nan=False, allow_infinity=False)
volume_threshold = st.integers(min_value=1, max_value=1000)
cost_model_choice = st.sampled_from(["NoCosts", "PerContract", "Tiered"])
fill_model_choice = st.sampled_from(["MarketAtBidAsk", "MidPrice", "VolumeAware"])
selector_choice = st.sampled_from(["FirstMatch", "NearestDelta"])
direction_st = st.sampled_from(["buy", "sell"])
option_type_st = st.sampled_from(["put", "call"])


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


# ── 5a. Risk Constraint Fuzz ─────────────────────────────────────────

class TestRiskConstraintFuzz:

    @given(limit=st.floats(min_value=10.0, max_value=500.0,
                           allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_max_delta_random_limits(self, limit):
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxDelta,
        )
        rm = RiskManager([MaxDelta(limit=limit)])
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label=f"MaxDelta({limit:.1f})")

    @given(limit=st.floats(min_value=1.0, max_value=200.0,
                           allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_max_vega_random_limits(self, limit):
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxVega,
        )
        rm = RiskManager([MaxVega(limit=limit)])
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label=f"MaxVega({limit:.1f})")

    @given(max_dd_pct=st.floats(min_value=0.01, max_value=0.99,
                                allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_max_drawdown_random_pcts(self, max_dd_pct):
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxDrawdown,
        )
        rm = RiskManager([MaxDrawdown(max_dd_pct=max_dd_pct)])
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label=f"MaxDrawdown({max_dd_pct:.2f})")

    @given(data=st.data())
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_combined_random_constraints(self, data):
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxDelta, MaxVega, MaxDrawdown,
        )
        constraints = []
        if data.draw(st.booleans()):
            constraints.append(MaxDelta(
                limit=data.draw(st.floats(10.0, 500.0,
                                          allow_nan=False, allow_infinity=False))
            ))
        if data.draw(st.booleans()):
            constraints.append(MaxVega(
                limit=data.draw(st.floats(10.0, 200.0,
                                          allow_nan=False, allow_infinity=False))
            ))
        if data.draw(st.booleans()):
            constraints.append(MaxDrawdown(
                max_dd_pct=data.draw(st.floats(0.05, 0.99,
                                               allow_nan=False, allow_infinity=False))
            ))
        assume(len(constraints) > 0)
        rm = RiskManager(constraints)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label="combined_risk")


# ── 5b. Multi-Leg Fuzz ───────────────────────────────────────────────

class TestMultiLegFuzz:

    @given(
        dir1=direction_st, type1=option_type_st,
        dir2=direction_st, type2=option_type_st,
        capital=capital_st,
        stocks_pct=alloc_pct,
    )
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_random_two_leg_strategy(self, dir1, type1, dir2, type2,
                                     capital, stocks_pct):
        options_pct = min(0.45, 1.0 - stocks_pct - 0.05)
        assume(options_pct > 0.05)
        cash_pct = 1.0 - stocks_pct - options_pct
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}

        def strat(schema):
            return two_leg_strategy(schema, dir1, type1, dir2, type2)

        py = run_python(alloc, capital, strat)
        rs = run_rust(alloc, capital, strat)
        assert_parity(py, rs,
                      label=f"2leg({dir1}_{type1},{dir2}_{type2})")


# ── 5c. Sell Direction Fuzz ───────────────────────────────────────────

class TestSellDirectionFuzz:

    @given(
        capital=capital_st,
        stocks_pct=alloc_pct,
        options_pct=st.floats(min_value=0.05, max_value=0.50,
                              allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_sell_put_random_params(self, capital, stocks_pct, options_pct):
        assume(stocks_pct + options_pct <= 0.98)
        cash_pct = 1.0 - stocks_pct - options_pct
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}
        py = run_python(alloc, capital, sell_put_strategy)
        rs = run_rust(alloc, capital, sell_put_strategy)
        assert_parity(py, rs, label="sell_put_fuzz")

    @given(
        capital=capital_st,
        stocks_pct=alloc_pct,
        options_pct=st.floats(min_value=0.05, max_value=0.50,
                              allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_sell_call_random_params(self, capital, stocks_pct, options_pct):
        assume(stocks_pct + options_pct <= 0.98)
        cash_pct = 1.0 - stocks_pct - options_pct
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}
        py = run_python(alloc, capital, sell_call_strategy)
        rs = run_rust(alloc, capital, sell_call_strategy)
        assert_parity(py, rs, label="sell_call_fuzz")

    @given(
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        ss_choice=selector_choice,
        data=st.data(),
    )
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_sell_with_random_models(self, cm_choice, fm_choice, ss_choice, data):
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        ss = _make_selector(ss_choice, data)
        strategy_fn = data.draw(st.sampled_from([sell_put_strategy, sell_call_strategy]))
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                        cost_model=cm, fill_model=fm, signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                      cost_model=cm, fill_model=fm, signal_selector=ss)
        assert_parity(py, rs,
                      label=f"sell_models({cm_choice},{fm_choice},{ss_choice})")


# ── 5d. Per-Leg Override Fuzz ─────────────────────────────────────────

class TestPerLegOverrideFuzz:

    @given(target=delta_target)
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_per_leg_nearest_delta_random(self, target):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )

        def strat(schema):
            s = buy_put_strategy(schema)
            s.legs[0].signal_selector = NearestDelta(target_delta=target)
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label=f"per_leg_delta({target:.3f})")

    @given(fm_choice=fill_model_choice, data=st.data())
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_per_leg_fill_model_random(self, fm_choice, data):
        fm = _make_fill_model(fm_choice, data)

        def strat(schema):
            s = buy_put_strategy(schema)
            s.legs[0].fill_model = fm
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label=f"per_leg_fill({fm_choice})")


# ── 5e. Rebalance Frequency Fuzz ─────────────────────────────────────

class TestRebalanceFreqFuzz:

    @given(freq=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_random_rebalance_freq(self, freq):
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        rebalance_freq=freq)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      rebalance_freq=freq)
        assert_parity(py, rs, label=f"rebalance_freq={freq}")


# ── 5f. SMA Days Fuzz ────────────────────────────────────────────────

class TestSMAFuzz:

    @given(sma_days=st.integers(min_value=5, max_value=200))
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_random_sma_days(self, sma_days):
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        sma_days=sma_days)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      sma_days=sma_days)
        assert_parity(py, rs, label=f"sma_days={sma_days}")

    @given(
        sma_days=st.integers(min_value=5, max_value=200),
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        data=st.data(),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_sma_with_random_models(self, sma_days, cm_choice, fm_choice, data):
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        sma_days=sma_days, cost_model=cm, fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      sma_days=sma_days, cost_model=cm, fill_model=fm)
        assert_parity(py, rs,
                      label=f"sma({sma_days},{cm_choice},{fm_choice})")


# ── 5g. Fixed Options Budget Fuzz ─────────────────────────────────────

class TestFixedBudgetFuzz:

    @given(budget=st.floats(min_value=1_000, max_value=500_000,
                            allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_fixed_budget_parity(self, budget):
        from options_portfolio_backtester.engine.engine import BacktestEngine

        def _run(force_python=False):
            import options_portfolio_backtester.engine._dispatch as _dispatch
            eng = BacktestEngine(DEFAULT_ALLOC, initial_capital=DEFAULT_CAPITAL)
            eng.stocks = __import__("tests.bench._parity_helpers",
                                    fromlist=["ivy_stocks"]).ivy_stocks()
            eng.stocks_data = __import__("tests.bench._parity_helpers",
                                         fromlist=["load_small_stocks"]).load_small_stocks()
            eng.options_data = __import__("tests.bench._parity_helpers",
                                          fromlist=["load_small_options"]).load_small_options()
            eng.options_strategy = buy_put_strategy(eng.options_data.schema)
            eng.options_budget = budget
            if force_python:
                orig = _dispatch.RUST_AVAILABLE
                try:
                    _dispatch.RUST_AVAILABLE = False
                    eng.run(rebalance_freq=1)
                finally:
                    _dispatch.RUST_AVAILABLE = orig
            else:
                eng.run(rebalance_freq=1)
            return eng

        py = _run(force_python=True)
        rs = _run(force_python=False)
        assert_parity(py, rs, label=f"fixed_budget({budget:.0f})")

    @given(
        budget=st.floats(min_value=1_000, max_value=500_000,
                         allow_nan=False, allow_infinity=False),
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        data=st.data(),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_fixed_budget_with_models(self, budget, cm_choice, fm_choice, data):
        from options_portfolio_backtester.engine.engine import BacktestEngine
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)

        def _run(force_python=False):
            import options_portfolio_backtester.engine._dispatch as _dispatch
            eng = BacktestEngine(DEFAULT_ALLOC, initial_capital=DEFAULT_CAPITAL,
                                 cost_model=cm, fill_model=fm)
            eng.stocks = __import__("tests.bench._parity_helpers",
                                    fromlist=["ivy_stocks"]).ivy_stocks()
            eng.stocks_data = __import__("tests.bench._parity_helpers",
                                         fromlist=["load_small_stocks"]).load_small_stocks()
            eng.options_data = __import__("tests.bench._parity_helpers",
                                          fromlist=["load_small_options"]).load_small_options()
            eng.options_strategy = buy_put_strategy(eng.options_data.schema)
            eng.options_budget = budget
            if force_python:
                orig = _dispatch.RUST_AVAILABLE
                try:
                    _dispatch.RUST_AVAILABLE = False
                    eng.run(rebalance_freq=1)
                finally:
                    _dispatch.RUST_AVAILABLE = orig
            else:
                eng.run(rebalance_freq=1)
            return eng

        py = _run(force_python=True)
        rs = _run(force_python=False)
        assert_parity(py, rs,
                      label=f"fixed_budget({budget:.0f},{cm_choice},{fm_choice})")


# ── 5h. Exit Threshold + Model Combos Fuzz ────────────────────────────

class TestExitThresholdModelFuzz:

    @given(
        profit_pct=st.one_of(st.none(), pct_floats),
        loss_pct=st.one_of(st.none(), pct_floats),
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        ss_choice=selector_choice,
        data=st.data(),
    )
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_thresholds_with_random_models(self, profit_pct, loss_pct,
                                           cm_choice, fm_choice, ss_choice, data):
        assume(profit_pct is not None or loss_pct is not None)
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        ss = _make_selector(ss_choice, data)

        def strat(schema):
            s = buy_put_strategy(schema)
            s.add_exit_thresholds(
                profit_pct if profit_pct is not None else math.inf,
                loss_pct if loss_pct is not None else math.inf,
            )
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat,
                        cost_model=cm, fill_model=fm, signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat,
                      cost_model=cm, fill_model=fm, signal_selector=ss)
        assert_parity(py, rs,
                      label=f"thresh_model(p={profit_pct},l={loss_pct},"
                            f"{cm_choice},{fm_choice},{ss_choice})")


# ── 5i. Kitchen Sink Fuzz (everything random at once) ─────────────────

class TestKitchenSinkFuzz:

    @given(
        stocks_pct=alloc_pct,
        options_pct=st.floats(min_value=0.05, max_value=0.50,
                              allow_nan=False, allow_infinity=False),
        capital=capital_st,
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        ss_choice=selector_choice,
        profit_pct=st.one_of(st.none(), pct_floats),
        loss_pct=st.one_of(st.none(), pct_floats),
        sma_days=st.one_of(st.none(), st.integers(min_value=5, max_value=200)),
        rebalance_freq=st.integers(min_value=1, max_value=5),
        direction=direction_st,
        option_type=option_type_st,
        data=st.data(),
    )
    @settings(max_examples=100, deadline=60000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_everything_random(self, stocks_pct, options_pct, capital,
                               cm_choice, fm_choice, ss_choice,
                               profit_pct, loss_pct, sma_days,
                               rebalance_freq, direction, option_type, data):
        assume(stocks_pct + options_pct <= 0.98)
        cash_pct = 1.0 - stocks_pct - options_pct
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}

        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        ss = _make_selector(ss_choice, data)

        # Optional risk constraints
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxDelta, MaxVega, MaxDrawdown,
        )
        constraints = []
        if data.draw(st.booleans()):
            constraints.append(MaxDelta(
                limit=data.draw(st.floats(10.0, 500.0,
                                          allow_nan=False, allow_infinity=False))
            ))
        if data.draw(st.booleans()):
            constraints.append(MaxDrawdown(
                max_dd_pct=data.draw(st.floats(0.05, 0.50,
                                               allow_nan=False, allow_infinity=False))
            ))
        rm = RiskManager(constraints) if constraints else RiskManager()

        strategy_map = {
            ("buy", "put"): buy_put_strategy,
            ("buy", "call"): buy_call_strategy,
            ("sell", "put"): sell_put_strategy,
            ("sell", "call"): sell_call_strategy,
        }
        base_strategy_fn = strategy_map[(direction, option_type)]

        def strat(schema):
            s = base_strategy_fn(schema)
            if profit_pct is not None or loss_pct is not None:
                s.add_exit_thresholds(
                    profit_pct if profit_pct is not None else math.inf,
                    loss_pct if loss_pct is not None else math.inf,
                )
            return s

        py = run_python(alloc, capital, strat,
                        rebalance_freq=rebalance_freq,
                        sma_days=sma_days,
                        cost_model=cm, fill_model=fm,
                        signal_selector=ss, risk_manager=rm)
        rs = run_rust(alloc, capital, strat,
                      rebalance_freq=rebalance_freq,
                      sma_days=sma_days,
                      cost_model=cm, fill_model=fm,
                      signal_selector=ss, risk_manager=rm)
        assert_parity(py, rs, label="kitchen_sink")


# ── 5j. Typical Strategies Fuzz ───────────────────────────────────────

class TestTypicalStrategyFuzz:
    """Fuzz typical multi-leg strategies with random parameters."""

    @given(
        capital=capital_st,
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        data=st.data(),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_put_spread_fuzz(self, capital, cm_choice, fm_choice, data):
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        py = run_python(DEFAULT_ALLOC, capital, buy_put_spread_strategy,
                        cost_model=cm, fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, capital, buy_put_spread_strategy,
                      cost_model=cm, fill_model=fm)
        assert_parity(py, rs, label="put_spread_fuzz")

    @given(
        capital=capital_st,
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        data=st.data(),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_strangle_fuzz(self, capital, cm_choice, fm_choice, data):
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        py = run_python(DEFAULT_ALLOC, capital, strangle_strategy,
                        cost_model=cm, fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, capital, strangle_strategy,
                      cost_model=cm, fill_model=fm)
        assert_parity(py, rs, label="strangle_fuzz")

    @given(
        capital=capital_st,
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        data=st.data(),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_straddle_fuzz(self, capital, cm_choice, fm_choice, data):
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        py = run_python(DEFAULT_ALLOC, capital, straddle_strategy,
                        cost_model=cm, fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, capital, straddle_strategy,
                      cost_model=cm, fill_model=fm)
        assert_parity(py, rs, label="straddle_fuzz")

    @given(
        capital=capital_st,
        cm_choice=cost_model_choice,
        fm_choice=fill_model_choice,
        data=st.data(),
    )
    @settings(max_examples=30, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_call_spread_fuzz(self, capital, cm_choice, fm_choice, data):
        cm = _make_cost_model(cm_choice, data)
        fm = _make_fill_model(fm_choice, data)
        py = run_python(DEFAULT_ALLOC, capital, sell_call_spread_strategy,
                        cost_model=cm, fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, capital, sell_call_spread_strategy,
                      cost_model=cm, fill_model=fm)
        assert_parity(py, rs, label="call_spread_fuzz")
