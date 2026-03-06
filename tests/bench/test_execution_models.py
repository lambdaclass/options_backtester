"""Regression tests for execution models (cost, fill, signal, risk, exits)."""

from __future__ import annotations

import pytest

from tests.bench._test_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    buy_put_strategy,
    strangle_strategy,
    run_backtest,
    assert_invariants,
    make_cost_model,
    make_fill_model,
    make_signal_selector,
)

pytestmark = [
    pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed"),
    pytest.mark.bench,
]


# ── Cost models ───────────────────────────────────────────────────────

class TestCostModels:
    @pytest.mark.parametrize("cost_name", ["NoCosts", "PerContract", "Tiered"])
    def test_cost_model(self, cost_name):
        eng = run_backtest(cost_model=make_cost_model(cost_name))
        assert_invariants(eng, min_trades=1, label=cost_name)


# ── Fill models ───────────────────────────────────────────────────────

class TestFillModels:
    @pytest.mark.parametrize("fill_name", ["MarketAtBidAsk", "MidPrice"])
    def test_fill_model(self, fill_name):
        eng = run_backtest(fill_model=make_fill_model(fill_name))
        assert_invariants(eng, min_trades=1, label=fill_name)


# ── Signal selectors ─────────────────────────────────────────────────

class TestSignalSelectors:
    @pytest.mark.parametrize("signal_name", ["FirstMatch", "NearestDelta", "MaxOpenInterest"])
    def test_signal_selector(self, signal_name):
        eng = run_backtest(signal_selector=make_signal_selector(signal_name))
        assert_invariants(eng, label=signal_name)


# ── Risk constraints ─────────────────────────────────────────────────

class TestRiskConstraints:
    def test_max_delta(self):
        from options_portfolio_backtester.portfolio.risk import RiskManager, MaxDelta
        rm = RiskManager()
        rm.add_constraint(MaxDelta(limit=100))
        eng = run_backtest(risk_manager=rm)
        assert_invariants(eng)

    def test_max_drawdown(self):
        from options_portfolio_backtester.portfolio.risk import RiskManager, MaxDrawdown
        rm = RiskManager()
        rm.add_constraint(MaxDrawdown(max_dd_pct=0.20))
        eng = run_backtest(risk_manager=rm)
        assert_invariants(eng)


# ── Exit thresholds ──────────────────────────────────────────────────

class TestExitThresholds:
    def test_profit_exit(self):
        from tests.bench._test_helpers import (
            _make_engine, load_small_stocks, load_small_options, ivy_stocks,
        )
        eng = _make_engine(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, ivy_stocks(),
            load_small_stocks(), load_small_options(), buy_put_strategy,
        )
        eng.profit_target = 0.5
        eng.run(rebalance_freq=1, rebalance_unit="BMS")
        assert_invariants(eng)

    def test_loss_exit(self):
        from tests.bench._test_helpers import (
            _make_engine, load_small_stocks, load_small_options, ivy_stocks,
        )
        eng = _make_engine(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, ivy_stocks(),
            load_small_stocks(), load_small_options(), buy_put_strategy,
        )
        eng.stop_loss = 0.3
        eng.run(rebalance_freq=1, rebalance_unit="BMS")
        assert_invariants(eng)


# ── Full model grid (3 x 2 x 3 = 18 combos) ────────────────────────

class TestModelGrid:
    @pytest.mark.parametrize("cost_name", ["NoCosts", "PerContract", "Tiered"])
    @pytest.mark.parametrize("fill_name", ["MarketAtBidAsk", "MidPrice"])
    @pytest.mark.parametrize("signal_name", ["FirstMatch", "NearestDelta", "MaxOpenInterest"])
    def test_model_combo(self, cost_name, fill_name, signal_name):
        eng = run_backtest(
            cost_model=make_cost_model(cost_name),
            fill_model=make_fill_model(fill_name),
            signal_selector=make_signal_selector(signal_name),
        )
        assert_invariants(eng, label=f"{cost_name}_{fill_name}_{signal_name}")
