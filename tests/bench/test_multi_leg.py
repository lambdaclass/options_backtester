"""Multi-leg strategy regression tests.

Each test runs the backtest ONCE and checks invariants.
"""

from __future__ import annotations

import pytest

from tests.bench._test_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    strangle_strategy,
    straddle_strategy,
    buy_put_spread_strategy,
    sell_call_spread_strategy,
    two_leg_strategy,
    run_backtest,
    assert_invariants,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


class TestMultiLegStrategies:

    def test_strangle(self):
        eng = run_backtest(strategy_fn=strangle_strategy)
        assert_invariants(eng, allow_negative_capital=True)

    def test_straddle(self):
        eng = run_backtest(strategy_fn=straddle_strategy)
        assert_invariants(eng)

    def test_put_spread(self):
        eng = run_backtest(strategy_fn=buy_put_spread_strategy)
        assert_invariants(eng, allow_negative_capital=True)

    def test_call_spread(self):
        eng = run_backtest(strategy_fn=sell_call_spread_strategy)
        assert_invariants(eng, allow_negative_capital=True)


class TestMixedDirections:

    _COMBOS = [
        ("buy", "put", "sell", "call"),
        ("sell", "put", "buy", "call"),
        ("buy", "put", "buy", "call"),
        ("sell", "put", "sell", "call"),
    ]

    @pytest.mark.parametrize("d1,t1,d2,t2", _COMBOS)
    def test_direction_combo(self, d1, t1, d2, t2):
        eng = run_backtest(
            strategy_fn=lambda schema: two_leg_strategy(schema, d1, t1, d2, t2)
        )
        has_sell = "sell" in (d1, d2)
        assert_invariants(eng, label=f"{d1}_{t1}_{d2}_{t2}",
                          allow_negative_capital=has_sell)


class TestPerLegOverrides:

    def test_per_leg_signal_selector(self):
        from options_portfolio_backtester.execution.signal_selector import NearestDelta

        eng = run_backtest(
            strategy_fn=strangle_strategy,
            signal_selector=NearestDelta(target_delta=-0.30),
        )
        assert_invariants(eng, allow_negative_capital=True)

    def test_per_leg_fill_model(self):
        from options_portfolio_backtester.execution.fill_model import MidPrice

        eng = run_backtest(
            strategy_fn=strangle_strategy,
            fill_model=MidPrice(),
        )
        assert_invariants(eng, allow_negative_capital=True)
