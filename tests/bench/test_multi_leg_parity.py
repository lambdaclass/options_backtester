"""Multi-leg strategy parity tests: Rust vs Python must match."""

from __future__ import annotations

import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    two_leg_strategy,
    run_rust,
    run_python,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# ── Strangle (2-leg same direction) ────────────────────────────────────

class TestStrangleParity:
    """2-leg strategies with same direction on both legs."""

    def test_buy_strangle_parity(self):
        def strat(schema):
            return two_leg_strategy(schema, "buy", "call", "buy", "put")
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label="BuyStrangle")

    def test_sell_strangle_parity(self):
        def strat(schema):
            return two_leg_strategy(schema, "sell", "call", "sell", "put")
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label="SellStrangle")


# ── Mixed Direction (2-leg) ────────────────────────────────────────────

class TestMixedDirectionParity:
    """2-leg strategies with opposing directions."""

    def test_buy_put_sell_call_parity(self):
        def strat(schema):
            return two_leg_strategy(schema, "buy", "put", "sell", "call")
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label="BuyPut+SellCall")

    def test_sell_put_buy_call_parity(self):
        def strat(schema):
            return two_leg_strategy(schema, "sell", "put", "buy", "call")
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label="SellPut+BuyCall")


# ── Per-Leg Override Parity ────────────────────────────────────────────

class TestPerLegOverrideParity:
    """Per-leg signal_selector and fill_model overrides."""

    def test_per_leg_signal_selector_parity(self):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        from options_portfolio_backtester.strategy.strategy import Strategy
        from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
        from options_portfolio_backtester.core.types import OptionType as Type, Direction

        def strat(schema):
            s = Strategy(schema)
            leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                              direction=Direction.BUY)
            leg.entry_filter = (
                (schema.underlying == "SPX") & (schema.dte >= 60)
            )
            leg.exit_filter = schema.dte <= 30
            leg.signal_selector = NearestDelta(target_delta=-0.30)
            s.add_legs([leg])
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label="PerLegNearestDelta")

    def test_per_leg_fill_model_parity(self):
        from options_portfolio_backtester.execution.fill_model import MidPrice
        from options_portfolio_backtester.strategy.strategy import Strategy
        from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
        from options_portfolio_backtester.core.types import OptionType as Type, Direction

        def strat(schema):
            s = Strategy(schema)
            leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                              direction=Direction.BUY)
            leg.entry_filter = (
                (schema.underlying == "SPX") & (schema.dte >= 60)
            )
            leg.exit_filter = schema.dte <= 30
            leg.fill_model = MidPrice()
            s.add_legs([leg])
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        assert_parity(py, rs, label="PerLegMidPrice")
