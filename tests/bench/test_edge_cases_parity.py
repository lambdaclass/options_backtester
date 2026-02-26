"""Edge cases that might cause divergence between Rust and Python."""

from __future__ import annotations

import numpy as np
import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    ivy_stocks,
    load_small_stocks,
    load_small_options,
    load_large_stocks,
    load_large_options,
    buy_put_strategy,
    buy_call_strategy,
    sell_put_strategy,
    run_rust,
    run_python,
    assert_parity,
    assert_balance_close,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


class TestAllocationEdgeCases:
    """Extreme allocation ratios."""

    def test_zero_options_allocation(self):
        alloc = {"stocks": 0.95, "options": 0.0, "cash": 0.05}
        py = run_python(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        rs = run_rust(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="0%options")
        # No option trades expected
        assert py.trade_log.empty
        assert rs.trade_log.empty

    def test_high_options_allocation(self):
        alloc = {"stocks": 0.05, "options": 0.90, "cash": 0.05}
        py = run_python(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        rs = run_rust(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="90%options")

    def test_minimal_stocks_allocation(self):
        alloc = {"stocks": 0.01, "options": 0.49, "cash": 0.50}
        py = run_python(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        rs = run_rust(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="1%stocks")


class TestCapitalEdgeCases:
    """Extreme capital amounts and rounding behavior."""

    def test_small_capital(self):
        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, 1_000, buy_put_strategy)
        rs = run_rust(alloc, 1_000, buy_put_strategy)
        assert_parity(py, rs, label="$1K")

    def test_large_capital(self):
        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, 100_000_000, buy_put_strategy)
        rs = run_rust(alloc, 100_000_000, buy_put_strategy)
        assert_parity(py, rs, label="$100M")


class TestRebalanceEdgeCases:
    """Different rebalance frequencies."""

    def test_high_rebalance_freq(self):
        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, DEFAULT_CAPITAL, buy_put_strategy,
                        rebalance_freq=5)
        rs = run_rust(alloc, DEFAULT_CAPITAL, buy_put_strategy,
                      rebalance_freq=5)
        assert_parity(py, rs, label="rebalance_freq=5")


class TestDirectionAndTypeCases:
    """Non-default leg directions and types."""

    def test_sell_direction_single_leg(self):
        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, DEFAULT_CAPITAL, sell_put_strategy)
        rs = run_rust(alloc, DEFAULT_CAPITAL, sell_put_strategy)
        assert_parity(py, rs, label="SellPut")

    def test_call_only_strategy(self):
        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, DEFAULT_CAPITAL, buy_call_strategy)
        rs = run_rust(alloc, DEFAULT_CAPITAL, buy_call_strategy)
        assert_parity(py, rs, label="BuyCall")


class TestSMAGating:
    """SMA gating must work identically in both paths."""

    def test_sma_gating_parity(self):
        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, DEFAULT_CAPITAL, buy_put_strategy,
                        sma_days=50)
        rs = run_rust(alloc, DEFAULT_CAPITAL, buy_put_strategy,
                      sma_days=50)
        assert_parity(py, rs, label="sma_days=50")


class TestStockOrderingParity:
    """Stock symbol ordering must not affect allocation.

    Regression test: Python's _buy_stocks previously used data-order prices
    with user-order percentages, causing misallocated stock quantities when
    the data had symbols in a different order than the user's stock list.
    """

    def test_stock_ordering_with_large_dataset(self):
        """Large dataset has symbols in different order than the stock list."""
        from options_portfolio_backtester.core.types import Stock

        stocks = [
            Stock("VOO", 0.20), Stock("TLT", 0.20), Stock("EWY", 0.15),
            Stock("PDBC", 0.15), Stock("IAU", 0.10), Stock("VNQI", 0.10),
            Stock("VTIP", 0.10),
        ]
        alloc = {"stocks": 0.97, "options": 0.03, "cash": 0.0}
        py = run_python(
            alloc, DEFAULT_CAPITAL, buy_put_strategy,
            stocks=stocks,
            stocks_data=load_large_stocks(),
            options_data=load_large_options(),
        )
        rs = run_rust(
            alloc, DEFAULT_CAPITAL, buy_put_strategy,
            stocks=stocks,
            stocks_data=load_large_stocks(),
            options_data=load_large_options(),
        )
        assert_parity(py, rs, label="stock-ordering")
        assert_balance_close(py, rs, label="stock-ordering")

    def test_stock_capital_reasonable_on_first_day(self):
        """Stock capital on day 2 should be close to stocks_allocation (97% of 1M)."""
        from options_portfolio_backtester.core.types import Stock

        stocks = [
            Stock("VOO", 0.20), Stock("TLT", 0.20), Stock("EWY", 0.15),
            Stock("PDBC", 0.15), Stock("IAU", 0.10), Stock("VNQI", 0.10),
            Stock("VTIP", 0.10),
        ]
        alloc = {"stocks": 0.97, "options": 0.03, "cash": 0.0}
        eng = run_python(
            alloc, DEFAULT_CAPITAL, buy_put_strategy,
            stocks=stocks,
            stocks_data=load_large_stocks(),
            options_data=load_large_options(),
        )
        stocks_cap_day2 = eng.balance["stocks capital"].iloc[1]
        expected = 0.97 * DEFAULT_CAPITAL
        # Stock capital should be within 5% of allocation (price movement in one day)
        assert abs(stocks_cap_day2 - expected) / expected < 0.05, (
            f"stocks_capital={stocks_cap_day2:.0f} too far from "
            f"allocation={expected:.0f}"
        )


class TestFixedOptionsBudget:
    """options_budget should limit options allocation to the fixed amount."""

    def test_fixed_budget_parity(self):
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)

        # Now with a fixed budget much smaller than allocation_options * capital
        from options_portfolio_backtester.engine.engine import BacktestEngine
        import options_portfolio_backtester.engine._dispatch as _dispatch

        def _run_with_budget(force_python):
            eng = BacktestEngine(DEFAULT_ALLOC, initial_capital=DEFAULT_CAPITAL)
            eng.stocks = ivy_stocks()
            eng.stocks_data = load_small_stocks()
            eng.options_data = load_small_options()
            eng.options_strategy = buy_put_strategy(eng.options_data.schema)
            eng.options_budget = 5_000.0
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

        py_budget = _run_with_budget(force_python=True)
        rs_budget = _run_with_budget(force_python=False)
        assert_parity(py_budget, rs_budget, label="fixed_budget")

        # With budget=5000, qty should be smaller than without budget
        # (default allocation = 0.3 * 1M = 300K >> 5000)
        if not py_budget.trade_log.empty and not py.trade_log.empty:
            budget_qty = py_budget.trade_log["totals"]["qty"].iloc[0]
            default_qty = py.trade_log["totals"]["qty"].iloc[0]
            assert budget_qty < default_qty, (
                f"Fixed budget qty ({budget_qty}) should be less than "
                f"default ({default_qty})"
            )


class TestLargeDatasetIntegerColumns:
    """Large dataset has integer strike/volume columns; Rust must handle them."""

    def test_integer_strike_column(self):
        """strike column is i64 in large dataset; Rust must cast to f64."""
        from options_portfolio_backtester.core.types import Stock

        stocks = [
            Stock("VOO", 0.20), Stock("TLT", 0.20), Stock("EWY", 0.15),
            Stock("PDBC", 0.15), Stock("IAU", 0.10), Stock("VNQI", 0.10),
            Stock("VTIP", 0.10),
        ]
        alloc = {"stocks": 0.97, "options": 0.03, "cash": 0.0}

        # Should not raise; previously failed with
        # "invalid series dtype: expected Float64, got i64 for strike"
        rs = run_rust(
            alloc, DEFAULT_CAPITAL, buy_put_strategy,
            stocks=stocks,
            stocks_data=load_large_stocks(),
            options_data=load_large_options(),
        )
        assert rs.run_metadata.get("dispatch_mode") == "rust-full"
        assert not rs.trade_log.empty


class TestNoMatchingEntries:
    """Tight filter that matches nothing: both paths should gracefully
    return empty trade logs."""

    def test_no_matching_entries(self):
        from options_portfolio_backtester.strategy.strategy import Strategy
        from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
        from options_portfolio_backtester.core.types import OptionType as Type, Direction

        def tight_strat(schema):
            s = Strategy(schema)
            leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                              direction=Direction.BUY)
            # DTE range that doesn't exist in test data
            leg.entry_filter = (
                (schema.underlying == "SPX")
                & (schema.dte >= 9999)
            )
            leg.exit_filter = schema.dte <= 30
            s.add_legs([leg])
            return s

        alloc = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
        py = run_python(alloc, DEFAULT_CAPITAL, tight_strat)
        rs = run_rust(alloc, DEFAULT_CAPITAL, tight_strat)
        assert_parity(py, rs, label="NoMatch")
        assert py.trade_log.empty
        assert rs.trade_log.empty
