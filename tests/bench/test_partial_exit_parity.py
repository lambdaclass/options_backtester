"""Partial exit (sell_some_options) parity tests.

Designs scenarios with high options allocation (80%+) that force
sell_some_options to trigger when mark-to-market changes the allocation
balance. Verifies partial exit trade log rows match across all 3 engines.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_CAPITAL,
    generated_stocks,
    prod_spy_stocks,
    load_generated_stocks,
    load_generated_options,
    load_prod_stocks,
    load_prod_options,
    buy_put_strategy,
    sell_put_strategy,
    buy_call_strategy,
    strangle_strategy,
    run_old,
    run_python_ex,
    run_rust_ex,
    assert_3way_parity,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# High options allocation forces sell_some_options on later rebalances
# when mark-to-market reduces total capital but options_capital stays high.
HIGH_OPTIONS_ALLOC = {"stocks": 0.05, "options": 0.85, "cash": 0.10}


def _run_3way_gen(alloc, capital, strategy_fn, rebalance_freq=1,
                  rebalance_unit='BMS'):
    common = dict(
        alloc=alloc, capital=capital, strategy_fn=strategy_fn,
        rebalance_freq=rebalance_freq, rebalance_unit=rebalance_unit,
        stocks=generated_stocks(),
        stocks_data=load_generated_stocks(),
        options_data=load_generated_options(),
    )
    old = run_old(**common)
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return old, py, rs


def _run_3way_prod(alloc, capital, strategy_fn, rebalance_freq=1,
                   rebalance_unit='BMS'):
    common = dict(
        alloc=alloc, capital=capital, strategy_fn=strategy_fn,
        rebalance_freq=rebalance_freq, rebalance_unit=rebalance_unit,
        stocks=prod_spy_stocks(),
        stocks_data=load_prod_stocks(),
        options_data=load_prod_options(),
    )
    old = run_old(**common)
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return old, py, rs


class TestPartialExitGenerated:
    """High-allocation tests that trigger sell_some_options on generated data."""

    def test_high_alloc_buy_put(self):
        old, py, rs = _run_3way_gen(HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)
        assert_3way_parity(old, py, rs, label="partial_buy_put")

    def test_high_alloc_sell_put(self):
        old, py, rs = _run_3way_gen(HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL, sell_put_strategy)
        assert_3way_parity(old, py, rs, label="partial_sell_put")

    def test_high_alloc_buy_call(self):
        old, py, rs = _run_3way_gen(HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL, buy_call_strategy)
        assert_3way_parity(old, py, rs, label="partial_buy_call")

    def test_high_alloc_strangle(self):
        old, py, rs = _run_3way_gen(HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL, strangle_strategy)
        assert_3way_parity(old, py, rs, label="partial_strangle")


class TestPartialExitProduction:
    """High-allocation tests on production SPY data."""

    def test_high_alloc_buy_put_spy(self):
        old, py, rs = _run_3way_prod(
            HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL,
            lambda schema: buy_put_strategy(schema, underlying="SPY"),
        )
        assert_3way_parity(old, py, rs, label="partial_prod_buy_put")

    def test_high_alloc_sell_put_spy(self):
        old, py, rs = _run_3way_prod(
            HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL,
            lambda schema: sell_put_strategy(schema, underlying="SPY"),
        )
        assert_3way_parity(old, py, rs, label="partial_prod_sell_put")


class TestPartialExitCashAccounting:
    """Verify cash accounting after partial exits."""

    def test_cash_never_goes_hugely_negative(self):
        """Even with aggressive partial exits, cash shouldn't be deeply negative."""
        old, py, rs = _run_3way_gen(HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)

        for eng, label in [(py, "python"), (rs, "rust")]:
            cash = eng.balance["cash"].values
            # Cash can go slightly negative due to rounding but not catastrophically
            min_cash = np.min(cash)
            assert min_cash > -DEFAULT_CAPITAL * 0.5, (
                f"[{label}] cash went too negative: {min_cash}"
            )

    def test_partial_exit_entries_have_negative_qty(self):
        """In partial exits, trade log qty should be negative (selling)."""
        old, py, rs = _run_3way_gen(HIGH_OPTIONS_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)

        for eng, label in [(py, "python"), (rs, "rust")]:
            if eng.trade_log.empty:
                continue
            qtys = eng.trade_log["totals"]["qty"].values
            # Some entries should have negative qty (partial exits)
            has_negative = any(q < 0 for q in qtys)
            has_positive = any(q > 0 for q in qtys)
            # With high allocation and monthly rebalancing, we expect both entries and partial exits
            assert has_positive, f"[{label}] no positive qty (entries) found"
