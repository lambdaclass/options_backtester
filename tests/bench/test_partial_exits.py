"""Regression tests for partial exit (sell_some_options) scenarios.

High options allocation (85%) forces sell_some_options to trigger when
mark-to-market changes shift the allocation balance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.bench._test_helpers import (
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
    strangle_strategy,
    run_backtest,
    assert_invariants,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)

HIGH_OPTIONS_ALLOC = {"stocks": 0.05, "options": 0.85, "cash": 0.10}


class TestPartialExitGenerated:

    def _run(self, strategy_fn):
        return run_backtest(
            alloc=HIGH_OPTIONS_ALLOC, capital=DEFAULT_CAPITAL,
            strategy_fn=strategy_fn,
            stocks=generated_stocks(),
            stocks_data=load_generated_stocks(),
            options_data=load_generated_options(),
        )

    def test_buy_put(self):
        eng = self._run(buy_put_strategy)
        assert_invariants(eng, min_trades=3, label="partial_buy_put")

    def test_sell_put(self):
        eng = self._run(sell_put_strategy)
        assert_invariants(eng, min_trades=3, label="partial_sell_put",
                          allow_negative_capital=True)

    def test_strangle(self):
        eng = self._run(strangle_strategy)
        assert_invariants(eng, label="partial_strangle",
                          allow_negative_capital=True)


class TestPartialExitProduction:

    def _run(self, strategy_fn):
        return run_backtest(
            alloc=HIGH_OPTIONS_ALLOC, capital=DEFAULT_CAPITAL,
            strategy_fn=lambda schema: strategy_fn(schema, underlying="SPY"),
            stocks=prod_spy_stocks(),
            stocks_data=load_prod_stocks(),
            options_data=load_prod_options(),
        )

    def test_buy_put_spy(self):
        eng = self._run(buy_put_strategy)
        assert_invariants(eng, label="partial_prod_buy_put")

    def test_sell_put_spy(self):
        eng = self._run(sell_put_strategy)
        assert_invariants(eng, label="partial_prod_sell_put",
                          allow_negative_capital=True)


class TestPartialExitCashAccounting:

    def test_cash_never_deeply_negative(self):
        eng = run_backtest(
            alloc=HIGH_OPTIONS_ALLOC, capital=DEFAULT_CAPITAL,
            strategy_fn=buy_put_strategy,
            stocks=generated_stocks(),
            stocks_data=load_generated_stocks(),
            options_data=load_generated_options(),
        )
        # High options allocation can cause cash to go moderately negative
        # due to timing of mark-to-market and rebalancing
        cash = eng.balance["cash"]
        assert (cash >= -50_000.0).all(), f"Cash deeply negative: min={cash.min()}"
