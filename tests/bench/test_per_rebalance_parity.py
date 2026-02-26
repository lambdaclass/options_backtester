"""Per-rebalance balance verification: Python vs Rust must match at EVERY date.

Catches compensating errors where final capital matches but intermediate
values diverge. Uses both generated and production datasets.

Monthly rebalancing (BMS) for confirmed parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    generated_stocks,
    prod_spy_stocks,
    load_generated_stocks,
    load_generated_options,
    load_prod_stocks,
    load_prod_options,
    buy_put_strategy,
    buy_call_strategy,
    sell_put_strategy,
    sell_call_strategy,
    strangle_strategy,
    straddle_strategy,
    buy_put_spread_strategy,
    run_python_ex,
    run_rust_ex,
    assert_per_rebalance_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


def _run_pair_gen(alloc, capital, strategy_fn, rebalance_freq=1,
                  rebalance_unit='BMS', sma_days=None):
    """Run Python + Rust on generated data."""
    common = dict(
        alloc=alloc, capital=capital, strategy_fn=strategy_fn,
        rebalance_freq=rebalance_freq, rebalance_unit=rebalance_unit,
        stocks=generated_stocks(),
        stocks_data=load_generated_stocks(),
        options_data=load_generated_options(),
        sma_days=sma_days,
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


def _run_pair_prod(alloc, capital, strategy_fn, rebalance_freq=1,
                   rebalance_unit='BMS', sma_days=None):
    """Run Python + Rust on production SPY data."""
    common = dict(
        alloc=alloc, capital=capital, strategy_fn=strategy_fn,
        rebalance_freq=rebalance_freq, rebalance_unit=rebalance_unit,
        stocks=prod_spy_stocks(),
        stocks_data=load_prod_stocks(),
        options_data=load_prod_options(),
        sma_days=sma_days,
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


class TestPerRebalanceGenerated:
    """Per-date balance parity on generated synthetic data."""

    def test_buy_put(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)
        assert len(py.balance) >= 10
        assert_per_rebalance_parity(py, rs, label="gen_buy_put")

    def test_buy_call(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_call_strategy)
        assert_per_rebalance_parity(py, rs, label="gen_buy_call")

    def test_sell_put(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, sell_put_strategy)
        assert_per_rebalance_parity(py, rs, label="gen_sell_put")

    def test_sell_call(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, sell_call_strategy)
        assert_per_rebalance_parity(py, rs, label="gen_sell_call")

    def test_strangle(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, strangle_strategy)
        assert_per_rebalance_parity(py, rs, label="gen_strangle")

    def test_straddle(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, straddle_strategy)
        assert_per_rebalance_parity(py, rs, label="gen_straddle")

    def test_high_options_alloc(self):
        alloc = {"stocks": 0.10, "options": 0.80, "cash": 0.10}
        py, rs = _run_pair_gen(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_per_rebalance_parity(py, rs, label="gen_high_opts")

    def test_with_sma(self):
        py, rs = _run_pair_gen(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy, sma_days=50,
        )
        assert_per_rebalance_parity(py, rs, label="gen_sma_50")


class TestPerRebalanceProduction:
    """Per-date balance parity on production SPY data."""

    def test_buy_put_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: buy_put_strategy(schema, underlying="SPY"),
        )
        assert_per_rebalance_parity(py, rs, label="prod_buy_put")

    def test_sell_call_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: sell_call_strategy(schema, underlying="SPY"),
        )
        assert_per_rebalance_parity(py, rs, label="prod_sell_call")

    def test_strangle_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: strangle_strategy(schema, underlying="SPY"),
        )
        assert_per_rebalance_parity(py, rs, label="prod_strangle")

    def test_put_spread_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: buy_put_spread_strategy(schema, underlying="SPY"),
        )
        assert_per_rebalance_parity(py, rs, label="prod_put_spread")
