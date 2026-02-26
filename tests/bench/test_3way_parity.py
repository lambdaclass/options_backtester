"""2-way parity tests: Python engine vs Rust engine.

Uses both the generated synthetic dataset (500 days, 7 stocks, ~18K options)
and the production SPY dataset (252 days, ~5K options).
Monthly rebalancing (rebalance_unit='BMS') for confirmed parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_portfolio_backtester.core.types import Stock

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    GENERATED_STOCKS_TUPLES,
    PROD_SPY_STOCKS_TUPLES,
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
    sell_call_spread_strategy,
    run_python_ex,
    run_rust_ex,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# ── Helpers ──────────────────────────────────────────────────────────

def _run_pair(alloc, capital, strategy_fn, rebalance_freq=1,
              rebalance_unit='BMS', stocks=None,
              stocks_data=None, options_data=None,
              sma_days=None):
    """Run Python and Rust engines and return (py, rs)."""
    common = dict(
        alloc=alloc, capital=capital, strategy_fn=strategy_fn,
        rebalance_freq=rebalance_freq, rebalance_unit=rebalance_unit,
        stocks=stocks, stocks_data=stocks_data, options_data=options_data,
        sma_days=sma_days,
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


def _run_pair_gen(alloc, capital, strategy_fn, rebalance_freq=1,
                  rebalance_unit='BMS', sma_days=None):
    """Run Python and Rust engines on the generated synthetic dataset."""
    return _run_pair(
        alloc, capital, strategy_fn,
        rebalance_freq=rebalance_freq,
        rebalance_unit=rebalance_unit,
        stocks=generated_stocks(),
        stocks_data=load_generated_stocks(),
        options_data=load_generated_options(),
        sma_days=sma_days,
    )


def _run_pair_prod(alloc, capital, strategy_fn, rebalance_freq=1,
                   rebalance_unit='BMS', sma_days=None):
    """Run Python and Rust engines on the production SPY dataset."""
    return _run_pair(
        alloc, capital, strategy_fn,
        rebalance_freq=rebalance_freq,
        rebalance_unit=rebalance_unit,
        stocks=prod_spy_stocks(),
        stocks_data=load_prod_stocks(),
        options_data=load_prod_options(),
        sma_days=sma_days,
    )


# ── Single-leg strategy tests (generated data) ──────────────────────

class TestParitySingleLeg:
    """Single-leg strategies on generated data with monthly rebalancing."""

    def test_buy_put(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)
        assert len(py.trade_log) > 10
        assert_parity(py, rs, label="buy_put")

    def test_buy_call(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_call_strategy)
        assert len(py.trade_log) > 10
        assert_parity(py, rs, label="buy_call")

    def test_sell_put(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, sell_put_strategy)
        assert len(py.trade_log) > 10
        assert_parity(py, rs, label="sell_put")

    def test_sell_call(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, sell_call_strategy)
        assert len(py.trade_log) > 10
        assert_parity(py, rs, label="sell_call")


# ── Multi-leg strategy tests (generated data) ───────────────────────

class TestParityMultiLeg:
    """Multi-leg strategies on generated data."""

    def test_strangle(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, strangle_strategy)
        assert len(py.trade_log) > 10
        assert_parity(py, rs, label="strangle")

    def test_straddle(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, straddle_strategy)
        assert len(py.trade_log) > 10
        assert_parity(py, rs, label="straddle")

    def test_put_spread(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_spread_strategy)
        assert_parity(py, rs, label="put_spread")

    def test_call_spread(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, DEFAULT_CAPITAL, sell_call_spread_strategy)
        assert_parity(py, rs, label="call_spread")


# ── Allocation variant tests ─────────────────────────────────────────

class TestParityAllocations:
    """Different allocation splits on generated data (buy_put, BMS)."""

    def test_alloc_97_3_0(self):
        alloc = {"stocks": 0.97, "options": 0.03, "cash": 0.0}
        py, rs = _run_pair_gen(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="97/3/0")

    def test_alloc_50_50_0(self):
        alloc = {"stocks": 0.50, "options": 0.50, "cash": 0.0}
        py, rs = _run_pair_gen(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="50/50/0")

    def test_alloc_10_80_10(self):
        alloc = {"stocks": 0.10, "options": 0.80, "cash": 0.10}
        py, rs = _run_pair_gen(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="10/80/10")

    def test_alloc_30_60_10(self):
        alloc = {"stocks": 0.30, "options": 0.60, "cash": 0.10}
        py, rs = _run_pair_gen(alloc, DEFAULT_CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="30/60/10")


# ── Capital variant tests ────────────────────────────────────────────

class TestParityCapital:
    """Different capital amounts."""

    def test_capital_10k(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, 10_000, buy_put_strategy)
        assert_parity(py, rs, label="10K")

    def test_capital_100k(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, 100_000, buy_put_strategy)
        assert_parity(py, rs, label="100K")

    def test_capital_10m(self):
        py, rs = _run_pair_gen(DEFAULT_ALLOC, 10_000_000, buy_put_strategy)
        assert_parity(py, rs, label="10M")


# ── Rebalance frequency tests ───────────────────────────────────────

class TestParityRebalanceFreq:
    """Different rebalance frequencies."""

    def test_monthly(self):
        py, rs = _run_pair_gen(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
            rebalance_freq=1, rebalance_unit='BMS',
        )
        assert_parity(py, rs, label="monthly")

    def test_weekly(self):
        py, rs = _run_pair_gen(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
            rebalance_freq=1, rebalance_unit='W-MON',
        )
        assert_parity(py, rs, label="weekly")

    def test_biweekly(self):
        py, rs = _run_pair_gen(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
            rebalance_freq=2, rebalance_unit='W-MON',
        )
        assert_parity(py, rs, label="biweekly")


# ── SMA gating tests ────────────────────────────────────────────────

class TestParitySMA:
    """SMA gating on stock purchases."""

    def test_sma_50(self):
        py, rs = _run_pair_gen(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
            sma_days=50,
        )
        assert_parity(py, rs, label="sma_50")

    def test_sma_200(self):
        py, rs = _run_pair_gen(
            DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
            sma_days=200,
        )
        assert_parity(py, rs, label="sma_200")


# ── Production SPY data tests ───────────────────────────────────────

class TestParityProduction:
    """Tests using real production SPY data."""

    def test_buy_put_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: buy_put_strategy(schema, underlying="SPY"),
        )
        assert_parity(py, rs, label="prod_buy_put")

    def test_sell_put_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: sell_put_strategy(schema, underlying="SPY"),
        )
        assert_parity(py, rs, label="prod_sell_put")

    def test_buy_call_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: buy_call_strategy(schema, underlying="SPY"),
        )
        assert_parity(py, rs, label="prod_buy_call")

    def test_strangle_spy(self):
        py, rs = _run_pair_prod(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            lambda schema: strangle_strategy(schema, underlying="SPY"),
        )
        assert_parity(py, rs, label="prod_strangle")


# ── Hypothesis fuzz tests ───────────────────────────────────────────

try:
    from hypothesis import given, settings, strategies as st, HealthCheck
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestParityFuzz:
    """Hypothesis-based fuzz testing of 2-way parity.

    Uses only buy_put and sell_put/sell_call (strategies with confirmed parity).
    """

    @given(
        options_pct=st.floats(min_value=0.05, max_value=0.50),
        capital=st.sampled_from([1_000_000, 5_000_000, 10_000_000]),
    )
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_generated(self, options_pct, capital):
        stocks_pct = (1.0 - options_pct) * 0.9
        cash_pct = (1.0 - options_pct) * 0.1
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}

        py, rs = _run_pair_gen(
            alloc, capital, buy_put_strategy,
            rebalance_freq=1, rebalance_unit='BMS',
        )
        assert_parity(py, rs, rtol=5e-3, atol=10.0,
                       label=f"fuzz_buy_put_{capital}")
