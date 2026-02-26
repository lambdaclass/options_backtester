"""Multi-dataset production parity tests.

Parametrized across diverse time-period slices (crisis, low-vol, COVID, bear,
IWM, QQQ) to stress different code paths. All tests skip if slice CSVs are
missing -- run `python tests/bench/extract_prod_slices.py` to generate them.

Sections (~120 tests):
A. 2-way parity (Python vs Rust) x 6 slices x 8 strategies             = 48
B. Per-rebalance balance verification x 6 slices x 4 strategies         = 24
C. Balance sheet and trade log invariants x 6 slices x 3 checks         = 18
D. Execution model 2-way parity x 6 slices x 3 combos x 2 strats       = 36
E. Weekly rebalancing 2-way x 6 slices x 2 strategies                   = 12
F. Full-range 2-way (slow) x 3 strategies                               =  3
G. Hypothesis fuzz x 2 methods                                          =  2
                                                                        ~ 143
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    PROD_SLICES,
    STRATEGY_MAP,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    _slice_data_exists,
    load_slice_stocks,
    load_slice_options,
    slice_stocks,
    buy_put_strategy,
    buy_call_strategy,
    sell_put_strategy,
    strangle_strategy,
    make_cost_model,
    make_fill_model,
    make_signal_selector,
    run_python_ex,
    run_rust_ex,
    assert_parity,
    assert_per_rebalance_parity,
)

pytestmark = [
    pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed"),
    pytest.mark.bench,
]

_SLICE_IDS = list(PROD_SLICES.keys())
_STRATEGY_NAMES = list(STRATEGY_MAP.keys())
_CORE_STRATEGIES = ["buy_put", "sell_put", "buy_call", "strangle"]


def _skip_if_missing(slice_id):
    if not _slice_data_exists(slice_id):
        pytest.skip(
            f"Slice {slice_id} CSVs not found -- "
            f"run: python tests/bench/extract_prod_slices.py"
        )


# -- Helpers ---------------------------------------------------------------

def _run_pair_slice(slice_id, strategy_fn, rebalance_freq=1,
                    rebalance_unit='BMS', sma_days=None,
                    alloc=None, capital=None, **engine_kwargs):
    """Run Python + Rust on a production slice."""
    underlying = PROD_SLICES[slice_id]["underlying"]
    common = dict(
        alloc=alloc or DEFAULT_ALLOC,
        capital=capital or DEFAULT_CAPITAL,
        strategy_fn=lambda schema, u=underlying: strategy_fn(schema, underlying=u),
        rebalance_freq=rebalance_freq,
        rebalance_unit=rebalance_unit,
        stocks=slice_stocks(slice_id),
        stocks_data=load_slice_stocks(slice_id),
        options_data=load_slice_options(slice_id),
        sma_days=sma_days,
        **engine_kwargs,
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


# ======================================================================
# A. 2-way parity: 6 slices x 8 strategies = 48 tests
# ======================================================================

@pytest.mark.parametrize("slice_id", _SLICE_IDS)
@pytest.mark.parametrize("strategy_name", _STRATEGY_NAMES)
class TestMultiDatasetParity:
    """2-way parity across diverse production slices and all strategies."""

    def test_parity(self, slice_id, strategy_name):
        _skip_if_missing(slice_id)
        strategy_fn = STRATEGY_MAP[strategy_name]
        py, rs = _run_pair_slice(slice_id, strategy_fn)
        assert_parity(py, rs, label=f"{slice_id}_{strategy_name}")


# ======================================================================
# B. Per-rebalance: 6 slices x 4 core strategies = 24 tests
# ======================================================================

@pytest.mark.parametrize("slice_id", _SLICE_IDS)
@pytest.mark.parametrize("strategy_name", _CORE_STRATEGIES)
class TestMultiDatasetPerRebalance:
    """Per-date balance parity across production slices."""

    def test_per_rebalance(self, slice_id, strategy_name):
        _skip_if_missing(slice_id)
        strategy_fn = STRATEGY_MAP[strategy_name]
        py, rs = _run_pair_slice(slice_id, strategy_fn)
        assert len(py.balance) >= 5, (
            f"Too few balance rows ({len(py.balance)}) for {slice_id}"
        )
        assert_per_rebalance_parity(
            py, rs, label=f"{slice_id}_{strategy_name}_per_date"
        )


# ======================================================================
# C. Invariants: 6 slices x 3 checks = 18 tests
# ======================================================================

@pytest.mark.parametrize("slice_id", _SLICE_IDS)
class TestMultiDatasetInvariants:
    """Balance sheet and trade log invariants across production slices."""

    def _run_pair(self, slice_id):
        return _run_pair_slice(slice_id, buy_put_strategy)

    def test_capital_equals_parts(self, slice_id):
        _skip_if_missing(slice_id)
        py, rs = self._run_pair(slice_id)
        for eng, path in [(py, "python"), (rs, "rust")]:
            bal = eng.balance
            if "options capital" in bal.columns and "stocks capital" in bal.columns:
                reconstructed = (
                    bal["cash"] + bal["stocks capital"] + bal["options capital"]
                )
                assert np.allclose(
                    bal["total capital"].values[1:],
                    reconstructed.values[1:],
                    rtol=1e-4, atol=1.0,
                ), f"[{slice_id}:{path}] total capital != cash + stocks + options"

    def test_capital_never_negative(self, slice_id):
        _skip_if_missing(slice_id)
        py, rs = self._run_pair(slice_id)
        for eng, path in [(py, "python"), (rs, "rust")]:
            tc = eng.balance["total capital"]
            assert (tc >= -1.0).all(), (
                f"[{slice_id}:{path}] negative total capital: min={tc.min()}"
            )

    def test_trade_log_not_empty(self, slice_id):
        _skip_if_missing(slice_id)
        py, rs = self._run_pair(slice_id)
        assert not py.trade_log.empty, f"[{slice_id}:python] trade log empty"
        assert not rs.trade_log.empty, f"[{slice_id}:rust] trade log empty"


# ======================================================================
# D. Execution models (2-way): 6 slices x 3 combos x 2 strategies = 36
# ======================================================================

_EXEC_COMBOS = [
    ("PerContract", "MarketAtBidAsk", "FirstMatch"),
    ("NoCosts", "MidPrice", "FirstMatch"),
    ("NoCosts", "MarketAtBidAsk", "NearestDelta"),
]
_EXEC_STRATEGIES = ["buy_put", "strangle"]


@pytest.mark.parametrize("slice_id", _SLICE_IDS)
@pytest.mark.parametrize("combo", _EXEC_COMBOS,
                         ids=[f"{c}-{f}-{s}" for c, f, s in _EXEC_COMBOS])
@pytest.mark.parametrize("strategy_name", _EXEC_STRATEGIES)
class TestMultiDatasetExecModels:
    """2-way execution model parity on production slices."""

    def test_exec_model_parity(self, slice_id, combo, strategy_name):
        _skip_if_missing(slice_id)
        cost_name, fill_name, signal_name = combo
        strategy_fn = STRATEGY_MAP[strategy_name]
        py, rs = _run_pair_slice(
            slice_id, strategy_fn,
            cost_model=make_cost_model(cost_name),
            fill_model=make_fill_model(fill_name),
            signal_selector=make_signal_selector(signal_name),
        )
        assert_parity(
            py, rs, rtol=5e-3, atol=50.0,
            label=f"{slice_id}_{strategy_name}_{cost_name}_{fill_name}_{signal_name}",
        )


# ======================================================================
# E. Weekly rebalancing (2-way): 6 slices x 2 strategies = 12
# ======================================================================

_WEEKLY_STRATEGIES = ["buy_put", "strangle"]


@pytest.mark.parametrize("slice_id", _SLICE_IDS)
@pytest.mark.parametrize("strategy_name", _WEEKLY_STRATEGIES)
class TestMultiDatasetWeekly:
    """2-way parity with weekly (W-MON) rebalancing."""

    def test_weekly_parity(self, slice_id, strategy_name):
        _skip_if_missing(slice_id)
        strategy_fn = STRATEGY_MAP[strategy_name]
        py, rs = _run_pair_slice(
            slice_id, strategy_fn, rebalance_unit='W-MON',
        )
        assert_parity(
            py, rs, label=f"{slice_id}_{strategy_name}_weekly"
        )


# ======================================================================
# F. Full-range 2-way (slow): 3 strategies
# ======================================================================

_FULL_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed"
)


def _full_range_available():
    return (
        os.path.isfile(os.path.join(_FULL_DATA_DIR, "options.csv"))
        and os.path.isfile(os.path.join(_FULL_DATA_DIR, "stocks.csv"))
    )


def _load_full_stocks():
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_FULL_DATA_DIR, "stocks.csv"))


def _load_full_options():
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(os.path.join(_FULL_DATA_DIR, "options.csv"))


def _full_spy_stocks():
    from options_portfolio_backtester.core.types import Stock
    return [Stock("SPY", 1.0)]


def _run_pair_full(strategy_fn):
    """Run Python and Rust engines on full 17-year SPY dataset."""
    common = dict(
        alloc=DEFAULT_ALLOC,
        capital=DEFAULT_CAPITAL,
        strategy_fn=lambda schema: strategy_fn(schema, underlying="SPY"),
        rebalance_freq=1,
        rebalance_unit='BMS',
        stocks=_full_spy_stocks(),
        stocks_data=_load_full_stocks(),
        options_data=_load_full_options(),
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


@pytest.mark.slow
class TestFullRange:
    """17-year SPY, 24.7M rows, ~4500 trading days.

    Run explicitly with: python -m pytest -m slow
    """

    def test_buy_put_full_range(self):
        if not _full_range_available():
            pytest.skip("Full-range data not found in data/processed/")
        py, rs = _run_pair_full(buy_put_strategy)
        assert len(py.balance) >= 100
        assert_parity(py, rs, rtol=5e-3, atol=50.0,
                       label="full_range_buy_put")

    def test_sell_put_full_range(self):
        if not _full_range_available():
            pytest.skip("Full-range data not found in data/processed/")
        py, rs = _run_pair_full(sell_put_strategy)
        assert len(py.balance) >= 100
        assert_parity(py, rs, rtol=5e-3, atol=50.0,
                       label="full_range_sell_put")

    def test_buy_call_full_range(self):
        if not _full_range_available():
            pytest.skip("Full-range data not found in data/processed/")
        py, rs = _run_pair_full(buy_call_strategy)
        assert len(py.balance) >= 100
        assert_parity(py, rs, rtol=5e-3, atol=50.0,
                       label="full_range_buy_call")


# ======================================================================
# G. Hypothesis fuzz: 2 tests x 30 examples each
# ======================================================================

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

_FUZZ_SLICE_IDS = list(PROD_SLICES.keys())


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestHypothesisFuzz:
    """Property-based fuzz testing across production slices."""

    @given(
        slice_id=st.sampled_from(_FUZZ_SLICE_IDS),
        options_pct=st.floats(min_value=0.05, max_value=0.50),
        capital=st.integers(min_value=100_000, max_value=5_000_000),
    )
    @settings(max_examples=30, deadline=None)
    def test_fuzz_allocation(self, slice_id, options_pct, capital):
        assume(_slice_data_exists(slice_id))
        stocks_pct = 1.0 - options_pct - 0.10
        if stocks_pct <= 0:
            assume(False)
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": 0.10}
        py, rs = _run_pair_slice(
            slice_id, buy_put_strategy,
            alloc=alloc, capital=capital,
        )
        assert_parity(py, rs, rtol=5e-3, atol=50.0,
                      label=f"fuzz_alloc_{slice_id}")

    @given(
        slice_id=st.sampled_from(_FUZZ_SLICE_IDS),
        cost_name=st.sampled_from(["NoCosts", "PerContract", "Tiered"]),
        fill_name=st.sampled_from(["MarketAtBidAsk", "MidPrice"]),
        signal_name=st.sampled_from(["FirstMatch", "NearestDelta", "MaxOpenInterest"]),
    )
    @settings(max_examples=30, deadline=None)
    def test_fuzz_models(self, slice_id, cost_name, fill_name, signal_name):
        assume(_slice_data_exists(slice_id))
        py, rs = _run_pair_slice(
            slice_id, buy_put_strategy,
            cost_model=make_cost_model(cost_name),
            fill_model=make_fill_model(fill_name),
            signal_selector=make_signal_selector(signal_name),
        )
        assert_parity(py, rs, rtol=5e-3, atol=50.0,
                      label=f"fuzz_models_{slice_id}")
