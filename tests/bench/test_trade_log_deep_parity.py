"""Deep row-level trade log verification — contracts, orders, strikes, dates.

Goes beyond shape/cost/qty to verify the actual content of each trade row
matches between Python and Rust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    run_rust,
    run_python,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


def _normalize_order(val):
    """Normalize Order enum or string to plain BTO/STC/STO/BTC."""
    s = str(val)
    for suffix in ("BTO", "STC", "STO", "BTC"):
        if s.endswith(suffix):
            return suffix
    return s


def _deep_trade_log_compare(py_eng, rs_eng, label=""):
    """Full row-level comparison of trade logs."""
    prefix = f"[{label}] " if label else ""
    py_tl = py_eng.trade_log
    rs_tl = rs_eng.trade_log

    assert py_tl.shape == rs_tl.shape, (
        f"{prefix}shape mismatch: py={py_tl.shape} rs={rs_tl.shape}"
    )
    if py_tl.empty:
        return

    # Get leg names
    legs = [c for c in py_tl.columns.get_level_values(0).unique()
            if c != "totals"]

    for leg in legs:
        # Contract IDs must match exactly
        py_c = py_tl[leg]["contract"].astype(str).values
        rs_c = rs_tl[leg]["contract"].astype(str).values
        assert np.array_equal(py_c, rs_c), (
            f"{prefix}{leg} contracts differ at rows: "
            f"{np.where(py_c != rs_c)[0]}"
        )

        # Order types
        py_o = np.array([_normalize_order(v) for v in py_tl[leg]["order"].values])
        rs_o = np.array([_normalize_order(v) for v in rs_tl[leg]["order"].values])
        assert np.array_equal(py_o, rs_o), (
            f"{prefix}{leg} orders differ: "
            f"py={py_o[:5]} rs={rs_o[:5]}"
        )

        # Strikes
        py_s = py_tl[leg]["strike"].astype(float).values
        rs_s = rs_tl[leg]["strike"].astype(float).values
        assert np.allclose(py_s, rs_s, rtol=1e-6), (
            f"{prefix}{leg} strikes differ"
        )

        # Type (call/put)
        py_t = py_tl[leg]["type"].astype(str).values
        rs_t = rs_tl[leg]["type"].astype(str).values
        assert np.array_equal(py_t, rs_t), (
            f"{prefix}{leg} types differ: py={py_t[:5]} rs={rs_t[:5]}"
        )

        # Underlying
        py_u = py_tl[leg]["underlying"].astype(str).values
        rs_u = rs_tl[leg]["underlying"].astype(str).values
        assert np.array_equal(py_u, rs_u), (
            f"{prefix}{leg} underlyings differ"
        )

        # Per-leg costs
        py_lc = py_tl[leg]["cost"].astype(float).values
        rs_lc = rs_tl[leg]["cost"].astype(float).values
        assert np.allclose(py_lc, rs_lc, rtol=1e-4), (
            f"{prefix}{leg} costs differ: max_diff="
            f"{np.max(np.abs(py_lc - rs_lc))}"
        )

    # Total costs
    py_tc = py_tl["totals"]["cost"].astype(float).values
    rs_tc = rs_tl["totals"]["cost"].astype(float).values
    assert np.allclose(py_tc, rs_tc, rtol=1e-4), (
        f"{prefix}totals costs differ"
    )

    # Quantities
    py_q = py_tl["totals"]["qty"].astype(float).values
    rs_q = rs_tl["totals"]["qty"].astype(float).values
    assert np.allclose(py_q, rs_q, rtol=1e-6), (
        f"{prefix}totals qty differ"
    )

    # Dates
    py_d = pd.to_datetime(py_tl["totals"]["date"]).values
    rs_d = pd.to_datetime(rs_tl["totals"]["date"]).values
    assert np.array_equal(py_d, rs_d), (
        f"{prefix}dates differ"
    )


# ── Single-leg strategies ─────────────────────────────────────────────

class TestTradeLogDeepSingleLeg:

    @pytest.mark.parametrize("strategy_name,strategy_fn", [
        ("buy_put", buy_put_strategy),
        ("buy_call", buy_call_strategy),
        ("sell_put", sell_put_strategy),
        ("sell_call", sell_call_strategy),
    ])
    def test_default_config(self, strategy_name, strategy_fn):
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn)
        _deep_trade_log_compare(py, rs, label=f"deep-{strategy_name}")

    def test_with_per_contract_costs(self):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=0.65)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        _deep_trade_log_compare(py, rs, label="deep-with-costs")

    def test_with_midprice(self):
        from options_portfolio_backtester.execution.fill_model import MidPrice
        fm = MidPrice()
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      fill_model=fm)
        _deep_trade_log_compare(py, rs, label="deep-midprice")

    def test_with_nearest_delta(self):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        ss = NearestDelta(target_delta=-0.30)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      signal_selector=ss)
        _deep_trade_log_compare(py, rs, label="deep-nearest-delta")

    def test_with_thresholds(self):
        import math

        def strat(schema):
            s = buy_put_strategy(schema)
            s.add_exit_thresholds(0.50, 0.30)
            return s

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strat)
        _deep_trade_log_compare(py, rs, label="deep-thresholds")


# ── Multi-leg strategies ──────────────────────────────────────────────

class TestTradeLogDeepMultiLeg:

    @pytest.mark.parametrize("strategy_name,strategy_fn", [
        ("put_spread", buy_put_spread_strategy),
        ("call_spread", sell_call_spread_strategy),
        ("strangle", strangle_strategy),
        ("straddle", straddle_strategy),
    ])
    def test_multi_leg_default(self, strategy_name, strategy_fn):
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn)
        _deep_trade_log_compare(py, rs, label=f"deep-multi-{strategy_name}")

    @pytest.mark.parametrize("strategy_name,strategy_fn", [
        ("put_spread", buy_put_spread_strategy),
        ("strangle", strangle_strategy),
    ])
    def test_multi_leg_with_costs(self, strategy_name, strategy_fn):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=0.65)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                      cost_model=cm)
        _deep_trade_log_compare(py, rs,
                                label=f"deep-multi-{strategy_name}-costs")
