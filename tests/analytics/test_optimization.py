"""Tests for analytics/optimization.py — grid_sweep and walk_forward."""

import pandas as pd
import numpy as np

from options_portfolio_backtester.analytics.optimization import (
    OptimizationResult, grid_sweep, walk_forward,
)
from options_portfolio_backtester.analytics.stats import BacktestStats


def _dummy_run_fn(param_a=1, param_b=2):
    """Dummy backtest function for grid sweep tests."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    capital = [100_000.0]
    for _ in range(49):
        capital.append(capital[-1] * (1 + 0.001 * param_a))
    bal = pd.DataFrame({"total capital": capital}, index=dates)
    bal["% change"] = bal["total capital"].pct_change()
    stats = BacktestStats._from_balance_python(bal)
    return stats, bal


def _failing_run_fn(param_a=1):
    """Fails when param_a == 2; picklable because module-level."""
    if param_a == 2:
        raise ValueError("boom")
    return _dummy_run_fn(param_a=param_a)


def _dummy_wf_fn(start_date, end_date):
    """Dummy walk-forward function."""
    dates = pd.bdate_range(start_date, end_date)
    if len(dates) < 2:
        dates = pd.bdate_range(start_date, periods=5)
    capital = np.linspace(100000, 105000, len(dates))
    bal = pd.DataFrame({"total capital": capital}, index=dates)
    bal["% change"] = bal["total capital"].pct_change()
    stats = BacktestStats._from_balance_python(bal)
    return stats, bal


class TestOptimizationResult:
    def test_fields(self):
        stats = BacktestStats()
        bal = pd.DataFrame()
        r = OptimizationResult(params={"x": 1}, stats=stats, balance=bal)
        assert r.params == {"x": 1}
        assert r.stats is stats


class TestGridSweep:
    def test_returns_results_for_all_combos(self):
        results = grid_sweep(
            _dummy_run_fn,
            param_grid={"param_a": [1, 2], "param_b": [10, 20]},
            max_workers=1,
        )
        assert len(results) == 4

    def test_sorted_by_sharpe_descending(self):
        results = grid_sweep(
            _dummy_run_fn,
            param_grid={"param_a": [1, 2, 3]},
            max_workers=1,
        )
        sharpes = [r.stats.sharpe_ratio for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_single_combo(self):
        results = grid_sweep(
            _dummy_run_fn,
            param_grid={"param_a": [1]},
            max_workers=1,
        )
        assert len(results) == 1

    def test_failing_fn_skipped(self):
        results = grid_sweep(
            _failing_run_fn,
            param_grid={"param_a": [1, 2, 3]},
            max_workers=1,
        )
        # param_a=2 should be skipped
        assert len(results) == 2


class TestWalkForward:
    def test_returns_splits(self):
        dates = pd.bdate_range("2020-01-01", periods=250)
        results = walk_forward(_dummy_wf_fn, dates, n_splits=3)
        assert len(results) == 3
        for is_result, oos_result in results:
            assert is_result.params["type"] == "in_sample"
            assert oos_result.params["type"] == "out_of_sample"

    def test_single_split(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        results = walk_forward(_dummy_wf_fn, dates, n_splits=1)
        assert len(results) == 1

    def test_failing_wf_fn_skipped(self):
        """Walk-forward skips splits where run_fn raises."""
        call_count = [0]

        def _failing_wf(start_date, end_date):
            call_count[0] += 1
            # Fail on call 1 (in-sample for split 0) → entire split 0 skipped
            if call_count[0] == 1:
                raise ValueError("boom")
            return _dummy_wf_fn(start_date, end_date)

        dates = pd.bdate_range("2020-01-01", periods=200)
        results = walk_forward(_failing_wf, dates, n_splits=2)
        # Split 0 fails (in-sample raises), split 1 succeeds
        assert len(results) == 1

    def test_custom_in_sample_pct(self):
        dates = pd.bdate_range("2020-01-01", periods=200)
        results = walk_forward(_dummy_wf_fn, dates, n_splits=2, in_sample_pct=0.8)
        assert len(results) == 2
