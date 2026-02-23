"""Walk-forward optimization and parameter grid sweep."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import numpy as np

from options_backtester.analytics.stats import BacktestStats


@dataclass
class OptimizationResult:
    """Result of a single parameter combination."""
    params: dict[str, Any]
    stats: BacktestStats
    balance: pd.DataFrame


def grid_sweep(
    run_fn: Callable[..., tuple[BacktestStats, pd.DataFrame]],
    param_grid: dict[str, list[Any]],
    max_workers: int | None = None,
) -> list[OptimizationResult]:
    """Run a parameter grid sweep using parallel execution.

    Args:
        run_fn: Function that takes **params and returns (BacktestStats, balance).
        param_grid: Dict mapping parameter names to lists of values.
        max_workers: Number of parallel workers (None = CPU count).

    Returns:
        List of OptimizationResult, sorted by Sharpe ratio descending.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results: list[OptimizationResult] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for combo in combos:
            params = dict(zip(keys, combo))
            future = executor.submit(run_fn, **params)
            futures[future] = params

        for future in as_completed(futures):
            params = futures[future]
            try:
                stats, balance = future.result()
                results.append(OptimizationResult(
                    params=params, stats=stats, balance=balance,
                ))
            except Exception:
                continue

    results.sort(key=lambda r: r.stats.sharpe_ratio, reverse=True)
    return results


def walk_forward(
    run_fn: Callable[[pd.Timestamp, pd.Timestamp], tuple[BacktestStats, pd.DataFrame]],
    dates: pd.DatetimeIndex,
    in_sample_pct: float = 0.70,
    n_splits: int = 5,
) -> list[tuple[OptimizationResult, OptimizationResult]]:
    """Walk-forward analysis with rolling in-sample/out-of-sample splits.

    Args:
        run_fn: Function that takes (start_date, end_date) and returns (stats, balance).
        dates: Full date range.
        in_sample_pct: Fraction of each window used for in-sample.
        n_splits: Number of walk-forward windows.

    Returns:
        List of (in_sample_result, out_of_sample_result) tuples.
    """
    total = len(dates)
    window_size = total // n_splits
    results = []

    for i in range(n_splits):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, total)
        split_idx = start_idx + int((end_idx - start_idx) * in_sample_pct)

        is_start = dates[start_idx]
        is_end = dates[split_idx - 1]
        oos_start = dates[split_idx]
        oos_end = dates[end_idx - 1]

        try:
            is_stats, is_balance = run_fn(is_start, is_end)
            oos_stats, oos_balance = run_fn(oos_start, oos_end)
            results.append((
                OptimizationResult(params={"split": i, "type": "in_sample"},
                                   stats=is_stats, balance=is_balance),
                OptimizationResult(params={"split": i, "type": "out_of_sample"},
                                   stats=oos_stats, balance=oos_balance),
            ))
        except Exception:
            continue

    return results
