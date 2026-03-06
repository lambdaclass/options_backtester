"""Regression tests for Rust parallel_sweep API."""

import pandas as pd
import pytest

try:
    import polars as pl
    from options_portfolio_backtester._ob_rust import (
        parallel_sweep as rust_parallel_sweep,
        run_backtest_py as rust_run_backtest,
    )
    from options_portfolio_backtester.analytics.optimization import rust_grid_sweep
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed")


def _pd_to_pl(df: pd.DataFrame) -> "pl.DataFrame":
    return pl.from_pandas(df)


def _dates_to_ns(dates: list[str]) -> list[int]:
    return [int(pd.Timestamp(d).value) for d in dates]


def _ensure_datetime_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _make_test_data():
    dates = ["2024-01-01"] * 4 + ["2024-01-15"] * 4 + ["2024-02-01"] * 4
    opts = pd.DataFrame({
        "optionroot": ["A", "B", "C", "D"] * 3,
        "underlying": ["SPX"] * 12,
        "underlying_last": [4500.0] * 12,
        "quotedate": dates,
        "type": ["put", "put", "call", "put"] * 3,
        "expiration": ["2024-03-01"] * 12,
        "strike": [4400.0, 4300.0, 4500.0, 4200.0] * 3,
        "bid": [5.0, 8.0, 3.0, 12.0,
                4.0, 7.0, 2.0, 11.0,
                3.5, 6.5, 1.5, 10.5],
        "ask": [6.0, 9.0, 4.0, 13.0,
                5.0, 8.0, 3.0, 12.0,
                4.5, 7.5, 2.5, 11.5],
        "volume": [100] * 12,
        "open_interest": [1000] * 12,
        "dte": [60] * 12,
    })
    stocks = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-15", "2024-02-01"] * 2,
        "symbol": ["SPY"] * 3 + ["TLT"] * 3,
        "adjClose": [450.0, 455.0, 460.0, 100.0, 101.0, 102.0],
    })
    opts = _ensure_datetime_cols(opts, ["quotedate", "expiration"])
    stocks = _ensure_datetime_cols(stocks, ["date"])
    return _pd_to_pl(opts), _pd_to_pl(stocks)


def _base_config():
    return {
        "allocation": {"stocks": 0.5, "options": 0.3, "cash": 0.2},
        "initial_capital": 100000.0,
        "shares_per_contract": 100,
        "legs": [{
            "name": "leg_1",
            "entry_filter": "(type == 'put') & (ask > 0)",
            "exit_filter": "type == 'put'",
            "direction": "ask",
            "type": "put",
            "entry_sort_col": None,
            "entry_sort_asc": True,
        }],
        "profit_pct": None,
        "loss_pct": None,
        "stocks": [("SPY", 0.6), ("TLT", 0.4)],
        "rebalance_dates": _dates_to_ns(["2024-01-01", "2024-01-15", "2024-02-01"]),
    }


def _schema():
    return {
        "contract": "optionroot",
        "date": "quotedate",
        "stocks_date": "date",
        "stocks_symbol": "symbol",
        "stocks_price": "adjClose",
    }


class TestSweep:

    def test_single_config_matches_direct(self):
        opts, stocks = _make_test_data()
        config = _base_config()
        schema = _schema()

        _balance, _trade_log, direct_stats = rust_run_backtest(
            opts, stocks, config, schema,
        )
        sweep_results = rust_parallel_sweep(
            opts, stocks, config, schema, [{"label": "base"}],
        )

        assert len(sweep_results) == 1
        r = sweep_results[0]
        assert r["label"] == "base"
        assert r["error"] is None
        assert abs(r["total_return"] - direct_stats["total_return"]) < 1e-10
        assert abs(r["final_cash"] - direct_stats["final_cash"]) < 1e-6

    def test_multiple_configs(self):
        opts, stocks = _make_test_data()
        config = _base_config()
        schema = _schema()

        overrides = [
            {"label": "tight", "profit_pct": 0.01, "loss_pct": 0.01},
            {"label": "wide", "profit_pct": 0.99, "loss_pct": 0.99},
        ]
        results = rust_parallel_sweep(opts, stocks, config, schema, overrides)

        assert len(results) == 2
        for r in results:
            assert r["error"] is None

    def test_per_leg_filter_overrides(self):
        opts, stocks = _make_test_data()
        config = _base_config()
        schema = _schema()

        overrides = [
            {"label": "broad", "leg_entry_filters": ["(type == 'put') & (ask > 0)"]},
            {"label": "narrow", "leg_entry_filters": ["(type == 'put') & (ask > 0) & (strike < 4300)"]},
        ]
        results = rust_parallel_sweep(opts, stocks, config, schema, overrides)

        by_label = {r["label"]: r for r in results}
        assert by_label["narrow"]["total_trades"] <= by_label["broad"]["total_trades"]

    def test_bad_filter_returns_error(self):
        opts, stocks = _make_test_data()
        config = _base_config()
        schema = _schema()

        overrides = [{"label": "bad", "leg_entry_filters": ["(((invalid syntax!!!"]}]
        results = rust_parallel_sweep(opts, stocks, config, schema, overrides)

        assert len(results) == 1
        r = results[0]
        assert r["error"] is not None or r["total_trades"] == 0

    def test_empty_param_grid(self):
        opts, stocks = _make_test_data()
        results = rust_parallel_sweep(
            opts, stocks, _base_config(), _schema(), [],
        )
        assert results == []

    def test_deterministic(self):
        opts, stocks = _make_test_data()
        config = _base_config()
        schema = _schema()
        overrides = [
            {"label": "a", "profit_pct": 0.5},
            {"label": "b", "loss_pct": 0.5},
        ]

        r1 = rust_parallel_sweep(opts, stocks, config, schema, overrides, n_workers=1)
        r2 = rust_parallel_sweep(opts, stocks, config, schema, overrides)

        r1.sort(key=lambda x: x["label"])
        r2.sort(key=lambda x: x["label"])
        for a, b in zip(r1, r2):
            assert abs(a["total_return"] - b["total_return"]) < 1e-10


class TestGridSweepWrapper:

    def test_sorted_by_sharpe(self):
        opts, stocks = _make_test_data()
        overrides = [{"label": "a"}, {"label": "b", "profit_pct": 0.01}]
        results = rust_grid_sweep(opts, stocks, _base_config(), _schema(), overrides)
        sharpes = [r["sharpe_ratio"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)
