"""Parity tests: verify Rust and Python implementations produce identical results.

These tests call both the Rust (_ob_rust) and Python implementations of each
hot-path function and assert numerical equivalence via np.allclose.

Skipped automatically when the Rust extension is not installed.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
    from options_portfolio_backtester._ob_rust import (
        compute_stats as rust_compute_stats,
        compile_filter as rust_compile_filter,
        apply_filter as rust_apply_filter,
        compute_exit_mask as rust_compute_exit_mask,
        run_backtest_py as rust_run_backtest,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _pd_to_pl(df: pd.DataFrame) -> "pl.DataFrame":
    """Convert pandas DataFrame to polars for Rust interop."""
    return pl.from_pandas(df)

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed")


class TestStatsParity:
    """Verify Rust stats match Python stats."""

    def _python_stats(self, daily_returns, trade_pnls, risk_free_rate):
        """Reproduce Rust stats logic in pure Python for comparison."""
        n = len(daily_returns)
        if n == 0:
            return {}

        total_return = 1.0
        for r in daily_returns:
            total_return *= (1 + r)
        total_return -= 1.0

        years = n / 252.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        daily_rf = risk_free_rate / 252.0
        excess = [r - daily_rf for r in daily_returns]
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        sharpe = mean_excess / std_excess * np.sqrt(252) if std_excess > 0 else 0

        downside = [r for r in excess if r < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
        sortino = mean_excess / down_std * np.sqrt(252) if down_std > 0 else 0

        # Drawdown
        peak = 1.0
        equity = 1.0
        max_dd = 0.0
        for r in daily_returns:
            equity *= (1 + r)
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        calmar = annualized_return / max_dd if max_dd > 0 else 0

        # Trade stats
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = sum(abs(p) for p in trade_pnls if p <= 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        wins = sum(1 for p in trade_pnls if p > 0)
        win_rate = wins / len(trade_pnls) if len(trade_pnls) > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "total_trades": len(trade_pnls),
        }

    def test_basic_returns(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.003, -0.008]
        pnls = [100.0, -50.0, 200.0, -30.0, 150.0]

        rust = rust_compute_stats(returns, pnls, 0.02)
        py = self._python_stats(returns, pnls, 0.02)

        for key in ["total_return", "annualized_return", "sharpe_ratio",
                     "sortino_ratio", "max_drawdown", "profit_factor",
                     "win_rate", "total_trades"]:
            assert np.isclose(rust[key], py[key], rtol=1e-6), (
                f"{key}: rust={rust[key]} vs python={py[key]}"
            )

    def test_long_series(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.0005, 0.015, 2520))  # ~10 years
        pnls = list(np.random.normal(50, 200, 200))

        rust = rust_compute_stats(returns, pnls, 0.03)
        py = self._python_stats(returns, pnls, 0.03)

        for key in ["total_return", "annualized_return", "sharpe_ratio",
                     "max_drawdown", "profit_factor"]:
            assert np.isclose(rust[key], py[key], rtol=1e-4), (
                f"{key}: rust={rust[key]} vs python={py[key]}"
            )

    def test_empty(self):
        rust = rust_compute_stats([], [], 0.0)
        assert rust["total_return"] == 0.0
        assert rust["total_trades"] == 0

    def test_all_positive(self):
        returns = [0.01, 0.02, 0.015]
        pnls = [100.0, 200.0]

        rust = rust_compute_stats(returns, pnls, 0.0)
        assert rust["max_drawdown"] == 0.0
        assert rust["win_rate"] == 1.0

    def test_all_negative(self):
        returns = [-0.01, -0.02, -0.015]
        pnls = [-100.0, -200.0]

        rust = rust_compute_stats(returns, pnls, 0.0)
        assert rust["max_drawdown"] > 0
        assert rust["win_rate"] == 0.0
        assert rust["profit_factor"] == 0.0


class TestFilterParity:
    """Verify Rust filter produces same results as pandas eval."""

    def _make_df(self):
        return pd.DataFrame({
            "type": ["call", "put", "put", "call", "put"],
            "ask": [1.0, 2.0, 0.0, 3.0, 1.5],
            "underlying": ["SPX", "SPX", "AAPL", "SPX", "SPX"],
            "dte": [30, 90, 90, 150, 60],
            "strike": [4000.0, 4100.0, 4200.0, 4300.0, 4400.0],
        })

    def test_simple_eq(self):
        df = self._make_df()
        query = "type == 'put'"

        py_result = df[df.eval(query)]
        rust_result = rust_apply_filter(query, _pd_to_pl(df))

        assert len(py_result) == rust_result.height
        assert list(py_result["strike"].values) == rust_result["strike"].to_list()

    def test_compound_filter(self):
        df = self._make_df()
        query = "(type == 'put') & (ask > 0) & (underlying == 'SPX')"

        py_result = df[df.eval(query)]
        rust_result = rust_apply_filter(query, _pd_to_pl(df))

        assert len(py_result) == rust_result.height

    def test_range_filter(self):
        df = self._make_df()
        query = "(underlying == 'SPX') & (dte >= 60) & (dte <= 120)"

        py_result = df[df.eval(query)]
        rust_result = rust_apply_filter(query, _pd_to_pl(df))

        assert len(py_result) == rust_result.height

    def test_empty_result(self):
        df = self._make_df()
        query = "(type == 'put') & (dte >= 200)"

        py_result = df[df.eval(query)]
        rust_result = rust_apply_filter(query, _pd_to_pl(df))

        assert len(py_result) == 0
        assert rust_result.height == 0


class TestExitMaskParity:
    """Verify Rust exit mask matches Python threshold logic."""

    def _python_exit_mask(self, entry_costs, current_costs, profit_pct, loss_pct):
        mask = []
        for e, c in zip(entry_costs, current_costs):
            should_exit = False
            if e != 0:
                pnl_pct = (c - e) / abs(e)
                if profit_pct is not None and pnl_pct >= profit_pct:
                    should_exit = True
                if loss_pct is not None and pnl_pct <= -loss_pct:
                    should_exit = True
            mask.append(should_exit)
        return mask

    def test_profit_exit(self):
        entries = [100.0, 100.0, 100.0]
        currents = [160.0, 110.0, 80.0]

        rust = rust_compute_exit_mask(entries, currents, profit_pct=0.5)
        py = self._python_exit_mask(entries, currents, 0.5, None)
        assert rust == py

    def test_loss_exit(self):
        entries = [100.0, 100.0, 100.0]
        currents = [160.0, 110.0, 70.0]

        rust = rust_compute_exit_mask(entries, currents, loss_pct=0.2)
        py = self._python_exit_mask(entries, currents, None, 0.2)
        assert rust == py

    def test_both_thresholds(self):
        entries = [100.0, 100.0, 100.0, 100.0]
        currents = [160.0, 110.0, 70.0, 100.0]

        rust = rust_compute_exit_mask(entries, currents, profit_pct=0.5, loss_pct=0.2)
        py = self._python_exit_mask(entries, currents, 0.5, 0.2)
        assert rust == py

    def test_large_batch(self):
        np.random.seed(42)
        n = 10_000
        entries = list(np.random.uniform(50, 200, n))
        currents = list(np.random.uniform(30, 250, n))

        rust = rust_compute_exit_mask(entries, currents, profit_pct=0.5, loss_pct=0.3)
        py = self._python_exit_mask(entries, currents, 0.5, 0.3)
        assert rust == py


class TestCompiledFilter:
    """Test the CompiledFilter class."""

    def test_reuse_across_calls(self):
        df1 = _pd_to_pl(pd.DataFrame({
            "type": ["call", "put"],
            "ask": [1.0, 2.0],
        }))
        df2 = _pd_to_pl(pd.DataFrame({
            "type": ["put", "put", "call"],
            "ask": [0.0, 3.0, 1.0],
        }))

        f = rust_compile_filter("(type == 'put') & (ask > 0)")
        r1 = f.apply(df1)
        r2 = f.apply(df2)

        assert r1.height == 1
        assert r2.height == 1

    def test_repr(self):
        f = rust_compile_filter("dte >= 60")
        assert "Cmp" in repr(f)


class TestFullBacktestParity:
    """Verify Rust full backtest loop produces consistent results."""

    def test_rust_backtest_runs(self):
        """Smoke test: Rust backtest completes without error."""
        # Build minimal options data
        dates = ["2024-01-01"] * 4 + ["2024-01-15"] * 4
        opts = pd.DataFrame({
            "optionroot": ["A", "B", "C", "D"] * 2,
            "underlying": ["SPX"] * 8,
            "underlying_last": [4500.0] * 8,
            "quotedate": dates,
            "type": ["put", "put", "call", "put"] * 2,
            "expiration": ["2024-03-01"] * 8,
            "strike": [4400.0, 4300.0, 4500.0, 4200.0] * 2,
            "bid": [5.0, 8.0, 3.0, 12.0, 4.0, 7.0, 2.0, 11.0],
            "ask": [6.0, 9.0, 4.0, 13.0, 5.0, 8.0, 3.0, 12.0],
            "volume": [100] * 8,
            "open_interest": [1000] * 8,
            "dte": [60] * 8,
        })
        stocks = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-15"] * 2,
            "symbol": ["SPY", "SPY", "TLT", "TLT"],
            "adjClose": [450.0, 455.0, 100.0, 101.0],
        })

        config = {
            "allocation": {"stocks": 0.5, "options": 0.3, "cash": 0.2},
            "initial_capital": 100000.0,
            "shares_per_contract": 100,
            "rebalance_freq": 1,
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
            "rebalance_dates": ["2024-01-01", "2024-01-15"],
        }
        schema = {
            "contract": "optionroot",
            "date": "quotedate",
            "stocks_date": "date",
            "stocks_symbol": "symbol",
            "stocks_price": "adjClose",
        }

        opts_pl = _pd_to_pl(opts)
        stocks_pl = _pd_to_pl(stocks)

        balance, trade_log, stats = rust_run_backtest(
            opts_pl, stocks_pl, config, schema,
        )

        assert balance.height > 0
        assert isinstance(stats, dict)
        assert "total_return" in stats
        assert "final_cash" in stats
