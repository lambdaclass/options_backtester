"""Tests for BacktestStats._from_balance_python — covers the Python fallback
path including period stats, lookback, turnover, herfindahl, trade stats,
summary text with monthly/turnover branches, and lookback_table."""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.analytics.stats import (
    BacktestStats, PeriodStats, LookbackReturns,
    _sharpe, _sortino,
    _compute_period_stats, _compute_lookback,
    _compute_turnover, _compute_herfindahl,
)


def _make_balance(returns, initial=100_000.0, start="2020-01-01"):
    dates = pd.date_range(start, periods=len(returns) + 1, freq="B")
    capital = [initial]
    for r in returns:
        capital.append(capital[-1] * (1 + r))
    df = pd.DataFrame({"total capital": capital}, index=dates)
    df["% change"] = df["total capital"].pct_change()
    return df


def _make_balance_with_stocks(returns, initial=100_000.0, start="2020-01-01"):
    """Balance with stock columns for turnover/herfindahl tests."""
    df = _make_balance(returns, initial, start)
    n = len(df)
    df["SPY"] = np.linspace(60000, 70000, n)
    df["SPY qty"] = 200
    df["IWM"] = np.linspace(30000, 25000, n)
    df["IWM qty"] = 150
    return df


# -- Force the Python path by calling _from_balance_python directly --

class TestFromBalancePythonReturnMetrics:
    def test_total_return(self):
        rets = [0.01] * 50
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        expected = (1.01 ** 50) - 1
        assert abs(s.total_return - expected) < 1e-6

    def test_annualized_return(self):
        rets = [0.001] * 252
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.annualized_return > 0

    def test_volatility(self):
        rets = [0.01, -0.01] * 50
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.volatility > 0

    def test_sharpe_and_sortino(self):
        rng = np.random.default_rng(42)
        rets = (rng.normal(0.005, 0.01, 100)).tolist()
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.sharpe_ratio != 0
        assert s.sortino_ratio != 0


class TestFromBalancePythonDrawdown:
    def test_drawdown_with_losses(self):
        rets = [0.01] * 10 + [-0.05] * 5 + [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.max_drawdown > 0
        assert s.max_drawdown_duration > 0

    def test_avg_drawdown(self):
        rets = [0.02] * 5 + [-0.03] * 3 + [0.02] * 5 + [-0.02] * 2 + [0.02] * 5
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.avg_drawdown > 0
        assert s.avg_drawdown_duration > 0

    def test_calmar_ratio(self):
        rets = [0.01] * 10 + [-0.03] + [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.calmar_ratio != 0

    def test_no_drawdown(self):
        rets = [0.01] * 20
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.max_drawdown == 0.0
        assert s.calmar_ratio == 0.0


class TestFromBalancePythonTailRatio:
    def test_tail_ratio_enough_data(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, 100).tolist()
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.tail_ratio > 0

    def test_tail_ratio_insufficient_data(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.tail_ratio == 0.0


class TestFromBalancePythonPeriodStats:
    def test_daily_stats(self):
        rng = np.random.default_rng(99)
        rets = rng.normal(0.005, 0.01, 50).tolist()
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.daily.mean != 0
        assert s.daily.vol > 0
        assert s.daily.best > 0
        assert s.daily.worst < s.daily.best

    def test_monthly_stats_multi_month(self):
        # Need >1 month of data
        rets = [0.002] * 60
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.monthly.mean > 0

    def test_yearly_stats_multi_year(self):
        # Need >1 year of data
        rets = [0.001] * 600
        bal = _make_balance(rets, start="2018-01-01")
        s = BacktestStats._from_balance_python(bal)
        assert s.yearly.mean > 0

    def test_skew_kurtosis_need_8_returns(self):
        rets = [0.01] * 3
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.daily.skew == 0.0
        assert s.daily.kurtosis == 0.0

    def test_skew_kurtosis_with_enough_data(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, 30).tolist()
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        # skew and kurtosis are non-zero with random data
        assert s.daily.skew != 0.0 or s.daily.kurtosis != 0.0


class TestFromBalancePythonLookback:
    def test_lookback_mtd_ytd(self):
        rets = [0.002] * 100
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.lookback.mtd is not None
        assert s.lookback.ytd is not None

    def test_lookback_trailing_periods(self):
        # 2 years of data
        rets = [0.001] * 504
        bal = _make_balance(rets, start="2018-06-01")
        s = BacktestStats._from_balance_python(bal)
        assert s.lookback.three_month is not None
        assert s.lookback.six_month is not None
        assert s.lookback.one_year is not None


class TestFromBalancePythonTurnoverHerfindahl:
    def test_turnover_with_stocks(self):
        rets = [0.001] * 50
        bal = _make_balance_with_stocks(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.turnover >= 0.0

    def test_herfindahl_with_stocks(self):
        rets = [0.001] * 50
        bal = _make_balance_with_stocks(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.herfindahl > 0.0

    def test_turnover_no_stocks(self):
        rets = [0.001] * 10
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.turnover == 0.0

    def test_herfindahl_no_stocks(self):
        rets = [0.001] * 10
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        assert s.herfindahl == 0.0


class TestFromBalancePythonTradeStats:
    def test_trade_stats_full(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        pnls = np.array([100.0, 200.0, -50.0, -30.0, 150.0])
        s = BacktestStats._from_balance_python(bal, trade_pnls=pnls)
        assert s.total_trades == 5
        assert s.wins == 3
        assert s.losses == 2
        assert s.win_pct == pytest.approx(60.0)
        assert s.largest_win == 200.0
        assert s.largest_loss == -50.0
        assert s.avg_win > 0
        assert s.avg_loss < 0
        assert s.avg_trade > 0

    def test_trade_stats_all_wins(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        pnls = np.array([100.0, 200.0])
        s = BacktestStats._from_balance_python(bal, trade_pnls=pnls)
        assert s.profit_factor == float("inf")

    def test_trade_stats_all_losses(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        pnls = np.array([-100.0, -200.0])
        s = BacktestStats._from_balance_python(bal, trade_pnls=pnls)
        assert s.profit_factor == 0.0
        assert s.largest_win == 0
        assert s.avg_win == 0

    def test_trade_stats_none(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal, trade_pnls=None)
        assert s.total_trades == 0


class TestSummaryText:
    def test_summary_with_monthly(self):
        rets = [0.002] * 60
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        text = s.summary()
        assert "Total Return" in text
        assert "Sharpe" in text
        # Monthly sharpe should appear
        assert "Monthly" in text

    def test_summary_with_turnover(self):
        rets = [0.001] * 50
        bal = _make_balance_with_stocks(rets)
        s = BacktestStats._from_balance_python(bal)
        text = s.summary()
        assert "Turnover" in text

    def test_summary_minimal(self):
        rets = [0.01] * 5
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        text = s.summary()
        assert "Total Return" in text


class TestLookbackTable:
    def test_lookback_table_nonempty(self):
        rets = [0.002] * 100
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        tbl = s.lookback_table()
        assert not tbl.empty
        assert "MTD" in tbl.columns

    def test_lookback_table_empty_when_no_data(self):
        s = BacktestStats()
        tbl = s.lookback_table()
        assert tbl.empty


class TestToDataframe:
    def test_has_expected_rows(self):
        rets = [0.002] * 60
        bal = _make_balance(rets)
        s = BacktestStats._from_balance_python(bal)
        df = s.to_dataframe()
        assert "Total return" in df.index
        assert "Sharpe ratio" in df.index
        assert "Herfindahl index" in df.index


# -- Unit tests for helper functions --

class TestSharpe:
    def test_positive_returns_with_variance(self):
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0.01, 0.005, 50))
        s = _sharpe(rets, 0.0, 252)
        assert s > 0

    def test_constant_positive_returns_zero_std(self):
        rets = pd.Series([0.01] * 50)
        # Constant returns → zero std → sharpe = 0
        assert _sharpe(rets, 0.0, 252) == 0.0

    def test_short_series(self):
        rets = pd.Series([0.01])
        assert _sharpe(rets, 0.0, 252) == 0.0

    def test_zero_std(self):
        rets = pd.Series([0.0] * 10)
        assert _sharpe(rets, 0.0, 252) == 0.0


class TestSortino:
    def test_positive_returns(self):
        rets = pd.Series([0.01] * 50)
        # No downside → sortino should be 0 (no downside deviation)
        assert _sortino(rets, 0.0, 252) == 0.0

    def test_mixed_returns(self):
        rets = pd.Series([0.02, -0.01, 0.015, -0.005] * 10)
        s = _sortino(rets, 0.0, 252)
        assert s != 0.0


class TestComputePeriodStats:
    def test_empty_returns(self):
        ps = _compute_period_stats(pd.Series(dtype=float), 0.0, 252)
        assert ps.mean == 0.0
        assert ps.vol == 0.0

    def test_non_empty(self):
        rets = pd.Series([0.01, -0.005, 0.008, 0.003, -0.002, 0.006, -0.001, 0.009, 0.002, 0.004])
        ps = _compute_period_stats(rets, 0.0, 252)
        assert ps.mean > 0
        assert ps.best == max(rets)
        assert ps.worst == min(rets)
        assert ps.skew != 0.0  # 10 samples ≥ 8


class TestComputeLookback:
    def test_short_series(self):
        tc = pd.Series([100.0], index=pd.date_range("2020-01-01", periods=1))
        lb = _compute_lookback(tc)
        assert lb.mtd is None

    def test_normal_series(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        tc = pd.Series(np.linspace(100000, 110000, 100), index=dates)
        lb = _compute_lookback(tc)
        assert lb.mtd is not None
        assert lb.ytd is not None


class TestComputeTurnover:
    def test_no_stock_columns(self):
        bal = pd.DataFrame({"total capital": [100, 101, 102]})
        assert _compute_turnover(bal) == 0.0

    def test_zero_total_capital(self):
        bal = pd.DataFrame({
            "total capital": [0, 0, 0],
            "SPY": [50, 51, 52],
            "SPY qty": [10, 10, 10],
        })
        assert _compute_turnover(bal) == 0.0

    def test_with_stocks(self):
        bal = pd.DataFrame({
            "total capital": [100000, 105000, 110000],
            "SPY": [60000, 63000, 70000],
            "SPY qty": [200, 200, 200],
            "IWM": [30000, 32000, 30000],
            "IWM qty": [150, 150, 150],
        })
        assert _compute_turnover(bal) >= 0.0

    def test_single_row(self):
        bal = pd.DataFrame({
            "total capital": [100000],
            "SPY": [60000],
            "SPY qty": [200],
        })
        assert _compute_turnover(bal) == 0.0


class TestComputeHerfindahl:
    def test_no_stock_columns(self):
        bal = pd.DataFrame({"total capital": [100, 101]})
        assert _compute_herfindahl(bal) == 0.0

    def test_single_stock(self):
        bal = pd.DataFrame({
            "total capital": [100000, 100000],
            "SPY": [100000, 100000],
            "SPY qty": [100, 100],
        })
        hhi = _compute_herfindahl(bal)
        assert abs(hhi - 1.0) < 1e-6  # single stock → concentration = 1

    def test_two_equal_stocks(self):
        bal = pd.DataFrame({
            "total capital": [100000, 100000],
            "SPY": [50000, 50000],
            "SPY qty": [100, 100],
            "IWM": [50000, 50000],
            "IWM qty": [100, 100],
        })
        hhi = _compute_herfindahl(bal)
        assert abs(hhi - 0.5) < 1e-6  # two equal → 0.25 + 0.25 = 0.5


class TestFromBalanceDispatch:
    """Test the from_balance classmethod dispatch (Rust → Python fallback)."""

    def test_from_balance_empty(self):
        bal = pd.DataFrame(columns=["total capital", "% change"])
        s = BacktestStats.from_balance(bal)
        assert s.total_return == 0.0

    def test_from_balance_falls_back_to_python(self):
        rets = [0.01] * 20
        bal = _make_balance(rets)
        # from_balance tries Rust first, falls back to Python
        s = BacktestStats.from_balance(bal)
        assert s.total_return > 0
        assert s.annualized_return > 0

    def test_from_balance_with_trade_pnls(self):
        rets = [0.01] * 20
        bal = _make_balance(rets)
        pnls = np.array([100.0, -50.0, 200.0])
        s = BacktestStats.from_balance(bal, trade_pnls=pnls)
        assert s.total_trades == 3


class TestFromBalanceRange:
    """Test the from_balance_range classmethod."""

    def test_empty_balance(self):
        bal = pd.DataFrame(columns=["total capital"])
        s = BacktestStats.from_balance_range(bal)
        assert s.total_return == 0.0

    def test_full_range(self):
        rets = [0.01] * 30
        bal = _make_balance(rets)
        s = BacktestStats.from_balance_range(bal)
        assert s.total_return > 0

    def test_with_start(self):
        rets = [0.01] * 30
        bal = _make_balance(rets, start="2020-01-01")
        # Slice from 10 business days in
        s = BacktestStats.from_balance_range(bal, start="2020-01-15")
        assert s.total_return > 0

    def test_with_end(self):
        rets = [0.01] * 30
        bal = _make_balance(rets, start="2020-01-01")
        s = BacktestStats.from_balance_range(bal, end="2020-01-15")
        assert s.total_return > 0

    def test_with_start_and_end(self):
        rets = [0.01] * 30
        bal = _make_balance(rets, start="2020-01-01")
        s = BacktestStats.from_balance_range(bal, start="2020-01-10", end="2020-01-20")
        assert s.total_return > 0

    def test_out_of_range_returns_empty(self):
        rets = [0.01] * 10
        bal = _make_balance(rets, start="2020-01-01")
        s = BacktestStats.from_balance_range(bal, start="2025-01-01")
        assert s.total_return == 0.0
