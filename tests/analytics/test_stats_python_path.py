"""Tests for BacktestStats.from_balance — covers the Rust compute_full_stats
path including period stats, lookback, turnover, herfindahl, trade stats,
summary text with monthly/turnover branches, and lookback_table."""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.analytics.stats import (
    BacktestStats, PeriodStats, LookbackReturns,
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


class TestFromBalanceReturnMetrics:
    def test_total_return(self):
        rets = [0.01] * 50
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        expected = (1.01 ** 50) - 1
        assert abs(s.total_return - expected) < 1e-6

    def test_annualized_return(self):
        rets = [0.001] * 252
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.annualized_return > 0

    def test_volatility(self):
        rets = [0.01, -0.01] * 50
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.volatility > 0

    def test_sharpe_and_sortino(self):
        rng = np.random.default_rng(42)
        rets = (rng.normal(0.005, 0.01, 100)).tolist()
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.sharpe_ratio != 0
        assert s.sortino_ratio != 0


class TestFromBalanceDrawdown:
    def test_drawdown_with_losses(self):
        rets = [0.01] * 10 + [-0.05] * 5 + [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.max_drawdown > 0
        assert s.max_drawdown_duration > 0

    def test_avg_drawdown(self):
        rets = [0.02] * 5 + [-0.03] * 3 + [0.02] * 5 + [-0.02] * 2 + [0.02] * 5
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.avg_drawdown > 0
        assert s.avg_drawdown_duration > 0

    def test_calmar_ratio(self):
        rets = [0.01] * 10 + [-0.03] + [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.calmar_ratio != 0

    def test_no_drawdown(self):
        rets = [0.01] * 20
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.max_drawdown == 0.0
        assert s.calmar_ratio == 0.0


class TestFromBalanceTailRatio:
    def test_tail_ratio_enough_data(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, 100).tolist()
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.tail_ratio > 0

    def test_tail_ratio_insufficient_data(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.tail_ratio == 0.0


class TestFromBalancePeriodStats:
    def test_daily_stats(self):
        rng = np.random.default_rng(99)
        rets = rng.normal(0.005, 0.01, 50).tolist()
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.daily.mean != 0
        assert s.daily.vol > 0
        assert s.daily.best > 0
        assert s.daily.worst < s.daily.best

    def test_skew_kurtosis_need_8_returns(self):
        rets = [0.01] * 3
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.daily.skew == 0.0
        assert s.daily.kurtosis == 0.0

    def test_skew_kurtosis_with_enough_data(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, 30).tolist()
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        # skew and kurtosis are non-zero with random data
        assert s.daily.skew != 0.0 or s.daily.kurtosis != 0.0


class TestFromBalanceLookback:
    def test_lookback_mtd_ytd(self):
        rets = [0.002] * 100
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.lookback.mtd is not None
        assert s.lookback.ytd is not None

    def test_lookback_trailing_periods(self):
        # 2 years of data
        rets = [0.001] * 504
        bal = _make_balance(rets, start="2018-06-01")
        s = BacktestStats.from_balance(bal)
        assert s.lookback.three_month is not None
        assert s.lookback.six_month is not None
        assert s.lookback.one_year is not None


class TestFromBalanceTurnoverHerfindahl:
    def test_turnover_with_stocks(self):
        rets = [0.001] * 50
        bal = _make_balance_with_stocks(rets)
        s = BacktestStats.from_balance(bal)
        assert s.turnover >= 0.0

    def test_herfindahl_with_stocks(self):
        rets = [0.001] * 50
        bal = _make_balance_with_stocks(rets)
        s = BacktestStats.from_balance(bal)
        assert s.herfindahl > 0.0

    def test_turnover_no_stocks(self):
        rets = [0.001] * 10
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.turnover == 0.0

    def test_herfindahl_no_stocks(self):
        rets = [0.001] * 10
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.herfindahl == 0.0


class TestFromBalanceTradeStats:
    def test_trade_stats_full(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        pnls = np.array([100.0, 200.0, -50.0, -30.0, 150.0])
        s = BacktestStats.from_balance(bal, trade_pnls=pnls)
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
        s = BacktestStats.from_balance(bal, trade_pnls=pnls)
        assert s.profit_factor == float("inf")

    def test_trade_stats_all_losses(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        pnls = np.array([-100.0, -200.0])
        s = BacktestStats.from_balance(bal, trade_pnls=pnls)
        assert s.profit_factor == 0.0
        assert s.largest_win == 0
        assert s.avg_win == 0

    def test_trade_stats_none(self):
        rets = [0.01] * 10
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal, trade_pnls=None)
        assert s.total_trades == 0


class TestSummaryText:
    def test_summary_minimal(self):
        rets = [0.01] * 5
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        text = s.summary()
        assert "Total Return" in text

    def test_summary_with_turnover(self):
        rets = [0.001] * 50
        bal = _make_balance_with_stocks(rets)
        s = BacktestStats.from_balance(bal)
        text = s.summary()
        assert "Turnover" in text


class TestLookbackTable:
    def test_lookback_table_nonempty(self):
        rets = [0.002] * 100
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
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
        s = BacktestStats.from_balance(bal)
        df = s.to_dataframe()
        assert "Total return" in df.index
        assert "Sharpe ratio" in df.index
        assert "Herfindahl index" in df.index


class TestFromBalanceSharpe:
    def test_positive_returns_with_variance(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.01, 0.005, 50).tolist()
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.sharpe_ratio > 0

    def test_all_positive_sortino_zero(self):
        rets = [0.01] * 50
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        # No downside returns -> sortino should be 0
        assert s.sortino_ratio == 0.0

    def test_mixed_returns_sortino_nonzero(self):
        rets = ([0.02, -0.01, 0.015, -0.005] * 10)
        bal = _make_balance(rets)
        s = BacktestStats.from_balance(bal)
        assert s.sortino_ratio != 0.0


class TestFromBalanceDispatch:
    """Test the from_balance classmethod."""

    def test_from_balance_empty(self):
        bal = pd.DataFrame(columns=["total capital", "% change"])
        s = BacktestStats.from_balance(bal)
        assert s.total_return == 0.0

    def test_from_balance_basic(self):
        rets = [0.01] * 20
        bal = _make_balance(rets)
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
