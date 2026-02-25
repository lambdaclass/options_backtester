"""Tests for BacktestStats — including the fixed profit_factor."""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.analytics.stats import (
    BacktestStats, PeriodStats, LookbackReturns,
    _compute_period_stats, _compute_lookback, _compute_turnover, _compute_herfindahl,
)


def _make_balance(returns: list[float], initial: float = 100_000.0) -> pd.DataFrame:
    """Build a balance DataFrame from a list of daily returns."""
    dates = pd.date_range("2020-01-01", periods=len(returns) + 1, freq="B")
    capital = [initial]
    for r in returns:
        capital.append(capital[-1] * (1 + r))
    df = pd.DataFrame({"total capital": capital}, index=dates)
    df["% change"] = df["total capital"].pct_change()
    return df


class TestProfitFactor:
    """Critical test: profit_factor must be dollar-based, not count-based."""

    def test_profit_factor_dollar_ratio(self):
        """profit_factor = gross_profit / gross_loss in dollars."""
        # 2 wins ($100, $200) and 1 loss (-$50)
        trade_pnls = np.array([100.0, 200.0, -50.0])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        # gross_profit = 300, gross_loss = 50, factor = 6.0
        assert stats.profit_factor == 6.0

    def test_profit_factor_not_count_ratio(self):
        """The old bug used win_count/loss_count. Verify it's NOT that."""
        # 1 big win ($1000) and 3 small losses (-$10 each)
        trade_pnls = np.array([1000.0, -10.0, -10.0, -10.0])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        # Dollar: 1000/30 = 33.33, Count would be: 1/3 = 0.33
        assert abs(stats.profit_factor - 33.333333) < 0.01

    def test_profit_factor_no_losses(self):
        trade_pnls = np.array([100.0, 200.0])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        assert stats.profit_factor == float("inf")

    def test_profit_factor_no_wins(self):
        trade_pnls = np.array([-100.0, -200.0])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        assert stats.profit_factor == 0.0


class TestReturnMetrics:
    def test_total_return(self):
        balance = _make_balance([0.01] * 252)  # 1% daily for a year
        stats = BacktestStats.from_balance(balance)
        # (1.01)^252 - 1 ~ 11.28
        assert stats.total_return > 10.0

    def test_zero_return(self):
        balance = _make_balance([0.0] * 10)
        stats = BacktestStats.from_balance(balance)
        assert abs(stats.total_return) < 1e-10

    def test_sharpe_positive(self):
        # Use varying positive returns so std > 0
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 252))
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.sharpe_ratio > 0


class TestDrawdown:
    def test_max_drawdown(self):
        # Go up, then crash, then recover
        returns = [0.10, 0.10, -0.30, -0.20, 0.10, 0.10]
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown > 0

    def test_no_drawdown(self):
        returns = [0.01] * 10
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown == 0.0

    def test_drawdown_duration(self):
        # Drop then flat then recover
        returns = [0.10, -0.20, -0.01, -0.01, 0.30]
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown_duration >= 2


class TestTradeStats:
    def test_wins_losses_count(self):
        trade_pnls = np.array([100, -50, 200, -30, 150])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        assert stats.wins == 3
        assert stats.losses == 2
        assert stats.total_trades == 5

    def test_win_pct(self):
        trade_pnls = np.array([100, -50, 200, -30, 150])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        assert abs(stats.win_pct - 60.0) < 1e-10

    def test_empty_balance(self):
        balance = pd.DataFrame()
        stats = BacktestStats.from_balance(balance)
        assert stats.total_trades == 0
        assert stats.total_return == 0.0

    def test_no_trade_pnls(self):
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance)
        assert stats.total_trades == 0
        assert stats.total_return > 0


class TestToDataframe:
    def test_shape(self):
        trade_pnls = np.array([100, -50])
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        df = stats.to_dataframe()
        assert df.shape[0] >= 30  # expanded stats (period, lookback, portfolio)
        assert df.shape[1] == 1

    def test_summary_string(self):
        balance = _make_balance([0.01] * 10)
        stats = BacktestStats.from_balance(balance)
        s = stats.summary()
        assert "Sharpe" in s
        assert "Max Drawdown" in s


class TestPeriodStats:
    def test_daily_stats_computed(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 252))
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.daily.mean != 0
        assert stats.daily.vol != 0
        assert stats.daily.sharpe != 0
        assert stats.daily.best > 0
        assert stats.daily.worst < 0

    def test_monthly_stats_computed(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 504))  # 2 years
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.monthly.mean != 0
        assert stats.monthly.vol != 0
        assert stats.monthly.sharpe != 0

    def test_yearly_stats_computed(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 756))  # 3 years
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.yearly.mean != 0
        assert stats.yearly.best > 0
        assert stats.yearly.worst != 0

    def test_skew_kurtosis_with_enough_data(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 252))
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.daily.skew != 0
        assert stats.daily.kurtosis != 0

    def test_skew_kurtosis_not_computed_with_few_points(self):
        returns = pd.Series([0.01, 0.02, -0.01])
        ps = _compute_period_stats(returns, 0.0, 252)
        assert ps.skew == 0  # not enough data (< 8)


class TestAvgDrawdown:
    def test_avg_drawdown_depth(self):
        returns = [0.10, -0.15, -0.05, 0.30, 0.05, -0.10, 0.20]
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.avg_drawdown > 0
        assert stats.avg_drawdown <= stats.max_drawdown

    def test_avg_drawdown_duration(self):
        returns = [0.10, -0.15, -0.05, 0.30, 0.05, -0.10, 0.20]
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.avg_drawdown_duration > 0
        assert stats.avg_drawdown_duration <= stats.max_drawdown_duration


class TestLookbackReturns:
    def test_mtd_and_ytd(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 504))
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.mtd is not None
        assert stats.lookback.ytd is not None

    def test_one_year_return(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 504))
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.one_year is not None

    def test_lookback_table(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 504))
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        table = stats.lookback_table()
        assert not table.empty
        assert "MTD" in table.columns

    def test_short_series_lookback_equals_total(self):
        returns = [0.01] * 10
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        # For periods longer than the data, lookback == total return
        assert stats.lookback.ten_year is not None
        assert abs(stats.lookback.ten_year - stats.total_return) < 1e-6


class TestTurnover:
    def test_turnover_zero_for_no_stocks(self):
        balance = _make_balance([0.01] * 10)
        assert _compute_turnover(balance) == 0.0

    def test_turnover_computed_with_stocks(self):
        rng = np.random.RandomState(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        total = 100_000 + np.cumsum(rng.normal(100, 500, 20))
        spy = total * 0.6 + rng.normal(0, 500, 20)
        balance = pd.DataFrame({
            "total capital": total,
            "SPY": spy,
            "SPY qty": spy / 300,
        }, index=dates)
        turnover = _compute_turnover(balance)
        assert turnover >= 0


class TestHerfindahl:
    def test_single_stock_hhi_is_one(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        balance = pd.DataFrame({
            "total capital": [100_000] * 10,
            "SPY": [100_000] * 10,
            "SPY qty": [300] * 10,
        }, index=dates)
        hhi = _compute_herfindahl(balance)
        assert abs(hhi - 1.0) < 0.01

    def test_two_equal_stocks_hhi(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        balance = pd.DataFrame({
            "total capital": [100_000] * 10,
            "SPY": [50_000] * 10,
            "SPY qty": [150] * 10,
            "QQQ": [50_000] * 10,
            "QQQ qty": [200] * 10,
        }, index=dates)
        hhi = _compute_herfindahl(balance)
        # 0.5^2 + 0.5^2 = 0.5
        assert abs(hhi - 0.5) < 0.01


class TestFromBalanceRange:
    def test_slice_start(self):
        returns = [0.01] * 20
        balance = _make_balance(returns)
        mid_date = balance.index[10]
        stats = BacktestStats.from_balance_range(balance, start=str(mid_date))
        # Should compute stats on roughly half the data
        assert stats.total_return > 0

    def test_slice_end(self):
        returns = [0.01] * 20
        balance = _make_balance(returns)
        mid_date = balance.index[10]
        stats = BacktestStats.from_balance_range(balance, end=str(mid_date))
        full_stats = BacktestStats.from_balance(balance)
        assert stats.total_return < full_stats.total_return

    def test_slice_both(self):
        returns = [0.01] * 30
        balance = _make_balance(returns)
        start = str(balance.index[5])
        end = str(balance.index[15])
        stats = BacktestStats.from_balance_range(balance, start=start, end=end)
        assert stats.total_return > 0

    def test_empty_balance(self):
        balance = pd.DataFrame()
        stats = BacktestStats.from_balance_range(balance)
        assert stats.total_return == 0.0

    def test_no_slice(self):
        returns = [0.01] * 10
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance_range(balance)
        full_stats = BacktestStats.from_balance(balance)
        assert abs(stats.total_return - full_stats.total_return) < 1e-6
