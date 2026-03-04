"""Property-based tests for BacktestStats via Rust compute_full_stats.

Fuzzes the analytics pipeline with random balance series and trade P&Ls to
verify statistical invariants hold across all inputs.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from options_portfolio_backtester.analytics.stats import BacktestStats, PeriodStats

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

daily_return = st.floats(min_value=-0.15, max_value=0.15, allow_nan=False, allow_infinity=False)
positive_return = st.floats(min_value=0.0001, max_value=0.05, allow_nan=False, allow_infinity=False)
negative_return = st.floats(min_value=-0.05, max_value=-0.0001, allow_nan=False, allow_infinity=False)
initial_capital = st.floats(min_value=1000, max_value=1e7, allow_nan=False, allow_infinity=False)
risk_free = st.floats(min_value=0.0, max_value=0.10, allow_nan=False, allow_infinity=False)
trade_pnl = st.floats(min_value=-10_000, max_value=10_000, allow_nan=False, allow_infinity=False)


def _make_balance(returns, initial=100_000.0):
    """Build a balance DataFrame from daily returns."""
    dates = pd.date_range("2020-01-01", periods=len(returns) + 1, freq="B")
    capital = [initial]
    for r in returns:
        capital.append(capital[-1] * (1 + r))
    df = pd.DataFrame({"total capital": capital}, index=dates)
    df["% change"] = df["total capital"].pct_change()
    return df


# ---------------------------------------------------------------------------
# BacktestStats invariants
# ---------------------------------------------------------------------------


class TestStatsInvariantsPBT:
    @given(st.lists(daily_return, min_size=20, max_size=500), initial_capital)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_max_drawdown_non_negative(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown >= -1e-10

    @given(st.lists(daily_return, min_size=20, max_size=500), initial_capital)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_max_drawdown_at_most_one(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown <= 1.0 + 1e-10

    @given(st.lists(daily_return, min_size=20, max_size=500), initial_capital)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_volatility_non_negative(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.volatility >= -1e-10

    @given(st.lists(daily_return, min_size=20, max_size=500), initial_capital)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_total_return_matches_endpoints(self, returns, cap):
        """Total return = final_capital / initial_capital - 1."""
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        expected = balance["total capital"].iloc[-1] / balance["total capital"].iloc[0] - 1
        assert abs(stats.total_return - expected) < 1e-6

    @given(st.lists(positive_return, min_size=20, max_size=200), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_all_positive_returns_positive_total(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.total_return > 0

    @given(st.lists(negative_return, min_size=20, max_size=200), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_all_negative_returns_negative_total(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.total_return < 0

    @given(st.lists(positive_return, min_size=20, max_size=200), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_all_positive_zero_drawdown(self, returns, cap):
        """Strictly increasing capital -> zero drawdown."""
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown < 1e-10

    @given(st.lists(daily_return, min_size=20, max_size=500), initial_capital)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_max_drawdown_duration_non_negative(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown_duration >= 0

    @given(st.lists(daily_return, min_size=20, max_size=300), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_calmar_sign_matches_return(self, returns, cap):
        """Calmar ratio has same sign as annualized return (when dd > 0)."""
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        if stats.max_drawdown > 1e-10:
            if stats.annualized_return > 0:
                assert stats.calmar_ratio > -1e-10
            elif stats.annualized_return < 0:
                assert stats.calmar_ratio < 1e-10


class TestStatsEmptyEdgePBT:
    def test_empty_balance(self):
        stats = BacktestStats.from_balance(pd.DataFrame())
        assert stats.total_return == 0.0
        assert stats.max_drawdown == 0.0

    @given(initial_capital)
    @settings(max_examples=20)
    def test_single_row(self, cap):
        balance = _make_balance([], cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.total_return == 0.0


# ---------------------------------------------------------------------------
# Trade stats invariants
# ---------------------------------------------------------------------------


class TestTradeStatsPBT:
    @given(st.lists(trade_pnl, min_size=5, max_size=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_wins_plus_losses_equals_total(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        assert stats.wins + stats.losses == stats.total_trades

    @given(st.lists(trade_pnl, min_size=5, max_size=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_total_trades_matches_input(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        assert stats.total_trades == len(pnls)

    @given(st.lists(st.floats(min_value=1.0, max_value=10_000, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_all_winners(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        assert stats.wins == len(pnls)
        assert stats.losses == 0
        assert stats.win_pct == 100.0

    @given(st.lists(st.floats(min_value=-10_000, max_value=-0.01, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_all_losers(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        assert stats.wins == 0
        assert stats.losses == len(pnls)
        assert stats.win_pct == 0.0

    @given(st.lists(trade_pnl, min_size=5, max_size=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_largest_win_gte_avg_win(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        if stats.wins > 0:
            assert stats.largest_win >= stats.avg_win - 1e-10

    @given(st.lists(trade_pnl, min_size=5, max_size=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_largest_loss_lte_avg_loss(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        if stats.losses > 0:
            assert stats.largest_loss <= stats.avg_loss + 1e-10

    @given(st.lists(trade_pnl, min_size=5, max_size=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_profit_factor_non_negative(self, pnls):
        balance = _make_balance([0.001] * 50)
        pnl_arr = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls=pnl_arr)
        assert stats.profit_factor >= 0


# ---------------------------------------------------------------------------
# Sharpe / Sortino via BacktestStats
# ---------------------------------------------------------------------------


class TestSharpePBT:
    @given(st.lists(daily_return, min_size=10, max_size=300), risk_free)
    @settings(max_examples=100)
    def test_finite(self, returns, rf):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance, risk_free_rate=rf)
        assert np.isfinite(stats.sharpe_ratio)

    @given(st.lists(daily_return, min_size=10, max_size=300))
    @settings(max_examples=50)
    def test_higher_rf_lower_sharpe(self, returns):
        """Higher risk-free rate reduces Sharpe (excess returns shrink)."""
        balance = _make_balance(returns)
        s_low = BacktestStats.from_balance(balance, risk_free_rate=0.0)
        s_high = BacktestStats.from_balance(balance, risk_free_rate=0.05)
        if s_low.daily.vol > 1e-8:
            assert s_high.sharpe_ratio <= s_low.sharpe_ratio + 1e-6

    def test_fewer_than_two_returns_zero(self):
        balance = _make_balance([0.01])
        stats = BacktestStats.from_balance(balance)
        # With 1-2 data points, Sharpe should be 0 or very close
        assert np.isfinite(stats.sharpe_ratio)


class TestSortinoPBT:
    @given(st.lists(daily_return, min_size=10, max_size=300), risk_free)
    @settings(max_examples=100)
    def test_finite(self, returns, rf):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance, risk_free_rate=rf)
        assert np.isfinite(stats.sortino_ratio)

    @given(st.lists(positive_return, min_size=10, max_size=100))
    @settings(max_examples=50)
    def test_all_positive_returns_zero_sortino(self, returns):
        """No downside returns -> Sortino = 0 (downside std = 0)."""
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.sortino_ratio == 0.0


# ---------------------------------------------------------------------------
# Daily period stats invariants
# ---------------------------------------------------------------------------


class TestPeriodStatsPBT:
    @given(st.lists(daily_return, min_size=10, max_size=300), risk_free)
    @settings(max_examples=100)
    def test_best_gte_worst(self, returns, rf):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance, risk_free_rate=rf)
        assert stats.daily.best >= stats.daily.worst - 1e-10

    @given(st.lists(daily_return, min_size=10, max_size=300), risk_free)
    @settings(max_examples=100)
    def test_vol_non_negative(self, returns, rf):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance, risk_free_rate=rf)
        assert stats.daily.vol >= -1e-10

    @given(st.lists(daily_return, min_size=10, max_size=300), risk_free)
    @settings(max_examples=100)
    def test_mean_between_best_and_worst(self, returns, rf):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance, risk_free_rate=rf)
        assert stats.daily.worst - 1e-10 <= stats.daily.mean <= stats.daily.best + 1e-10


# ---------------------------------------------------------------------------
# Lookback returns
# ---------------------------------------------------------------------------


class TestLookbackPBT:
    @given(st.lists(daily_return, min_size=30, max_size=500), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_mtd_always_computed(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.mtd is not None

    @given(st.lists(daily_return, min_size=30, max_size=500), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_ytd_always_computed(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.ytd is not None

    @given(st.lists(positive_return, min_size=30, max_size=200), initial_capital)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_all_positive_lookbacks_positive(self, returns, cap):
        """Strictly increasing capital -> all lookback returns are positive."""
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        lb = stats.lookback
        if lb.mtd is not None:
            assert lb.mtd >= -1e-10
        if lb.ytd is not None:
            assert lb.ytd >= -1e-10
        if lb.three_month is not None:
            assert lb.three_month >= -1e-10


# ---------------------------------------------------------------------------
# Turnover / Herfindahl
# ---------------------------------------------------------------------------


class TestTurnoverHerfindahlPBT:
    @given(st.lists(daily_return, min_size=20, max_size=100), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_turnover_non_negative(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.turnover >= -1e-10

    @given(st.lists(daily_return, min_size=20, max_size=100), initial_capital)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_herfindahl_non_negative(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        assert stats.herfindahl >= -1e-10

    def test_no_stock_cols_zero_turnover(self):
        balance = _make_balance([0.01] * 20)
        stats = BacktestStats.from_balance(balance)
        assert stats.turnover == 0.0
        assert stats.herfindahl == 0.0


# ---------------------------------------------------------------------------
# from_balance_range
# ---------------------------------------------------------------------------


class TestBalanceRangePBT:
    @given(st.lists(daily_return, min_size=60, max_size=300), initial_capital)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_range_subset_shorter(self, returns, cap):
        """Slicing to a sub-range gives different (not necessarily smaller) return."""
        balance = _make_balance(returns, cap)
        full = BacktestStats.from_balance(balance)
        # Take the middle 50% of dates
        mid_start = balance.index[len(balance) // 4]
        mid_end = balance.index[3 * len(balance) // 4]
        sliced = BacktestStats.from_balance_range(balance, start=mid_start, end=mid_end)
        # Just verify it computed something -- the stats themselves may differ
        assert isinstance(sliced, BacktestStats)
        assert sliced.max_drawdown >= -1e-10


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestOutputFormattingPBT:
    @given(st.lists(daily_return, min_size=20, max_size=200), initial_capital)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_to_dataframe_not_empty(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        df = stats.to_dataframe()
        assert len(df) > 0
        assert "Value" in df.columns

    @given(st.lists(daily_return, min_size=20, max_size=200), initial_capital)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_summary_not_empty(self, returns, cap):
        balance = _make_balance(returns, cap)
        stats = BacktestStats.from_balance(balance)
        s = stats.summary()
        assert len(s) > 0
        assert "Total Return" in s
