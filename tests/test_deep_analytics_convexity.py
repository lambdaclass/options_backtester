"""Deep analytics, convexity, dispatch, and data provider tests.

Covers:
- BacktestStats edge cases (empty, single-row, lookback, period stats)
- Convexity scoring internals (_find_target_put, _convexity_ratio)
- Convexity backtest helpers (_monthly_rebalance_dates, _stock_price_on, _find_date_range)
- Convexity allocator strategies
- Dispatch layer
- Data schema and filter DSL
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.analytics.stats import (
    BacktestStats,
    PeriodStats,
    LookbackReturns,
)
from options_portfolio_backtester.convexity.scoring import (
    _find_target_put,
    _convexity_ratio,
)
from options_portfolio_backtester.convexity.backtest import (
    _monthly_rebalance_dates,
    _stock_price_on,
    _find_date_range,
    _close_position,
    run_unhedged,
    BacktestResult,
)
from options_portfolio_backtester.convexity.config import (
    BacktestConfig,
    InstrumentConfig,
    default_config,
)
from options_portfolio_backtester.convexity.allocator import (
    pick_cheapest,
    allocate_equal_weight,
    allocate_inverse_vol,
)
from options_portfolio_backtester import _ob_rust
from options_portfolio_backtester.data.schema import Schema, Field, Filter


# ===========================================================================
# BacktestStats edge cases
# ===========================================================================


def _make_balance(n_days=252, start_capital=100_000, daily_return=0.0004):
    """Create a synthetic balance DataFrame with realistic structure."""
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    capital = [start_capital]
    for i in range(1, n_days):
        capital.append(capital[-1] * (1 + daily_return + np.random.normal(0, 0.005)))
    df = pd.DataFrame({"total capital": capital}, index=dates)
    df["% change"] = df["total capital"].pct_change()
    return df


class TestStatsEmpty:
    def test_empty_balance(self):
        stats = BacktestStats.from_balance(pd.DataFrame())
        assert stats.total_return == 0.0
        assert stats.sharpe_ratio == 0.0

    def test_single_row_balance(self):
        df = pd.DataFrame(
            {"total capital": [100_000], "% change": [np.nan]},
            index=[pd.Timestamp("2020-01-01")],
        )
        stats = BacktestStats.from_balance(df)
        assert stats.total_return == 0.0


class TestStatsAccuracy:
    def test_total_return_positive_market(self):
        np.random.seed(42)
        balance = _make_balance(n_days=252, daily_return=0.001)
        stats = BacktestStats.from_balance(balance)
        # With strong positive daily return and fixed seed, total return should be positive
        assert stats.total_return > 0

    def test_total_return_negative_market(self):
        np.random.seed(42)
        balance = _make_balance(n_days=252, daily_return=-0.002)
        stats = BacktestStats.from_balance(balance)
        assert stats.total_return < 0

    def test_sharpe_positive_for_good_market(self):
        np.random.seed(42)
        balance = _make_balance(n_days=252, daily_return=0.001)
        stats = BacktestStats.from_balance(balance)
        assert stats.sharpe_ratio > 0

    def test_max_drawdown_non_negative(self):
        np.random.seed(42)
        balance = _make_balance(n_days=252)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown >= 0

    def test_volatility_non_negative(self):
        np.random.seed(42)
        balance = _make_balance()
        stats = BacktestStats.from_balance(balance)
        assert stats.volatility >= 0

    def test_calmar_ratio_computed(self):
        np.random.seed(42)
        balance = _make_balance(n_days=252, daily_return=0.001)
        stats = BacktestStats.from_balance(balance)
        if stats.max_drawdown > 0:
            assert abs(stats.calmar_ratio - stats.annualized_return / stats.max_drawdown) < 0.01


class TestPeriodStatsDetail:
    def test_daily_stats_populated(self):
        np.random.seed(42)
        balance = _make_balance(n_days=252)
        stats = BacktestStats.from_balance(balance)
        assert stats.daily.vol > 0
        assert stats.daily.best > stats.daily.worst

    def test_monthly_stats_populated(self):
        np.random.seed(42)
        balance = _make_balance(n_days=504)  # ~2 years
        stats = BacktestStats.from_balance(balance)
        assert stats.monthly.vol > 0

    def test_yearly_stats_with_enough_data(self):
        np.random.seed(42)
        balance = _make_balance(n_days=756)  # ~3 years
        stats = BacktestStats.from_balance(balance)
        assert stats.yearly.mean != 0.0 or stats.yearly.vol != 0.0

    def test_skew_kurtosis_need_8_samples(self):
        # Only 5 daily returns → skew/kurtosis should be 0
        balance = _make_balance(n_days=6)
        stats = BacktestStats.from_balance(balance)
        assert stats.daily.skew == 0.0
        assert stats.daily.kurtosis == 0.0


class TestLookbackReturns:
    def test_lookback_mtd(self):
        balance = _make_balance(n_days=252)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.mtd is not None

    def test_lookback_ytd(self):
        balance = _make_balance(n_days=252)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.ytd is not None

    def test_lookback_one_year(self):
        balance = _make_balance(n_days=300)
        stats = BacktestStats.from_balance(balance)
        assert stats.lookback.one_year is not None

    def test_lookback_short_data_still_computes(self):
        """Even short data computes lookback by finding closest available date."""
        np.random.seed(42)
        balance = _make_balance(n_days=30)
        stats = BacktestStats.from_balance(balance)
        # With 30 days of data, the 10yr lookback uses the earliest available date
        # so it returns the total return (not None)
        assert stats.lookback.mtd is not None


class TestTradeStats:
    def test_with_pnls(self):
        balance = _make_balance(n_days=252)
        pnls = np.array([100, -50, 200, -30, 150, -80, 50])
        stats = BacktestStats.from_balance(balance, trade_pnls=pnls)
        assert stats.total_trades == 7
        assert stats.wins == 4
        assert stats.losses == 3
        assert stats.win_pct > 0
        assert stats.profit_factor > 0
        assert stats.largest_win == 200
        assert stats.largest_loss == -80
        assert stats.avg_trade > 0

    def test_all_winners(self):
        balance = _make_balance(n_days=252)
        pnls = np.array([10, 20, 30])
        stats = BacktestStats.from_balance(balance, trade_pnls=pnls)
        assert stats.wins == 3
        assert stats.losses == 0
        assert stats.profit_factor == float("inf")

    def test_all_losers(self):
        balance = _make_balance(n_days=252)
        pnls = np.array([-10, -20, -30])
        stats = BacktestStats.from_balance(balance, trade_pnls=pnls)
        assert stats.wins == 0
        assert stats.losses == 3

    def test_no_pnls(self):
        balance = _make_balance(n_days=252)
        stats = BacktestStats.from_balance(balance, trade_pnls=None)
        assert stats.total_trades == 0


class TestBalanceRange:
    def test_slice_by_date(self):
        balance = _make_balance(n_days=252)
        stats = BacktestStats.from_balance_range(
            balance, start="2020-03-01", end="2020-06-30"
        )
        assert stats.total_return != 0.0 or len(balance) < 10

    def test_empty_slice(self):
        balance = _make_balance(n_days=10)
        stats = BacktestStats.from_balance_range(balance, start="2025-01-01")
        assert stats.total_return == 0.0


class TestToDataframe:
    def test_output_is_dataframe(self):
        balance = _make_balance()
        stats = BacktestStats.from_balance(balance)
        df = stats.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "Total return" in df.index

    def test_summary_string(self):
        balance = _make_balance()
        stats = BacktestStats.from_balance(balance)
        s = stats.summary()
        assert "Total Return" in s
        assert "Sharpe" in s


class TestTurnoverAndHerfindahl:
    def test_turnover_with_stocks(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame({
            "total capital": np.linspace(100_000, 110_000, 10),
            "AAPL": np.linspace(50_000, 55_000, 10),
            "AAPL qty": [100] * 10,
            "GOOG": np.linspace(50_000, 55_000, 10),
            "GOOG qty": [50] * 10,
        }, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert stats.turnover >= 0

    def test_turnover_no_stocks(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({"total capital": [100_000] * 5}, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert stats.turnover == 0.0

    def test_herfindahl_single_stock(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({
            "total capital": [100_000] * 5,
            "AAPL": [100_000] * 5,
            "AAPL qty": [100] * 5,
        }, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert abs(stats.herfindahl - 1.0) < 0.01

    def test_herfindahl_two_equal_stocks(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({
            "total capital": [100_000] * 5,
            "AAPL": [50_000] * 5,
            "AAPL qty": [100] * 5,
            "GOOG": [50_000] * 5,
            "GOOG qty": [50] * 5,
        }, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert abs(stats.herfindahl - 0.5) < 0.01


# ===========================================================================
# Sharpe / Sortino helpers
# ===========================================================================


class TestSharpeViaStats:
    def test_positive_returns(self):
        dates = pd.bdate_range("2020-01-01", periods=251)
        rets = [0.01, 0.02, 0.015, 0.01, 0.005] * 50
        capital = [100_000.0]
        for r in rets:
            capital.append(capital[-1] * (1 + r))
        df = pd.DataFrame({"total capital": capital}, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert stats.sharpe_ratio > 0

    def test_single_value_finite(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame({"total capital": [100_000, 101_000]}, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert np.isfinite(stats.sharpe_ratio)


class TestSortinoViaStats:
    def test_no_downside_returns_zero(self):
        rets = [0.01, 0.02, 0.03] * 10
        dates = pd.bdate_range("2020-01-01", periods=len(rets) + 1)
        capital = [100_000.0]
        for r in rets:
            capital.append(capital[-1] * (1 + r))
        df = pd.DataFrame({"total capital": capital}, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert stats.sortino_ratio == 0.0

    def test_with_downside(self):
        rets = [0.01, -0.02, 0.015, -0.01, 0.005] * 50
        dates = pd.bdate_range("2020-01-01", periods=len(rets) + 1)
        capital = [100_000.0]
        for r in rets:
            capital.append(capital[-1] * (1 + r))
        df = pd.DataFrame({"total capital": capital}, index=dates)
        df["% change"] = df["total capital"].pct_change()
        stats = BacktestStats.from_balance(df)
        assert isinstance(stats.sortino_ratio, float)


# ===========================================================================
# Convexity scoring internals
# ===========================================================================


class TestFindTargetPut:
    def test_exact_match(self):
        deltas = np.array([-0.05, -0.10, -0.15, -0.20, -0.25])
        dtes = np.array([30, 30, 30, 30, 30], dtype=np.int32)
        asks = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        idx = _find_target_put(deltas, dtes, asks, -0.15, 14, 60)
        assert idx == 2

    def test_closest_match(self):
        deltas = np.array([-0.05, -0.10, -0.20, -0.25])
        dtes = np.array([30, 30, 30, 30], dtype=np.int32)
        asks = np.array([1.0, 1.5, 2.5, 3.0])
        idx = _find_target_put(deltas, dtes, asks, -0.15, 14, 60)
        # -0.10 and -0.20 are equidistant; should pick one
        assert idx in {1, 2}

    def test_dte_filter_excludes_short(self):
        deltas = np.array([-0.15])
        dtes = np.array([10], dtype=np.int32)
        asks = np.array([1.0])
        idx = _find_target_put(deltas, dtes, asks, -0.15, 14, 60)
        assert idx is None

    def test_dte_filter_excludes_long(self):
        deltas = np.array([-0.15])
        dtes = np.array([90], dtype=np.int32)
        asks = np.array([1.0])
        idx = _find_target_put(deltas, dtes, asks, -0.15, 14, 60)
        assert idx is None

    def test_zero_ask_excluded(self):
        deltas = np.array([-0.15])
        dtes = np.array([30], dtype=np.int32)
        asks = np.array([0.0])
        assert _find_target_put(deltas, dtes, asks, -0.15, 14, 60) is None

    def test_nan_delta_excluded(self):
        deltas = np.array([np.nan, -0.20])
        dtes = np.array([30, 30], dtype=np.int32)
        asks = np.array([1.0, 2.0])
        idx = _find_target_put(deltas, dtes, asks, -0.15, 14, 60)
        assert idx == 1

    def test_all_excluded(self):
        deltas = np.array([np.nan])
        dtes = np.array([5], dtype=np.int32)
        asks = np.array([0.0])
        assert _find_target_put(deltas, dtes, asks, -0.15, 14, 60) is None

    def test_empty_arrays(self):
        idx = _find_target_put(
            np.array([]), np.array([], dtype=np.int32), np.array([]),
            -0.15, 14, 60,
        )
        assert idx is None


class TestConvexityRatio:
    def test_basic_computation(self):
        # strike=100, underlying=110, ask=2, tail_drop=0.20
        # tail_price = 110 * 0.8 = 88
        # tail_payoff = max(100 - 88, 0) * 100 = 1200
        # annual_cost = 2 * 100 * 12 = 2400
        # ratio = 1200 / 2400 = 0.5
        ratio, payoff, cost = _convexity_ratio(100, 110, 2, 0.20)
        assert abs(ratio - 0.5) < 0.001
        assert abs(payoff - 1200.0) < 0.01
        assert abs(cost - 2400.0) < 0.01

    def test_otm_put_zero_payoff(self):
        # strike=80, underlying=110, tail_price=88 → payoff=max(80-88,0)=0
        ratio, payoff, cost = _convexity_ratio(80, 110, 2, 0.20)
        assert ratio == 0.0
        assert payoff == 0.0

    def test_zero_ask(self):
        ratio, _, cost = _convexity_ratio(100, 110, 0, 0.20)
        assert ratio == 0.0
        assert cost == 0.0

    def test_deep_itm(self):
        ratio, payoff, cost = _convexity_ratio(200, 110, 1, 0.20)
        # tail_price=88, payoff=(200-88)*100=11200
        assert payoff == 11200.0
        assert ratio > 0


# ===========================================================================
# Convexity backtest helpers
# ===========================================================================


class TestMonthlyRebalanceDates:
    def test_basic(self):
        dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"])
        indices = _monthly_rebalance_dates(dates)
        assert indices == [0, 2]

    def test_single_month(self):
        dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
        indices = _monthly_rebalance_dates(dates)
        assert indices == [0]

    def test_empty(self):
        assert _monthly_rebalance_dates(pd.DatetimeIndex([])) == []


class TestStockPriceOn:
    def test_exact_match(self):
        dates_ns = np.array([100, 200, 300], dtype=np.int64)
        prices = np.array([10.0, 20.0, 30.0])
        assert _stock_price_on(dates_ns, prices, 200) == 20.0

    def test_between_dates(self):
        dates_ns = np.array([100, 300], dtype=np.int64)
        prices = np.array([10.0, 30.0])
        # 200 is between 100 and 300, should return price at 100
        assert _stock_price_on(dates_ns, prices, 200) == 10.0

    def test_before_first_date(self):
        dates_ns = np.array([100, 200], dtype=np.int64)
        prices = np.array([10.0, 20.0])
        assert _stock_price_on(dates_ns, prices, 50) is None


class TestFindDateRange:
    def test_exact_match(self):
        dates_ns = np.array([100, 100, 100, 200, 200, 300], dtype=np.int64)
        start, end = _find_date_range(dates_ns, 100)
        assert start == 0
        assert end == 3

    def test_no_match(self):
        dates_ns = np.array([100, 200, 300], dtype=np.int64)
        start, end = _find_date_range(dates_ns, 150)
        assert start == end


# ===========================================================================
# Convexity config
# ===========================================================================


class TestConvexityConfig:
    def test_instrument_config_defaults(self):
        ic = InstrumentConfig(symbol="SPY", options_file="x", stocks_file="y")
        assert ic.target_delta == -0.10
        assert ic.dte_min == 14
        assert ic.dte_max == 60
        assert ic.tail_drop == 0.20

    def test_backtest_config_defaults(self):
        bc = BacktestConfig()
        assert bc.initial_capital == 1_000_000.0
        assert bc.budget_pct == 0.005

    def test_default_config(self):
        cfg = default_config()
        assert len(cfg.instruments) == 1
        assert cfg.instruments[0].symbol == "SPY"

    def test_frozen(self):
        ic = InstrumentConfig(symbol="SPY", options_file="x", stocks_file="y")
        with pytest.raises(AttributeError):
            ic.symbol = "QQQ"


# ===========================================================================
# Convexity allocator
# ===========================================================================


class TestAllocator:
    def test_pick_cheapest(self):
        scores = {"SPY": 1.5, "QQQ": 2.0, "IWM": 0.8}
        assert pick_cheapest(scores) == "QQQ"

    def test_pick_cheapest_empty_raises(self):
        with pytest.raises(ValueError):
            pick_cheapest({})

    def test_equal_weight(self):
        alloc = allocate_equal_weight(["SPY", "QQQ"], 10_000)
        assert abs(alloc["SPY"] - 5_000) < 0.01
        assert abs(alloc["QQQ"] - 5_000) < 0.01

    def test_equal_weight_empty(self):
        assert allocate_equal_weight([], 10_000) == {}

    def test_inverse_vol(self):
        alloc = allocate_inverse_vol({"SPY": 0.15, "QQQ": 0.30}, 10_000)
        # SPY has lower vol → gets more budget
        assert alloc["SPY"] > alloc["QQQ"]
        assert abs(sum(alloc.values()) - 10_000) < 0.01

    def test_inverse_vol_zero_vol(self):
        """Zero vol should fall back to equal weight."""
        alloc = allocate_inverse_vol({"SPY": 0.0, "QQQ": 0.0}, 10_000)
        assert abs(alloc["SPY"] - 5_000) < 0.01

    def test_inverse_vol_mixed_zero(self):
        alloc = allocate_inverse_vol({"SPY": 0.15, "QQQ": 0.0}, 10_000)
        # QQQ has zero vol → excluded from inv-vol, only SPY gets budget
        assert alloc["SPY"] == 10_000


# ===========================================================================
# Dispatch layer
# ===========================================================================


class TestRustExtension:
    def test_rust_extension_importable(self):
        assert _ob_rust is not None


# ===========================================================================
# Schema / Filter DSL
# ===========================================================================


class TestSchemaBasic:
    def test_stocks_schema(self):
        s = Schema.stocks()
        assert s["symbol"] == "symbol"
        assert s["date"] == "date"

    def test_options_schema(self):
        s = Schema.options()
        assert s["type"] == "type"
        assert s["strike"] == "strike"

    def test_update(self):
        s = Schema.stocks()
        s.update({"custom": "custom_col"})
        assert s["custom"] == "custom_col"

    def test_contains(self):
        s = Schema.stocks()
        assert "symbol" in s
        assert "nonexistent" not in s

    def test_equality(self):
        s1 = Schema.stocks()
        s2 = Schema.stocks()
        assert s1 == s2

    def test_inequality_different_mappings(self):
        s1 = Schema.stocks()
        s2 = Schema.stocks()
        s2.update({"symbol": "ticker"})
        assert s1 != s2


class TestFilterDSL:
    def test_field_comparison(self):
        s = Schema.options()
        f = s.strike > 100
        assert isinstance(f, Filter)
        assert "strike > 100" in f.query

    def test_field_equality_string(self):
        s = Schema.options()
        f = s.type == "put"
        assert "'put'" in f.query

    def test_filter_and(self):
        s = Schema.options()
        f = (s.strike > 100) & (s.type == "put")
        assert "&" in f.query

    def test_filter_or(self):
        s = Schema.options()
        f = (s.strike > 100) | (s.strike < 50)
        assert "|" in f.query

    def test_filter_invert(self):
        s = Schema.options()
        f = ~(s.strike > 100)
        assert "!" in f.query

    def test_filter_call_on_dataframe(self):
        s = Schema.options()
        f = s.strike > 100
        df = pd.DataFrame({"strike": [50, 100, 150, 200]})
        result = f(df)
        assert result.sum() == 2

    def test_field_arithmetic(self):
        s = Schema.options()
        f = s.strike * 1.1
        assert isinstance(f, Field)
        assert "1.1" in f.mapping

    def test_field_subtraction(self):
        s = Schema.options()
        f = s.strike - 10
        assert "- 10" in f.mapping

    def test_field_comparison_between_fields(self):
        s = Schema.options()
        f = s.strike > s.underlying_last
        assert "strike" in f.query and "underlying_last" in f.query
