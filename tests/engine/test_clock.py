"""Tests for TradingClock — date iteration and rebalance scheduling."""

import pandas as pd
import numpy as np

from options_portfolio_backtester.engine.clock import TradingClock


def _make_data(n_dates=5):
    """Create minimal stocks + options DataFrames for clock tests."""
    dates = pd.bdate_range("2020-01-06", periods=n_dates, freq="B")
    stocks = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["SPY", "IWM"] * n_dates,
        "adjClose": np.random.uniform(300, 400, n_dates * 2),
    })
    options = pd.DataFrame({
        "quotedate": np.repeat(dates, 3),
        "optionroot": [f"SPY_C_{i}" for i in range(n_dates * 3)],
        "volume": np.random.randint(100, 10000, n_dates * 3),
    })
    return stocks, options, dates


class TestDailyIteration:
    def test_yields_correct_number_of_dates(self):
        stocks, options, dates = _make_data(5)
        clock = TradingClock(stocks, options)
        result = list(clock.iter_dates())
        assert len(result) == 5

    def test_yields_tuples_of_date_stocks_options(self):
        stocks, options, dates = _make_data(3)
        clock = TradingClock(stocks, options)
        for date, s, o in clock.iter_dates():
            assert isinstance(date, pd.Timestamp)
            assert isinstance(s, pd.DataFrame)
            assert isinstance(o, pd.DataFrame)


class TestAllDates:
    def test_returns_all_unique_dates(self):
        stocks, options, dates = _make_data(5)
        clock = TradingClock(stocks, options)
        assert len(clock.all_dates) == 5


class TestRebalanceDates:
    def test_zero_freq_returns_empty(self):
        stocks, options, dates = _make_data(5)
        clock = TradingClock(stocks, options)
        rb = clock.rebalance_dates(0)
        assert len(rb) == 0

    def test_negative_freq_returns_empty(self):
        stocks, options, dates = _make_data(5)
        clock = TradingClock(stocks, options)
        rb = clock.rebalance_dates(-1)
        assert len(rb) == 0

    def test_positive_freq_returns_dates(self):
        # Use enough dates to span multiple months
        dates = pd.bdate_range("2020-01-06", periods=60, freq="B")
        stocks = pd.DataFrame({
            "date": np.repeat(dates, 1),
            "symbol": ["SPY"] * 60,
            "adjClose": np.random.uniform(300, 400, 60),
        })
        options = pd.DataFrame({
            "quotedate": np.repeat(dates, 1),
            "optionroot": [f"SPY_C_{i}" for i in range(60)],
            "volume": np.random.randint(100, 10000, 60),
        })
        clock = TradingClock(stocks, options)
        rb = clock.rebalance_dates(1)
        assert len(rb) > 0
        assert isinstance(rb, pd.DatetimeIndex)


class TestMonthlyIteration:
    def test_monthly_mode_yields_first_of_month_dates(self):
        # Span 3 months of business days
        dates = pd.bdate_range("2020-01-06", periods=60, freq="B")
        stocks = pd.DataFrame({
            "date": np.repeat(dates, 1),
            "symbol": ["SPY"] * 60,
            "adjClose": np.random.uniform(300, 400, 60),
        })
        options = pd.DataFrame({
            "quotedate": np.repeat(dates, 1),
            "optionroot": [f"SPY_C_{i}" for i in range(60)],
            "volume": np.random.randint(100, 10000, 60),
        })
        clock = TradingClock(stocks, options, monthly=True)
        result = list(clock.iter_dates())
        # monthly=True should yield fewer dates than daily
        assert len(result) <= 60
        assert len(result) >= 1
        for date, s, o in result:
            assert isinstance(date, pd.Timestamp)
