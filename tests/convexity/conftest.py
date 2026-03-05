"""Shared fixtures for convexity tests."""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.convexity.config import BacktestConfig, InstrumentConfig


class MockOptionsData:
    """Mock HistoricalOptionsData with ._data attribute."""

    def __init__(self, df: pd.DataFrame):
        self._data = df


class MockStocksData:
    """Mock TiingoData with ._data attribute."""

    def __init__(self, df: pd.DataFrame):
        self._data = df


def _make_put_row(date, strike, bid, ask, delta, underlying, dte, iv, expiration):
    return {
        "quotedate": pd.Timestamp(date),
        "expiration": pd.Timestamp(expiration),
        "type": "put",
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "delta": delta,
        "underlying_last": underlying,
        "dte": dte,
        "impliedvol": iv,
    }


@pytest.fixture
def instrument_config():
    return InstrumentConfig(
        symbol="TEST",
        options_file="test_options.csv",
        stocks_file="test_stocks.csv",
        target_delta=-0.10,
        dte_min=14,
        dte_max=60,
        tail_drop=0.20,
    )


@pytest.fixture
def backtest_config():
    return BacktestConfig(
        initial_capital=100_000.0,
        budget_pct=0.005,
    )


@pytest.fixture
def synthetic_options():
    """Three months of synthetic put options, 3 strikes per day."""
    rows = []
    dates = pd.bdate_range("2020-01-02", "2020-03-31")
    for date in dates:
        expiration = date + pd.Timedelta(days=30)
        underlying = 400.0
        for strike, bid, ask, delta, iv in [
            (360.0, 2.5, 3.0, -0.08, 0.20),
            (370.0, 3.5, 4.0, -0.12, 0.22),
            (380.0, 5.0, 5.5, -0.18, 0.25),
        ]:
            rows.append(_make_put_row(date, strike, bid, ask, delta, underlying, 30, iv, expiration))

    df = pd.DataFrame(rows)
    return MockOptionsData(df)


@pytest.fixture
def synthetic_stocks():
    """Three months of synthetic stock prices."""
    dates = pd.bdate_range("2020-01-02", "2020-03-31")
    np.random.seed(42)
    prices = 400.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, len(dates)))
    df = pd.DataFrame({"date": dates, "adjClose": prices})
    return MockStocksData(df)


@pytest.fixture
def empty_options():
    """Empty options DataFrame."""
    df = pd.DataFrame(columns=[
        "quotedate", "expiration", "type", "strike", "bid", "ask",
        "delta", "underlying_last", "dte", "impliedvol",
    ])
    return MockOptionsData(df)


@pytest.fixture
def empty_stocks():
    """Empty stocks DataFrame."""
    df = pd.DataFrame(columns=["date", "adjClose"])
    return MockStocksData(df)
