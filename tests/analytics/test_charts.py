"""Tests for weights_chart."""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for testing

from options_portfolio_backtester.analytics.charts import weights_chart


def _make_balance() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame({
        "total capital": np.linspace(10000, 11000, 10),
        "cash": np.linspace(2000, 1000, 10),
        "stocks capital": np.linspace(8000, 10000, 10),
        "SPY qty": [50.0] * 10,
        "TLT qty": [100.0] * 10,
    }, index=dates)


def test_weights_chart_returns_fig_ax():
    balance = _make_balance()
    fig, ax = weights_chart(balance)
    assert fig is not None
    assert ax is not None
    assert ax.get_ylabel() == "Weight"


def test_weights_chart_no_positions():
    dates = pd.bdate_range("2024-01-02", periods=5)
    balance = pd.DataFrame({
        "total capital": [10000.0] * 5,
        "cash": [10000.0] * 5,
    }, index=dates)
    fig, ax = weights_chart(balance)
    assert fig is not None
    assert "no positions" in ax.get_title().lower()


def test_weights_chart_single_symbol():
    dates = pd.bdate_range("2024-01-02", periods=5)
    balance = pd.DataFrame({
        "total capital": [10000.0] * 5,
        "cash": [2000.0] * 5,
        "SPY qty": [80.0] * 5,
    }, index=dates)
    fig, ax = weights_chart(balance)
    assert fig is not None
