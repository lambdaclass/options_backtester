"""Tests for convexity backtest module."""

import pandas as pd
import pytest

from options_portfolio_backtester.convexity.backtest import (
    BacktestResult,
    run_backtest,
    run_unhedged,
)


class TestRunBacktest:
    def test_returns_monthly_records(
        self, synthetic_options, synthetic_stocks, backtest_config,
    ):
        result = run_backtest(synthetic_options, synthetic_stocks, backtest_config)
        assert isinstance(result, BacktestResult)
        assert len(result.records) == 3

    def test_daily_balance_populated(
        self, synthetic_options, synthetic_stocks, backtest_config,
    ):
        result = run_backtest(synthetic_options, synthetic_stocks, backtest_config)
        assert not result.daily_balance.empty
        assert (result.daily_balance["balance"] > 0).all()

    def test_budget_deducted(
        self, synthetic_options, synthetic_stocks, backtest_config,
    ):
        result = run_backtest(synthetic_options, synthetic_stocks, backtest_config)
        assert result.records["put_cost"].iloc[0] > 0
        assert result.records["contracts"].iloc[0] > 0

    def test_empty_options(
        self, empty_options, synthetic_stocks, backtest_config,
    ):
        result = run_backtest(empty_options, synthetic_stocks, backtest_config)
        assert result.records.empty
        assert result.daily_balance.empty


class TestRunUnhedged:
    def test_returns_correct_shape(self, synthetic_stocks, backtest_config):
        daily = run_unhedged(synthetic_stocks, backtest_config)
        assert not daily.empty
        assert "balance" in daily.columns
        assert "pct_change" in daily.columns
        dates = pd.bdate_range("2020-01-02", "2020-03-31")
        assert len(daily) == len(dates)

    def test_initial_value_matches_capital(self, synthetic_stocks, backtest_config):
        daily = run_unhedged(synthetic_stocks, backtest_config)
        assert abs(daily["balance"].iloc[0] - backtest_config.initial_capital) < 0.01
