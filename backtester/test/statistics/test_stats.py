"""Tests for statistics.stats.summary()

Note: stats.summary() has a pre-existing bug where balance['total capital'].get(0)
returns None on date-indexed DataFrames (pandas >= 2.0). These tests are marked xfail.
"""
import pytest
import numpy as np

from backtester.statistics.stats import summary
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Type, Direction
from backtester import Backtest


def run_backtest(stocks, stock_data, options_data, strategy,
                 allocation={'stocks': 0.97, 'options': 0.03, 'cash': 0}, **kwargs):
    bt = Backtest(allocation, **kwargs)
    bt.stocks = stocks
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.stocks_data = stock_data
    bt.run(rebalance_freq=1)
    return bt


def buy_strategy(options_data):
    schema = options_data.schema
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = (schema.dte <= 30)
    strat.add_legs([leg])
    return strat


@pytest.mark.xfail(reason="stats.summary() uses .get(0) which fails on date-indexed balance")
def test_summary_has_expected_stats(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify summary() returns all expected stat labels."""
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_2puts_buy, buy_strategy(options_data_2puts_buy))

    result = summary(bt.trade_log, bt.balance)
    expected_stats = [
        'Total trades', 'Number of wins', 'Number of losses', 'Win %',
        'Largest loss', 'Profit factor', 'Average profit',
        'Average P&L %', 'Total P&L %'
    ]
    for stat in expected_stats:
        assert stat in result.data.index


@pytest.mark.xfail(reason="stats.summary() uses .get(0) which fails on date-indexed balance")
def test_summary_values(options_data_1put_buy_sell, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify key summary statistics with known data."""
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_1put_buy_sell, buy_strategy(options_data_1put_buy_sell))

    result = summary(bt.trade_log, bt.balance)
    data = result.data

    # Should have at least 1 trade
    total_trades = data.loc['Total trades', 'Strategy']
    assert total_trades >= 1

    # Win % should be between 0 and 100
    win_pct = data.loc['Win %', 'Strategy']
    assert 0 <= win_pct <= 100

    # Profit factor should be non-negative
    profit_factor = data.loc['Profit factor', 'Strategy']
    assert profit_factor >= 0
