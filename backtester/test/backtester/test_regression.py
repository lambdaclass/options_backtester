"""Regression tests that capture exact numerical output of backtests.

These serve as the correctness oracle for performance optimizations.
If any optimization changes a numerical result, these tests will catch it.
"""
import numpy as np

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


def sell_strategy(options_data):
    schema = options_data.schema
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.SELL)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = (schema.dte <= 30)
    strat.add_legs([leg])
    return strat


def test_buy_regression(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Captures exact trade_log and balance output for a BUY backtest."""
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_2puts_buy, buy_strategy(options_data_2puts_buy))

    tol = 0.0001

    # Trade log costs
    assert np.allclose(bt.trade_log['totals']['cost'].values, [100, 150], rtol=tol)
    assert np.allclose(bt.trade_log['leg_1']['cost'].values, [100, 150], rtol=tol)

    # Trade log quantities
    assert np.allclose(bt.trade_log['totals']['qty'].values,
                       [300, (((97 + 3 * 0.5) * 0.03 - 1.5) / 1.5) * 100], rtol=tol)

    # Balance snapshot
    assert np.isclose(bt.balance.loc['2014-12-15']['options qty'], 300, rtol=tol)
    assert np.allclose(
        bt.balance.loc['2014-12-15'][['options qty', 'total capital', 'puts capital']].values,
        [300, 985000, 15000], rtol=tol)


def test_sell_regression(options_data_1put_buy_sell, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Captures exact trade_log output for a BUY/SELL backtest."""
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_1put_buy_sell, buy_strategy(options_data_1put_buy_sell))

    import math
    tol = 0.0001
    assert np.allclose(bt.trade_log['totals']['qty'].values,
                       [300, math.ceil((60000 - (970000 + 30000 * 2) * 0.03) / 200)], rtol=tol)
    assert np.allclose(bt.trade_log['totals']['cost'].values, [100, -200], rtol=tol)
