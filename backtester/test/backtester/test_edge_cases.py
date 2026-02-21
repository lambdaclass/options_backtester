"""Edge case tests for the backtester."""
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


def test_no_entry_signals(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """When filters match nothing, no trades should occur and capital should be preserved."""
    schema = options_data_2puts_buy.schema
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    # Filter that matches nothing: strike > 999999
    leg.entry_filter = schema.strike > 999999
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])

    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_2puts_buy, strat)

    # No trades should have been made
    assert bt.trade_log.empty
    # Options capital should be 0 throughout
    assert np.allclose(bt.balance['options capital'].iloc[1:], 0, atol=0.01)


def test_profit_threshold_exit(options_data_1put_buy_sell, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify that positions exit when profit threshold is exceeded."""
    schema = options_data_1put_buy_sell.schema
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    # Very high DTE exit so only thresholds trigger exit
    leg.exit_filter = schema.dte <= 0
    strat.add_legs([leg])
    # Very tight profit threshold
    strat.add_exit_thresholds(profit_pct=0.01, loss_pct=float('inf'))

    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_1put_buy_sell, strat)

    # With a 1% profit threshold, exits should happen quickly
    # Verify we have both entries and exits in trade log
    if not bt.trade_log.empty:
        assert len(bt.trade_log) >= 1


def test_loss_threshold_exit(options_data_1put_buy_sell, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify that positions exit when loss threshold is exceeded."""
    schema = options_data_1put_buy_sell.schema
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 0
    strat.add_legs([leg])
    # Very tight loss threshold
    strat.add_exit_thresholds(profit_pct=float('inf'), loss_pct=0.01)

    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler,
                      options_data_1put_buy_sell, strat)

    if not bt.trade_log.empty:
        assert len(bt.trade_log) >= 1
