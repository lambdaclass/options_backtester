import numpy as np
import math
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Type, Direction
from backtester import Backtest


def test_sell_some_options_1legs(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):

    options_data = options_data_2puts_buy
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler, options_data,
                      options_1leg_buy_strategy(options_data))
    tolerance = 0.0001
    assert np.isclose(bt.balance.loc['2014-12-15']['options qty'], 300, rtol=tolerance)
    assert np.allclose(bt.balance.loc['2014-12-15'][['options qty', 'total capital', 'puts capital']].values,
                       [300, 985000, 15000],
                       rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['cost'].values, [100, 150], rtol=tolerance)
    assert np.allclose(bt.trade_log['leg_1']['cost'].values, [100, 150], rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['qty'].values, [300, (((97 + 3 * 0.5) * 0.03 - 1.5) / 1.5) * 100],
                       rtol=tolerance)


def test_sell_some_options_2legs_buy(options_data_2legs_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    options_data = options_data_2legs_buy

    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler, options_data,
                      options_2legs_buy_strategy(options_data))
    tolerance = 0.0001

    assert np.allclose(bt.balance.loc['2014-12-15'][[
        'options qty', 'total capital', 'puts capital', 'calls capital', 'options capital'
    ]].values, [150, 985000, 7500, 7500, 15000],
                       rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['qty'].values, [150, (((97 + 3 * 0.5) * 0.03 - 1.5) / 1.5) * 100 // 2],
                       rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['cost'].values, [200, 300], rtol=tolerance)


def test_sell_some_options_1leg_buy_sell(options_data_1put_buy_sell, ivy_portfolio_5assets_datahandler,
                                         ivy_5assets_portfolio):
    options_data = options_data_1put_buy_sell
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler, options_data,
                      options_1leg_buy_strategy(options_data))
    tolerance = 0.0001
    assert np.allclose(bt.trade_log['totals']['qty'].values,
                       [300, math.ceil((60000 - (970000 + 30000 * 2) * 0.03) / 200)],
                       rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['cost'].values, [100, -200], rtol=tolerance)


def test_sell_some_options_2leg_buy_sell(options_data_buy_and_sell_2legs, ivy_portfolio_5assets_datahandler,
                                         ivy_5assets_portfolio):
    options_data = options_data_buy_and_sell_2legs
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler, options_data,
                      options_2legs_buy_strategy(options_data))
    tolerance = 0.0001
    assert np.allclose(bt.trade_log['totals']['qty'].values, [
        150, ((970000 + 150 * (50 * 2)) * 0.03 - 150 * (50 * 2)) // 300,
        ((150 * 200 + 48 * 300) - (955450 + 150 * 200 + 48 * 300) * 0.03) // 200
    ],
                       rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['cost'].values, [200, 300, -200], rtol=tolerance)


def test_sell_some_options_1leg_buy_sell_all(options_data_1put_buy_sell_all, ivy_portfolio_5assets_datahandler,
                                             ivy_5assets_portfolio):
    options_data = options_data_1put_buy_sell_all
    bt = run_backtest(ivy_5assets_portfolio, ivy_portfolio_5assets_datahandler, options_data,
                      options_1leg_buy_strategy(options_data))
    tolerance = 0.0001
    assert np.allclose(bt.trade_log['totals']['qty'].values, [
        300, ((970000 + 300 * 25) * 0.03 - 300 * 25) // 100, 300,
        math.ceil((1000 * 218 - (((970000 + 300 * 25) * 0.97 + 300 * 150 + 218 * 1000) * 0.03)) / 1000)
    ],
                       rtol=tolerance)
    assert np.allclose(bt.trade_log['totals']['cost'].values, [100, 100, -150, -1000], rtol=tolerance)


def run_backtest(stocks,
                 stock_data,
                 options_data,
                 strategy,
                 allocation={
                     'stocks': 0.97,
                     'options': 0.03,
                     'cash': 0
                 },
                 **kwargs):
    bt = Backtest(allocation, **kwargs)
    bt.stocks = stocks
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.stocks_data = stock_data

    bt.run(rebalance_freq=1)
    return bt


def options_1leg_buy_strategy(options_data):

    options_schema = options_data.schema
    test_strat = Strategy(options_schema)
    leg1 = StrategyLeg("leg_1", options_schema, option_type=Type.PUT, direction=Direction.BUY)
    leg1.entry_filter = ((options_schema.underlying == "SPX") & (options_schema.dte >= 60))
    leg1.exit_filter = (options_schema.dte <= 30)
    test_strat.add_legs([leg1])

    return test_strat


def options_2legs_buy_strategy(options_data):

    options_schema = options_data.schema

    test_strat = Strategy(options_schema)
    leg_1 = StrategyLeg("leg_1", options_schema, option_type=Type.PUT, direction=Direction.BUY)
    leg_1.entry_filter = (options_schema.underlying == "SPX") & (options_schema.dte >= 60)
    leg_1.exit_filter = (options_schema.dte <= 30)

    leg_2 = StrategyLeg("leg_2", options_schema, option_type=Type.CALL, direction=Direction.BUY)
    leg_2.entry_filter = (options_schema.underlying == "SPX") & (options_schema.dte >= 60)
    leg_2.exit_filter = (options_schema.dte <= 30)
    test_strat.add_legs([leg_1, leg_2])
    return test_strat
