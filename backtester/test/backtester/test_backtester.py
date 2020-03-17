import numpy as np

from backtester.enums import Stock
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Type, Direction
from backtester import Backtest


def test_backtest(sample_datahandler_stocks, sample_datahandler_options):
    tl_long, balance_long = run_backtest(sample_datahandler_stocks, sample_datahandler_options, Direction.BUY)

    last_day_balance_long = balance_long.iloc[-1].values

    leg_1_costs = tl_long['leg_1']['cost']
    leg_2_costs = tl_long['leg_2']['cost']
    total_costs = tl_long['totals']['cost']
    dates = tl_long['totals']['date'].dt.strftime("%Y-%m-%d")

    # We test with np.isclose instead of true equality because of possible floating point inaccuracies.
    tol = 0.000001

    assert (np.isclose(last_day_balance_long, [
        1.001336e+06, 507298.034603, 201228.973492, 49853.700558, 242954.82254999998, 0.0, 0.0, 0.0, 16415.0, 0.0,
        494037.49659999995, -0.004165, 1.001336
    ],
                       atol=tol)).all()

    assert (np.isclose(leg_1_costs, [195010.0, -197060.0, 189250.0, -185650.0], atol=tol)).all()
    assert (np.isclose(leg_2_costs, [5.0, 0.0, 40.0, 0.0], atol=tol)).all()
    assert (np.isclose(total_costs, [195015.0, -197060.0, 189290.0, -185650.0], atol=tol)).all()
    assert (dates == ['2017-01-03', '2017-02-01', '2017-03-01', '2017-04-03']).all()

    tl_short, balance_short = run_backtest(sample_datahandler_stocks, sample_datahandler_options, Direction.SELL)

    last_day_balance_short = balance_short.iloc[-1].values

    leg_1_costs = tl_short['leg_1']['cost']
    leg_2_costs = tl_short['leg_2']['cost']
    total_costs = tl_short['totals']['cost']
    dates = tl_short['totals']['date'].dt.strftime("%Y-%m-%d")

    assert (np.isclose(last_day_balance_short, [
        1010305.1222639999, 511978.532206, 202876.70306, 50312.368716, 245137.518282, 0.0, 0.0, 0.0, 16562.0, 0.0,
        498326.590058, -0.004166, 1.010305
    ],
                       atol=tol)).all()

    assert (np.isclose(leg_1_costs, [-188980.0, 186060.0], atol=tol)).all()
    assert (np.isclose(leg_2_costs, [-5.0, 10.0], atol=tol)).all()
    assert (np.isclose(total_costs, [-188985.0, 186070.0], atol=tol)).all()
    assert (dates == ['2017-03-01', '2017-04-03']).all()


def run_backtest(stock_data, options_data, direction):
    schema = options_data.schema
    test_strat = strategy(direction, schema)
    stocks = portfolio()

    bt = Backtest({'stocks': 0.50, 'options': 0.50, 'cash': 0})
    bt.stocks = stocks
    bt.options_strategy = test_strat
    bt.options_data = options_data
    bt.stocks_data = stock_data

    bt.run(rebalance_freq=1)
    return bt.trade_log, bt.balance


def strategy(direction, schema):
    test_strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=direction)
    leg1.entry_filter = ((schema.contract == "SPX170317C00300000") &
                         (schema.dte == 73)) | ((schema.contract == 'SPX170421C00500000') & (schema.dte == 51))

    leg1.exit_filter = (schema.dte == 44) | (schema.dte == 18)

    leg2 = StrategyLeg("leg_2", schema, option_type=Type.PUT, direction=direction)
    leg2.entry_filter = ((schema.contract == 'SPX170317P00300000') &
                         (schema.dte == 73)) | ((schema.contract == 'SPX170421P01375000') & (schema.dte == 51))

    leg2.exit_filter = (schema.dte == 44) | (schema.dte == 18)

    test_strat.add_legs([leg1, leg2])

    return test_strat


def portfolio():
    VOO = Stock('VOO', 0.4)
    TUR = Stock('TUR', 0.1)
    RSX = Stock('RSX', 0.5)

    stocks = [VOO, TUR, RSX]
    return stocks
