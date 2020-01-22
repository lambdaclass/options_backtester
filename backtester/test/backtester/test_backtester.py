import numpy as np

from backtester.datahandler import HistoricalOptionsData
from backtester.strategy import Strategy, StrategyLeg
from backtester.option import Type, Direction
from backtester import Backtest


def test_backtest():
    tl_short = run_backtest(Direction.SELL)

    leg_1_costs = tl_short['leg_1']['cost']
    leg_2_costs = tl_short['leg_2']['cost']
    total_costs = tl_short['totals']['cost']
    dates = tl_short['totals']['date'].dt.strftime("%Y-%m-%d")

    # We test with np.isclose instead of true equality because of possible floating point inaccuracies.
    tol = 0.000001

    assert ((np.isclose(leg_1_costs, [-184140.0, -137170.0, 183980.0, 136510.0], atol=tol)).all())
    assert ((np.isclose(leg_2_costs, [-5.0, -30.0, 10.0, 5.0], atol=tol)).all())
    assert ((np.isclose(total_costs, [-184145.0, -137200.0, 183990.0, 136515.0], atol=tol)).all())
    assert ((dates == ['2017-02-17', '2017-03-17', '2017-04-19', '2017-05-17']).all())

    tl_long = run_backtest(Direction.BUY)

    leg_1_costs = tl_long['leg_1']['cost']
    leg_2_costs = tl_long['leg_2']['cost']
    total_costs = tl_long['totals']['cost']
    dates = tl_long['totals']['date'].dt.strftime("%Y-%m-%d")

    assert ((np.isclose(leg_1_costs, [196880.0, 184580.0, -208220.0, 137620.0, -183540.0, -135750.0], atol=tol)).all())
    assert ((np.isclose(leg_2_costs, [25.0, 50.0, -0.0, 85.0, -0.0, -0.0], atol=tol)).all())
    assert ((np.isclose(total_costs, [196905.0, 184630.0, -208220.0, 137705.0, -183540.0, -135750.0], atol=tol)).all())
    assert ((dates == ['2017-01-13', '2017-02-17', '2017-03-15', '2017-03-17', '2017-04-19', '2017-05-17']).all())


def run_backtest(direction):
    data = HistoricalOptionsData("backtester/test/backtester/test_data/test_data.csv")
    schema = data.schema

    test_strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=direction)

    leg1.entry_filter = (schema.dte == 63)
    leg1.exit_filter = (schema.dte <= 2)

    leg2 = StrategyLeg("leg_2", schema, option_type=Type.PUT, direction=direction)

    leg2.entry_filter = (schema.dte == 63)
    leg2.exit_filter = (schema.dte <= 2)

    test_strat.add_legs([leg1, leg2])

    bt = Backtest()
    bt.strategy = test_strat
    bt.data = data
    bt.stop_if_broke = False

    bt.run(monthly=False)
    return bt.trade_log
