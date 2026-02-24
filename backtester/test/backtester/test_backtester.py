import numpy as np
import pandas as pd

from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Type, Direction
from backtester import Backtest


def test_backtest(sample_stock_portfolio, sample_stocks_datahandler, sample_options_datahandler):
    bt = run_backtest(sample_stocks_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy(Direction.BUY, sample_options_datahandler.schema),
                      stocks=sample_stock_portfolio)
    tl_long, balance_long = bt.trade_log, bt.balance

    symbols = [stock.symbol for stock in sample_stock_portfolio]
    balance_columns = ['total capital', 'cash'] + symbols + [
        'options qty', 'calls capital', 'puts capital', 'stocks qty', 'options capital', 'stocks capital', '% change',
        'accumulated return'
    ]

    last_day_balance_long = balance_long.iloc[-1]
    last_day_balance_long = last_day_balance_long[balance_columns].values

    leg_1_costs = tl_long['leg_1']['cost']
    leg_2_costs = tl_long['leg_2']['cost']
    total_costs = tl_long['totals']['cost']
    dates = tl_long['totals']['date'].dt.strftime("%Y-%m-%d")

    # We test with np.isclose instead of true equality because of possible floating point inaccuracies.
    tol = 0.000001

    assert np.allclose(last_day_balance_long, [
        1.001336e+06, 507298.034603, 201228.973492, 49853.700558, 242954.82254999998, 0.0, 0.0, 0.0, 16415.0, 0.0,
        494037.49659999995, -0.004165, 1.001336
    ],
                       atol=tol)

    assert np.allclose(leg_1_costs, [195010.0, -197060.0, 189250.0, -185650.0], atol=tol)
    assert np.allclose(leg_2_costs, [5.0, 0.0, 40.0, 0.0], atol=tol)
    assert np.allclose(total_costs, [195015.0, -197060.0, 189290.0, -185650.0], atol=tol)
    assert (dates == ['2017-01-03', '2017-02-01', '2017-03-01', '2017-04-03']).all()

    bt = run_backtest(sample_stocks_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy(Direction.SELL, sample_options_datahandler.schema),
                      stocks=sample_stock_portfolio)
    tl_short, balance_short = bt.trade_log, bt.balance

    last_day_balance_short = balance_short.iloc[-1]
    last_day_balance_short = last_day_balance_short[balance_columns].values

    leg_1_costs = tl_short['leg_1']['cost']
    leg_2_costs = tl_short['leg_2']['cost']
    total_costs = tl_short['totals']['cost']
    dates = tl_short['totals']['date'].dt.strftime("%Y-%m-%d")

    assert np.allclose(last_day_balance_short, [
        1010305.1222639999, 511978.532206, 202876.70306, 50312.368716, 245137.518282, 0.0, 0.0, 0.0, 16562.0, 0.0,
        498326.590058, -0.004166, 1.010305
    ],
                       atol=tol)

    assert np.allclose(leg_1_costs, [-188980.0, 186060.0], atol=tol)
    assert np.allclose(leg_2_costs, [-5.0, 10.0], atol=tol)
    assert np.allclose(total_costs, [-188985.0, 186070.0], atol=tol)
    assert (dates == ['2017-03-01', '2017-04-03']).all()


# We use Portfolio Visualizer (https://www.portfoliovisualizer.com/backtest-portfolio)
# to find the actual return for the Ivy porfolio.


def test_only_stocks(ivy_portfolio, ivy_portfolio_datahandler, sample_options_datahandler):
    allocation = {'stocks': 1.0, 'options': 0.0, 'cash': 0.0}
    bt = run_backtest(ivy_portfolio_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy(Direction.BUY, sample_options_datahandler.schema),
                      stocks=ivy_portfolio,
                      allocation=allocation)

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['total capital'], balance['cash'] + balance['stocks capital'], rtol=tolerance)
    assert np.allclose(balance['total capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)
    assert np.allclose(balance['options capital'], 0, rtol=tolerance)

    actual_return = 1.025
    return_tolerance = 0.01
    assert np.isclose(balance['accumulated return'].iloc[-1], actual_return, rtol=return_tolerance)


def test_only_cash(sample_stock_portfolio, sample_stocks_datahandler, sample_options_datahandler):
    allocation = {'stocks': 0.0, 'options': 0.0, 'cash': 1.0}
    bt = run_backtest(sample_stocks_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy(Direction.BUY, sample_options_datahandler.schema),
                      stocks=sample_stock_portfolio,
                      allocation=allocation)

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['total capital'], balance['cash'], rtol=tolerance)
    assert np.allclose(balance['total capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)
    assert np.allclose(balance['stocks capital'], 0, rtol=tolerance)
    assert np.allclose(balance['options capital'], 0, rtol=tolerance)
    assert np.allclose(balance['% change'], 0, rtol=tolerance)
    assert np.allclose(balance['accumulated return'], 1.0, rtol=tolerance)


def test_only_options(sample_stock_portfolio, sample_stocks_datahandler, sample_options_datahandler):
    allocation = {'stocks': 0.0, 'options': 1.0, 'cash': 0.0}
    bt = run_backtest(sample_stocks_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy(Direction.BUY, sample_options_datahandler.schema),
                      stocks=sample_stock_portfolio,
                      allocation=allocation)

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['total capital'], balance['cash'] + balance['options capital'], rtol=tolerance)
    assert np.allclose(balance['total capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)
    assert np.allclose(balance['stocks capital'], 0, rtol=tolerance)

    actual_return = 0.992025
    return_tolerance = 0.01
    assert np.allclose(balance['accumulated return'].iloc[-1], actual_return, rtol=return_tolerance)


def test_current_stock_capital(sample_stocks_datahandler, sample_stock_portfolio):
    bt = Backtest({'stocks': 1, 'options': 0, 'cash': 0})
    bt.stocks_data = sample_stocks_datahandler
    bt.stocks = sample_stock_portfolio
    stock_symbols = [stock.symbol for stock in sample_stock_portfolio]
    bt._stocks_inventory = pd.DataFrame({'symbol': stock_symbols, 'price': [0, 0, 0], 'qty': [1, 5, 10]})
    stocks = sample_stocks_datahandler._data[sample_stocks_datahandler._data['date'] == pd.Timestamp('2017-05-15')]

    tolerance = 0.0000001
    capital = bt._current_stock_capital(stocks)
    actual_capital = 572.107697
    assert np.isclose(capital, actual_capital, atol=tolerance)


def test_buy_stocks(sample_stocks_datahandler, sample_stock_portfolio):
    bt = Backtest({'stocks': 1, 'options': 0, 'cash': 0})
    bt.stocks_data = sample_stocks_datahandler
    bt.stocks = sample_stock_portfolio
    sma_days = 30
    bt.stocks_data.sma(sma_days)
    stocks = bt.stocks_data._data[bt.stocks_data._data['date'] == pd.Timestamp('2017-03-01')]
    allocation = 100_000

    tolerance = 0.0000001

    bt.current_cash = allocation
    bt._buy_stocks(stocks, allocation, 0)
    inventory = bt._stocks_inventory
    assert np.isclose(bt.current_cash, 157.41366399999242, atol=tolerance)
    assert (inventory['symbol'].values == ['VOO', 'TUR', 'RSX']).all()
    assert np.allclose(inventory['price'].values, [207.646172, 32.502688, 17.754331], atol=tolerance)
    assert (inventory['qty'].values == [192., 307., 2816.]).all()

    bt.current_cash = allocation
    bt._buy_stocks(stocks, allocation, sma_days)
    inventory = bt._stocks_inventory
    assert np.isclose(bt.current_cash, 50153.60976, atol=tolerance)
    assert (inventory['symbol'].values == ['VOO', 'TUR', 'RSX']).all()
    assert np.allclose(inventory['price'].values, [207.646172, 32.502688, 17.754331], atol=tolerance)
    assert (inventory['qty'].values == [192., 307., 0.]).all()


def test_initialize_inventories_columns(sample_stock_portfolio, sample_stocks_datahandler, sample_options_datahandler):
    """Verify _initialize_inventories creates DataFrame with correct MultiIndex columns."""
    bt = Backtest({'stocks': 0.5, 'options': 0.5, 'cash': 0})
    bt.stocks_data = sample_stocks_datahandler
    bt.stocks = sample_stock_portfolio
    bt.options_strategy = sample_options_strategy(Direction.BUY, sample_options_datahandler.schema)
    bt._initialize_inventories()

    cols = bt._options_inventory.columns
    assert isinstance(cols, pd.MultiIndex)
    # Should contain leg columns plus totals
    assert ('totals', 'cost') in cols
    assert ('totals', 'qty') in cols
    assert ('totals', 'date') in cols
    assert ('leg_1', 'contract') in cols
    assert ('leg_2', 'contract') in cols


def run_backtest(stock_data,
                 options_data,
                 strategy,
                 stocks=None,
                 allocation={
                     'stocks': 0.50,
                     'options': 0.50,
                     'cash': 0
                 },
                 **kwargs):
    bt = Backtest(allocation, **kwargs)
    if stocks:
        bt.stocks = stocks
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.stocks_data = stock_data

    bt.run(rebalance_freq=1)
    return bt


def test_entry_sort(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify entry_sort=('strike', True) picks the contract with the lowest strike."""
    options_data = options_data_2puts_buy
    schema = options_data.schema

    # Strategy with entry_sort picking lowest strike
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = (schema.dte <= 30)
    leg.entry_sort = ('strike', True)  # ascending → lowest strike first
    strat.add_legs([leg])

    bt = Backtest({'stocks': 0.97, 'options': 0.03, 'cash': 0})
    bt.stocks = ivy_5assets_portfolio
    bt.options_strategy = strat
    bt.options_data = options_data
    bt.stocks_data = ivy_portfolio_5assets_datahandler
    bt.run(rebalance_freq=1)

    # First entry should have the lowest strike (650)
    assert bt.trade_log['leg_1']['strike'].iloc[0] == 650.0

    # Now test with descending sort → highest strike first (700)
    strat2 = Strategy(schema)
    leg2 = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg2.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg2.exit_filter = (schema.dte <= 30)
    leg2.entry_sort = ('strike', False)  # descending → highest strike first
    strat2.add_legs([leg2])

    bt2 = Backtest({'stocks': 0.97, 'options': 0.03, 'cash': 0})
    bt2.stocks = ivy_5assets_portfolio
    bt2.options_strategy = strat2
    bt2.options_data = options_data
    bt2.stocks_data = ivy_portfolio_5assets_datahandler
    bt2.run(rebalance_freq=1)

    assert bt2.trade_log['leg_1']['strike'].iloc[0] == 700.0


def test_options_budget_fixed(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify options_budget as a fixed float overrides percentage allocation."""
    options_data = options_data_2puts_buy
    schema = options_data.schema

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = (schema.dte <= 30)
    strat.add_legs([leg])

    budget = 5000.0
    bt = Backtest({'stocks': 0.97, 'options': 0.03, 'cash': 0})
    bt.options_budget = budget
    bt.stocks = ivy_5assets_portfolio
    bt.options_strategy = strat
    bt.options_data = options_data
    bt.stocks_data = ivy_portfolio_5assets_datahandler
    bt.run(rebalance_freq=1)

    # First entry: qty should be budget // cost_per_contract
    first_cost = bt.trade_log['totals']['cost'].iloc[0]
    first_qty = bt.trade_log['totals']['qty'].iloc[0]
    assert first_qty == budget // first_cost


def test_options_budget_callable(options_data_2puts_buy, ivy_portfolio_5assets_datahandler, ivy_5assets_portfolio):
    """Verify options_budget as a callable is invoked with (date, total_capital)."""
    options_data = options_data_2puts_buy
    schema = options_data.schema

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = (schema.dte <= 30)
    strat.add_legs([leg])

    calls = []

    def budget_fn(date, total_capital):
        calls.append((date, total_capital))
        return 5000.0

    bt = Backtest({'stocks': 0.97, 'options': 0.03, 'cash': 0})
    bt.options_budget = budget_fn
    bt.stocks = ivy_5assets_portfolio
    bt.options_strategy = strat
    bt.options_data = options_data
    bt.stocks_data = ivy_portfolio_5assets_datahandler
    bt.run(rebalance_freq=1)

    # The callable should have been invoked at each rebalance
    assert len(calls) > 0
    # Each call should have received (pd.Timestamp, number)
    for date, capital in calls:
        assert isinstance(date, pd.Timestamp)
        assert isinstance(capital, (int, float, np.floating))


def test_run_metadata_attached(sample_stock_portfolio, sample_stocks_datahandler, sample_options_datahandler):
    bt = run_backtest(sample_stocks_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy(Direction.BUY, sample_options_datahandler.schema),
                      stocks=sample_stock_portfolio)

    meta = bt.run_metadata
    assert meta['framework'] == 'backtester.Backtest'
    assert meta['dispatch_mode'] == 'python-legacy'
    assert isinstance(meta['git_sha'], str)
    assert len(meta['config_hash']) == 64
    assert len(meta['data_snapshot_hash']) == 64
    assert meta['data_snapshot']['options_rows'] > 0
    assert meta['data_snapshot']['stocks_rows'] > 0
    assert bt.trade_log.attrs['run_metadata'] == meta
    assert bt.balance.attrs['run_metadata'] == meta


def sample_options_strategy(direction, schema):
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
