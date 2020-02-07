import numpy as np

from asset_backtester import Backtest, Portfolio, Asset

# We use Portfolio Visualizer (https://www.portfoliovisualizer.com/backtest-portfolio)
# to find the actual return for the test porfolios.


def test_ivy_portfolio(sample_datahandler):
    bt = run_backtest(sample_datahandler, ivy_portfolio())

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['capital'], balance['cash'] + balance['total value'], rtol=tolerance)
    assert np.allclose(balance['capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)

    actual_return = 1.2041
    return_tolerance = 0.01
    assert np.isclose(balance['accumulated return'].iloc[-1], actual_return, rtol=return_tolerance)


def test_ivy_monthly_rebalance(sample_datahandler):
    bt = run_backtest(sample_datahandler, ivy_portfolio(), periods=1)

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['capital'], balance['cash'] + balance['total value'], rtol=tolerance)
    assert np.allclose(balance['capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)

    actual_return = 1.2043
    return_tolerance = 0.01
    assert np.isclose(balance['accumulated return'].iloc[-1], actual_return, rtol=return_tolerance)


def test_all_weather_portfolio(sample_datahandler):
    bt = run_backtest(sample_datahandler, all_weather_portfolio())

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['capital'], balance['cash'] + balance['total value'], rtol=tolerance)
    assert np.allclose(balance['capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)

    actual_return = 1.1874
    return_tolerance = 0.01
    assert np.isclose(balance['accumulated return'].iloc[-1], actual_return, rtol=return_tolerance)


def test_all_weather_monthly_rebalance(sample_datahandler):
    bt = run_backtest(sample_datahandler, all_weather_portfolio(), periods=1)

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['capital'], balance['cash'] + balance['total value'], rtol=tolerance)
    assert np.allclose(balance['capital'], bt.initial_capital * balance['accumulated return'], rtol=tolerance)

    actual_return = 1.1828
    return_tolerance = 0.01
    assert np.isclose(balance['accumulated return'].iloc[-1], actual_return, rtol=return_tolerance)


def test_constant_price(constant_price_datahandler):
    bt = run_backtest(constant_price_datahandler, ivy_portfolio())

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['% change'], 0.0, rtol=tolerance)
    assert np.allclose(balance['capital'], bt.initial_capital, rtol=tolerance)
    assert np.allclose(balance['total value'], bt.initial_capital, rtol=tolerance)
    assert np.allclose(balance['accumulated return'], 1.0, rtol=tolerance)


def test_zero_initial_capital(sample_datahandler):
    bt = run_backtest(sample_datahandler, ivy_portfolio(), initial_capital=0)

    balance = bt.balance[1:]
    tolerance = 0.0001
    assert np.allclose(balance['capital'], balance['cash'] + balance['total value'], rtol=tolerance)
    assert np.allclose(balance['cash'], 0.0, rtol=tolerance)
    assert np.allclose(balance['total value'], 0.0, rtol=tolerance)


# Helpers
def run_backtest(data, portfolio, initial_capital=1_000_000, periods=None):
    bt = Backtest(data.schema, initial_capital=initial_capital)
    bt.portfolio = portfolio
    bt.data = data
    bt.run(periods=periods)

    return bt


def ivy_portfolio():
    portfolio = Portfolio()
    assets = [Asset('VTI', 0.2), Asset('VEU', 0.2), Asset('BND', 0.2), Asset('VNQ', 0.2), Asset('DBC', 0.2)]

    return portfolio.add_assets(assets)


def all_weather_portfolio():
    portfolio = Portfolio()
    assets = [Asset('VTI', 0.3), Asset('TLT', 0.4), Asset('IEF', 0.15), Asset('GLD', 0.075), Asset('DBC', 0.075)]

    return portfolio.add_assets(assets)
