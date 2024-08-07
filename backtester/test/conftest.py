import os

import pytest

from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.enums import Stock

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DATA_STOCKS = os.path.join(TEST_DIR, 'test_data', 'test_data_stocks.csv')
IVY_PORTFOLIO_DATA = os.path.join(TEST_DIR, 'test_data', 'ivy_portfolio.csv')
SAMPLE_DATA_OPTIONS = os.path.join(TEST_DIR, 'test_data', 'test_data_options.csv')

IVY_PORTFOLIO_5ASSETS_DATA = os.path.join(TEST_DIR, 'test_data', 'ivy_5assets_data.csv')
TWO_PUTS_TWO_CALLS_DATA = os.path.join(TEST_DIR, 'test_data', 'options_data.csv')

# DataHandler fixtures


# @pytest.fixture(scope='module')
def sample_stocks_datahandler():
    data = TiingoData(SAMPLE_DATA_STOCKS)
    return data


@pytest.fixture(scope='module')
def ivy_portfolio_datahandler():
    data = TiingoData(IVY_PORTFOLIO_DATA)
    return data


@pytest.fixture(scope='module')
def constant_price_stocks():
    data = TiingoData(SAMPLE_DATA_STOCKS)
    data['adjClose'] = data['close'] = 10.0
    return data


# @pytest.fixture(scope='module')
def sample_options_datahandler():
    data = HistoricalOptionsData(SAMPLE_DATA_OPTIONS)
    return data


@pytest.fixture(scope='module')
def ivy_portfolio_5assets_datahandler():
    data = TiingoData(IVY_PORTFOLIO_5ASSETS_DATA)
    data._data['adjClose'] = 10
    return data


# Stock Porfolio fixtures


@pytest.fixture(scope='module')
def ivy_portfolio():
    return [Stock('VTI', 0.2), Stock('VEU', 0.2), Stock('BND', 0.2), Stock('VNQ', 0.2), Stock('DBC', 0.2)]


# @pytest.fixture(scope='module')
def sample_stock_portfolio():
    VOO = Stock('VOO', 0.4)
    TUR = Stock('TUR', 0.1)
    RSX = Stock('RSX', 0.5)

    return [VOO, TUR, RSX]


@pytest.fixture(scope='module')
def ivy_5assets_portfolio():
    VTI = Stock("VTI", 0.2)
    VEU = Stock("VEU", 0.2)
    BND = Stock("BND", 0.2)
    VNQ = Stock("VNQ", 0.2)
    DBC = Stock("DBC", 0.2)
    return [VTI, VEU, BND, VNQ, DBC]


@pytest.fixture(scope='module')
def options_data_2puts_buy():
    data = HistoricalOptionsData(TWO_PUTS_TWO_CALLS_DATA)
    data._data.at[2, 'ask'] = 1  # SPX6500 put 2014-12-15
    data._data.at[2, 'bid'] = 0.5  # SPX6500 put 2014-12-15

    data._data.at[51, 'ask'] = 1.5  # SPX7000 put 2015-01-02
    data._data.at[50, 'bid'] = 0.5  # SPX6500 put 2015-01-02

    data._data.at[130, 'bid'] = 0.5  # SPX6500 put 2015-02-02
    data._data.at[131, 'bid'] = 1.5  # SPX7000 put 2015-02-02

    data._data.at[206, 'bid'] = 0.5  # SPX6500 put 2015-03-02
    data._data.at[207, 'bid'] = 1.5  # SPX7000 put 2015-03-02
    return data


@pytest.fixture(scope='module')
def options_data_2legs_buy():
    data = HistoricalOptionsData(TWO_PUTS_TWO_CALLS_DATA)
    data._data.at[0, 'ask'] = 1  # SPX6500 call 2014-12-15
    data._data.at[0, 'bid'] = 0.5  # SPX6500 call 2014-12-15
    data._data.at[2, 'ask'] = 1  # SPX6500 put 2014-12-15
    data._data.at[2, 'bid'] = 0.5  # SPX6500 put 2014-12-15

    data._data.at[51, 'ask'] = 1.5  # SPX7000 put 2015-01-02
    data._data.at[50, 'bid'] = 0.5  # SPX6500 put 2015-01-02
    data._data.at[49, 'ask'] = 1.5  # SPX7000 call 2015-01-02
    data._data.at[48, 'bid'] = 0.5  # SPX6500 call 2015-01-02

    data._data.at[130, 'bid'] = 0.5  # SPX6500 put 2015-02-02
    data._data.at[131, 'bid'] = 1.5  # SPX7000 put 2015-02-02
    data._data.at[128, 'bid'] = 0.5  # SPX6500 call 2015-02-02
    data._data.at[129, 'bid'] = 1.5  # SPX7000 call 2015-02-02

    data._data.at[206, 'bid'] = 0.5  # SPX6500 put 2015-03-02
    data._data.at[207, 'bid'] = 1.5  # SPX7000 put 2015-03-02
    data._data.at[204, 'bid'] = 0.5  # SPX6500 call 2015-03-02
    data._data.at[205, 'bid'] = 1.5  # SPX7000 call 2015-03-02
    return data


@pytest.fixture(scope='module')
def options_data_1put_buy_sell():
    data = HistoricalOptionsData(TWO_PUTS_TWO_CALLS_DATA)
    data._data.at[2, 'ask'] = 1  # SPX6500 put 2014-12-15
    data._data.at[2, 'bid'] = 0.5  # SPX6500 put 2014-12-15

    data._data.at[50, 'ask'] = 1.5  # SPX6500 put 2015-01-02
    data._data.at[50, 'bid'] = 1  # SPX6500 put 2015-01-02

    data._data.at[130, 'bid'] = 2  # SPX6500 put 2015-02-02
    data._data.at[130, 'ask'] = 2.5  # SPX6500 put 2015-02-02

    data._data.at[206, 'bid'] = 2  # SPX6500 put 2015-03-02
    data._data.at[206, 'ask'] = 2.5  # SPX7000 put 2015-03-02
    return data


@pytest.fixture(scope='module')
def options_data_buy_and_sell_2legs():
    data = HistoricalOptionsData(TWO_PUTS_TWO_CALLS_DATA)
    data._data.at[0, 'ask'] = 1  # SPX6500 call 2014-12-15
    data._data.at[0, 'bid'] = 0.5  # SPX6500 call 2014-12-15
    data._data.at[2, 'ask'] = 1  # SPX6500 put 2014-12-15
    data._data.at[2, 'bid'] = 0.5  # SPX6500 put 2014-12-15

    data._data.at[51, 'ask'] = 1.5  # SPX7000 put 2015-01-02
    data._data.at[50, 'bid'] = 0.5  # SPX6500 put 2015-01-02
    data._data.at[49, 'ask'] = 1.5  # SPX7000 call 2015-01-02
    data._data.at[48, 'bid'] = 0.5  # SPX6500 call 2015-01-02

    data._data.at[130, 'bid'] = 0.5  # SPX6500 put 2015-02-02
    data._data.at[131, 'bid'] = 1.5  # SPX7000 put 2015-02-02
    data._data.at[128, 'bid'] = 0.5  # SPX6500 call 2015-02-02
    data._data.at[129, 'bid'] = 1.5  # SPX7000 call 2015-02-02

    data._data.at[206, 'bid'] = 1  # SPX6500 put 2015-03-02
    data._data.at[207, 'bid'] = 1.5  # SPX7000 put 2015-03-02
    data._data.at[204, 'bid'] = 1.  # SPX6500 call 2015-03-02
    data._data.at[205, 'bid'] = 1.5  # SPX7000 call 2015-03-02
    return data


@pytest.fixture(scope='module')
def options_data_1put_buy_sell_all():
    data = HistoricalOptionsData(TWO_PUTS_TWO_CALLS_DATA)
    data._data.at[2, 'ask'] = 1  # SPX6500 put 2014-12-15
    data._data.at[2, 'bid'] = 0.5  # SPX6500 put 2014-12-15

    data._data.at[50, 'ask'] = 0.25  # SPX6500 put 2015-01-02
    data._data.at[50, 'bid'] = 0.25  # SPX6500 put 2015-01-02
    data._data.at[51, 'ask'] = 1  # SPX6500 put 2015-01-02

    data._data.at[131, 'bid'] = 10  # SPX7000 put 2015-02-02
    data._data.at[130, 'bid'] = 1.5  # SPX6500 put 2015-02-02

    data._data.at[206, 'bid'] = 2  # SPX6500 put 2015-03-02
    data._data.at[206, 'ask'] = 2.5  # SPX6500 put 2015-03-02
    return data
