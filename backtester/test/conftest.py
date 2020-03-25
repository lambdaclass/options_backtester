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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def sample_options_datahandler():
    data = HistoricalOptionsData(SAMPLE_DATA_OPTIONS)
    return data


@pytest.fixture(scope='module')
def ivy_portfolio_5assets_datahandler():
    data = TiingoData(IVY_PORTFOLIO_5ASSETS_DATA)
    data._data['adjClose'] = 10
    return data


@pytest.fixture(scope='module')
def sample_2puts_2calls_datahandler():
    data = HistoricalOptionsData(TWO_PUTS_TWO_CALLS_DATA)
    return data


# Stock Porfolio fixtures


@pytest.fixture(scope='module')
def ivy_portfolio():
    return [Stock('VTI', 0.2), Stock('VEU', 0.2), Stock('BND', 0.2), Stock('VNQ', 0.2), Stock('DBC', 0.2)]


@pytest.fixture(scope='module')
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