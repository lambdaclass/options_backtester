import os

import pytest

from backtester.datahandler import HistoricalOptionsData, TiingoData

TEST_DIR = os.path.abspath(os.path.dirname(__file__))

SAMPLE_DATA_STOCKS = os.path.join(TEST_DIR, 'backtester', 'test_data', 'test_data_stocks.csv')
SAMPLE_DATA_OPTIONS = os.path.join(TEST_DIR, 'backtester', 'test_data', 'test_data.csv')


@pytest.fixture(scope='module')
def sample_datahandler_stocks():
    data = TiingoData(SAMPLE_DATA_STOCKS)
    return data


@pytest.fixture(scope='module')
def sample_datahandler_options():
    data = HistoricalOptionsData(SAMPLE_DATA_OPTIONS)
    return data
