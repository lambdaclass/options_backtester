import os

import pytest

from asset_backtester.datahandler import HistoricalAssetData

TEST_DIR = os.path.abspath(os.path.dirname(__file__))

# 2019 data for TLT, GLD, IEF, VTI, VEU, BND, VNQ and DBC
SAMPLE_DATA = os.path.join(TEST_DIR, 'test_data', 'sample_data.csv')


@pytest.fixture(scope='module')
def sample_datahandler():
    data = HistoricalAssetData(SAMPLE_DATA)
    return data


@pytest.fixture(scope='module')
def constant_price_datahandler():
    data = HistoricalAssetData(SAMPLE_DATA)
    data['adjClose'] = data['close'] = 10.0
    return data
