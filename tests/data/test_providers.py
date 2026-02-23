"""Tests for data providers."""

import os
import pytest
import pandas as pd

from options_backtester.data.providers import (
    CsvOptionsProvider, CsvStocksProvider,
    DataProvider, OptionsDataProvider, StocksDataProvider,
)
from options_backtester.data.schema import Schema

# Locate test data relative to the old test fixtures
TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "backtester", "test", "test_data")


@pytest.fixture
def options_provider():
    path = os.path.join(TEST_DATA, "test_options_data.csv")
    if not os.path.exists(path):
        pytest.skip("Test data not available")
    return CsvOptionsProvider(path)


@pytest.fixture
def stocks_provider():
    path = os.path.join(TEST_DATA, "test_stocks_data.csv")
    if not os.path.exists(path):
        pytest.skip("Test data not available")
    return CsvStocksProvider(path)


class TestCsvOptionsProvider:
    def test_is_data_provider(self, options_provider):
        assert isinstance(options_provider, DataProvider)
        assert isinstance(options_provider, OptionsDataProvider)

    def test_has_schema(self, options_provider):
        assert options_provider.schema is not None

    def test_data_is_dataframe(self, options_provider):
        assert isinstance(options_provider.data, pd.DataFrame)

    def test_start_end_dates(self, options_provider):
        assert isinstance(options_provider.start_date, pd.Timestamp)
        assert isinstance(options_provider.end_date, pd.Timestamp)
        assert options_provider.start_date <= options_provider.end_date

    def test_len(self, options_provider):
        assert len(options_provider) > 0

    def test_iter_dates(self, options_provider):
        groups = list(options_provider.iter_dates())
        assert len(groups) > 0


class TestCsvStocksProvider:
    def test_is_data_provider(self, stocks_provider):
        assert isinstance(stocks_provider, DataProvider)
        assert isinstance(stocks_provider, StocksDataProvider)

    def test_has_schema(self, stocks_provider):
        assert stocks_provider.schema is not None

    def test_data_is_dataframe(self, stocks_provider):
        assert isinstance(stocks_provider.data, pd.DataFrame)

    def test_start_end_dates(self, stocks_provider):
        assert isinstance(stocks_provider.start_date, pd.Timestamp)
        assert isinstance(stocks_provider.end_date, pd.Timestamp)

    def test_len(self, stocks_provider):
        assert len(stocks_provider) > 0


class TestSchemaReExport:
    def test_schema_import(self):
        from options_backtester.data.schema import Schema, Field, Filter
        assert Schema is not None
        assert Field is not None
        assert Filter is not None

    def test_options_schema(self):
        s = Schema.options()
        assert "bid" in s
        assert "ask" in s
