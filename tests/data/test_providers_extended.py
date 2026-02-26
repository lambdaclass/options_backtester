"""Extended tests for data providers — accessors, iteration, edge cases."""

import os
import pandas as pd
import pytest

from options_portfolio_backtester.data.providers import (
    TiingoData, HistoricalOptionsData,
    CsvOptionsProvider, CsvStocksProvider,
)
from options_portfolio_backtester.data.schema import Schema, Filter


@pytest.fixture
def stocks_csv(tmp_path):
    """Create a minimal stocks CSV for testing."""
    csv = tmp_path / "stocks.csv"
    csv.write_text(
        "symbol,date,open,close,high,low,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor\n"
        "SPY,2020-01-02,320,322,323,319,1000000,322,323,319,320,1000000,0,1\n"
        "SPY,2020-01-03,322,321,324,320,1100000,321,324,320,322,1100000,0,1\n"
        "SPY,2020-01-06,321,323,325,320,1200000,323,325,320,321,1200000,0,1\n"
    )
    return str(csv)


@pytest.fixture
def options_csv(tmp_path):
    """Create a minimal options CSV for testing."""
    csv = tmp_path / "options.csv"
    csv.write_text(
        "underlying,underlying_last,quotedate,optionroot,type,expiration,strike,bid,ask,volume,openinterest,last,impliedvol,delta,gamma,theta,vega\n"
        "SPY,322,2020-01-02,SPY_C_450,call,2020-02-21,450,1.5,2.0,500,1000,1.75,0.25,0.30,0.02,-0.05,0.15\n"
        "SPY,322,2020-01-02,SPY_P_300,put,2020-02-21,300,0.8,1.2,300,800,1.00,0.20,-0.20,0.01,-0.03,0.10\n"
        "SPY,321,2020-01-03,SPY_C_450,call,2020-02-21,450,1.4,1.9,400,1000,1.65,0.24,0.28,0.02,-0.05,0.14\n"
        "SPY,321,2020-01-03,SPY_P_300,put,2020-02-21,300,0.9,1.3,350,800,1.10,0.21,-0.21,0.01,-0.03,0.11\n"
        "SPY,323,2020-01-06,SPY_C_450,call,2020-02-21,450,1.6,2.1,600,1000,1.85,0.26,0.32,0.02,-0.05,0.16\n"
        "SPY,323,2020-01-06,SPY_P_300,put,2020-02-21,300,0.7,1.1,250,800,0.90,0.19,-0.19,0.01,-0.03,0.09\n"
    )
    return str(csv)


class TestTiingoData:
    def test_len(self, stocks_csv):
        td = TiingoData(stocks_csv)
        assert len(td) == 3

    def test_getitem_schema_key(self, stocks_csv):
        td = TiingoData(stocks_csv)
        result = td["symbol"]
        assert isinstance(result, pd.Series)
        assert (result == "SPY").all()

    def test_setitem(self, stocks_csv):
        td = TiingoData(stocks_csv)
        td["custom"] = [1, 2, 3]
        assert "custom" in td.schema
        assert td._data["custom"].tolist() == [1, 2, 3]

    def test_repr(self, stocks_csv):
        td = TiingoData(stocks_csv)
        r = repr(td)
        assert "SPY" in r

    def test_start_end_dates(self, stocks_csv):
        td = TiingoData(stocks_csv)
        assert td.start_date == pd.Timestamp("2020-01-02")
        assert td.end_date == pd.Timestamp("2020-01-06")

    def test_iter_dates(self, stocks_csv):
        td = TiingoData(stocks_csv)
        dates = list(td.iter_dates())
        assert len(dates) == 3

    def test_apply_filter(self, stocks_csv):
        td = TiingoData(stocks_csv)
        f = Filter("adjClose > 321")
        result = td.apply_filter(f)
        assert len(result) == 2  # 322 and 323

    def test_getattr_passthrough_method(self, stocks_csv):
        """__getattr__ delegates to _data; head() is a DataFrame method."""
        td = TiingoData(stocks_csv)
        result = td.head(2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_getattr_passthrough_property(self, stocks_csv):
        """__getattr__ delegates to _data; shape is a property."""
        td = TiingoData(stocks_csv)
        assert td.shape == (3, 14)  # 3 rows, 14 columns

    def test_iter_months(self, stocks_csv):
        td = TiingoData(stocks_csv)
        months = list(td.iter_months())
        # All 3 dates are in January 2020, so iter_months groups to 1 month
        assert len(months) >= 1

    def test_sma(self, stocks_csv):
        td = TiingoData(stocks_csv)
        td.sma(2)
        assert "sma" in td._data.columns
        assert "sma" in td.schema


class TestHistoricalOptionsData:
    def test_len(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        assert len(hod) == 6

    def test_dte_column_added(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        assert "dte" in hod._data.columns
        assert (hod._data["dte"] > 0).all()

    def test_getitem_schema_key(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        result = hod["underlying"]
        assert (result == "SPY").all()

    def test_getitem_series_indexing(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        mask = hod._data["type"] == "call"
        result = hod[mask]
        assert len(result) == 3

    def test_setitem(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        hod["flag"] = True
        assert "flag" in hod.schema

    def test_repr(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        r = repr(hod)
        assert "SPY" in r

    def test_iter_dates(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        dates = list(hod.iter_dates())
        assert len(dates) == 3

    def test_iter_months(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        months = list(hod.iter_months())
        assert len(months) >= 1

    def test_getattr_passthrough(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        result = hod.head(3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_apply_filter(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        f = Filter("strike > 400")
        result = hod.apply_filter(f)
        assert len(result) == 3  # only the call rows at strike 450

    def test_start_end_dates(self, options_csv):
        hod = HistoricalOptionsData(options_csv)
        assert hod.start_date == pd.Timestamp("2020-01-02")
        assert hod.end_date == pd.Timestamp("2020-01-06")


class TestCsvStocksProvider:
    def test_data_property(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        assert isinstance(p.data, pd.DataFrame)
        assert len(p.data) == 3

    def test_underscore_data(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        assert p._data is p.data

    def test_schema(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        assert isinstance(p.schema, Schema)

    def test_setitem_getitem(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        p["flag"] = [1, 2, 3]
        assert p._data["flag"].tolist() == [1, 2, 3]

    def test_len(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        assert len(p) == 3

    def test_iter_dates(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        dates = list(p.iter_dates())
        assert len(dates) == 3

    def test_iter_months(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        months = list(p.iter_months())
        assert len(months) >= 1

    def test_apply_filter(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        f = Filter("adjClose > 321")
        result = p.apply_filter(f)
        assert len(result) == 2

    def test_start_end_date(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        assert p.start_date == pd.Timestamp("2020-01-02")
        assert p.end_date == pd.Timestamp("2020-01-06")

    def test_sma(self, stocks_csv):
        p = CsvStocksProvider(stocks_csv)
        p.sma(2)
        assert "sma" in p._data.columns


class TestCsvOptionsProvider:
    def test_data_property(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        assert isinstance(p.data, pd.DataFrame)
        assert len(p.data) == 6

    def test_underscore_data(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        assert p._data is p.data

    def test_setitem_getitem(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        p["flag"] = range(6)
        result = p["flag"]
        assert len(result) == 6

    def test_len(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        assert len(p) == 6

    def test_iter_dates(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        dates = list(p.iter_dates())
        assert len(dates) == 3

    def test_iter_months(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        months = list(p.iter_months())
        assert len(months) >= 1

    def test_apply_filter(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        f = Filter("strike > 400")
        result = p.apply_filter(f)
        assert len(result) == 3

    def test_start_end_date(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        assert p.start_date == pd.Timestamp("2020-01-02")
        assert p.end_date == pd.Timestamp("2020-01-06")

    def test_schema(self, options_csv):
        p = CsvOptionsProvider(options_csv)
        assert isinstance(p.schema, Schema)
