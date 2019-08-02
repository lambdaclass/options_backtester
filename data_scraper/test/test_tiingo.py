import logging
import unittest
from unittest.mock import patch
import os
import shutil
import pandas as pd

from data_scraper import tiingo

logging.disable(level=logging.CRITICAL)


class TestTiingo(unittest.TestCase):
    """Tests Tiingo data scraper"""

    test_dir = os.path.join(os.getcwd(), os.path.dirname(__file__))
    test_data_path = os.path.realpath(os.path.join(test_dir, "data"))
    tiingo_data_path = os.path.join(test_data_path, "tiingo")

    @classmethod
    def setUpClass(cls):
        assert "TIINGO_API_KEY" in os.environ, "$TIINGO_API_KEY env variable must be set"
        cls.save_data_path = os.environ.get("SAVE_DATA_PATH", None)
        os.environ["SAVE_DATA_PATH"] = cls.test_data_path

    @classmethod
    def tearDownClass(cls):
        if cls.save_data_path:
            os.environ["SAVE_DATA_PATH"] = cls.save_data_path
    @patch("data_scraper.tiingo.send_report", return_value=None)
    def test_fetch_gld(self, mocked_notification):
        """Fetch GLD data"""
        tiingo.fetch_data(["GLD"])
        gld_dir = os.path.join(TestTiingo.tiingo_data_path, "GLD")
        self.addCleanup(TestTiingo.remove_files, os.path.dirname(gld_dir))

        if self.assertTrue(os.path.exists(gld_dir)):
            self.assertTrue(mocked_notification.called)
            file_name = "GLD_" + pd.Timestamp.today().strftime(
                "%Y%m%d") + ".csv"
            file_path = os.path.join(gld_dir, file_name)
            gld_df = pd.read_csv(file_path)
            self.assertTrue(all(gld_df["symbol"] == "GLD"))
            expected_columns = [
                "symbol", "date", "adjClose", "adjHigh", "adjLow", "adjOpen",
                "adjVolume", "close", "divCash", "high", "low", "open",
                "splitFactor", "volume"
            ]
            self.assertEqual(gld_df.columns, expected_columns)

    @patch("data_scraper.tiingo.send_report", return_value=None)
    def test_fetch_invalid_symbol(self, mocked_report):
        """Fetching invalid symbol data should send notification"""
        tiingo.fetch_data(["FOOBAR"])
        self.assertTrue(mocked_report.called)

    @patch("data_scraper.tiingo.pdr.get_data_tiingo")  # mock pandas_datareader
    @patch("data_scraper.tiingo.send_report", return_value=None)
    def test_no_connection(self, mocked_report, mocked_pdr):
        """Raise ConnectionError and send notification when host is unreachable"""
        mocked_pdr.side_effect = ConnectionError("This is a test")

        with self.assertRaises(ConnectionError):
            tiingo.fetch_data(["IBM"])
            self.assertTrue(mocked_report.called)
    
    @patch("data_scraper.tiingo.pdr.get_data_tiingo")  # mock pandas_datareader
    @patch("data_scraper.tiingo.retry_failure", return_value=None)
    def test_retry(self, mocked_retry, mocked_pdr):
        """Raise ConnectionError and retry when host is unreachable"""
        mocked_pdr.side_effect = ConnectionError("This is a test")
        with self.assertRaises(ConnectionError):
            tiingo.fetch_data(["IBM"])
            self.assertTrue(mocked_retry.called)
            self.assertTrue(mocked_retry.call_count == 10)

    def remove_files(file_path):
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


if __name__ == "__main__":
    unittest.main()
