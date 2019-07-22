import logging
import unittest
from unittest.mock import patch
import os
import shutil

from requests import ConnectionError
import pandas as pd

from data_scraper import cboe

logging.disable(level=logging.CRITICAL)


class TestCBOE(unittest.TestCase):
    """Tests CBOE data scraper"""

    test_dir = os.path.join(os.getcwd(), os.path.dirname(__file__))
    test_data_path = os.path.realpath(os.path.join(test_dir, "data"))
    cboe_data_path = os.path.join(test_data_path, "cboe")
    spx_data_path = os.path.join(cboe_data_path, "SPX_March_2019.csv")

    @classmethod
    def setUpClass(cls):
        cls.save_data_path = os.environ.get("SAVE_DATA_PATH", None)
        os.environ["SAVE_DATA_PATH"] = cls.test_data_path

    @classmethod
    def tearDownClass(cls):
        if cls.save_data_path:
            os.environ["SAVE_DATA_PATH"] = cls.save_data_path

    @patch("data_scraper.cboe.slack_notification", return_value=None)
    def test_fetch_spy(self, mocked_notification):
        """Fetch todays SPY quote"""
        cboe.fetch_data(["SPY"])
        spy_dir = os.path.join(TestCBOE.cboe_data_path, "SPY_daily")
        self.addCleanup(TestCBOE.remove_files, spy_dir)

        if self.assertTrue(os.path.exists(spy_dir)):
            self.assertTrue(mocked_notification.called)
            file_name = "SPY_" + pd.Timestamp.today().strftime(
                "%Y%m%d") + ".csv"
            file_path = os.path.join(spy_dir, file_name)
            spy_df = pd.read_csv(file_path, parse_dates=["quotedate"])
            self.assertTrue(all(spy_df["underlying"] == "SPX"))
            self.assertEqual(spy_df["quotedate"].nunique(), 1)
            counts = spy_df["type"].value_counts()
            self.assertEqual(counts["put"] + counts["call"], len(spy_df))

    @patch("data_scraper.cboe.slack_notification", return_value=None)
    def test_fetch_invalid_symbol(self, mocked_notification):
        """Fetching invalid symbol should send notification"""
        cboe.fetch_data(["FOOBAR"])
        self.assertTrue(mocked_notification.called)

    @patch("data_scraper.cboe.url", new="http://www.aldkfjaskldfjsa.com")
    @patch("data_scraper.cboe.slack_notification", return_value=None)
    def test_no_connection(self, mocked_notification):
        """Raise ConnectionError and send notification when host is unreachable"""
        with self.assertRaises(ConnectionError):
            cboe.fetch_data(["SPX"])
            self.assertTrue(mocked_notification.called)

    @patch("data_scraper.cboe.utils.remove_file", return_value=None)
    @patch("data_scraper.cboe.slack_notification", return_value=None)
    def test_data_aggregation(self, mocked_notification, mocked_remove):
        """Test data aggregation happy path"""
        cboe.aggregate_monthly_data(["SPX"])
        aggregate_file = os.path.join(TestCBOE.cboe_data_path, "SPX",
                                      "SPX_20190301_to_20190329.csv")
        self.addCleanup(TestCBOE.remove_files, os.path.dirname(aggregate_file))
        self.assertTrue(mocked_remove.called)
        self.assertFalse(mocked_notification.called)

        if self.assertTrue(os.path.exists(aggregate_file)):
            spx_df = pd.read_csv(TestCBOE.spx_data_path)
            aggregate_df = pd.read_csv(aggregate_file)
            self.assertTrue(spx_df.equals(aggregate_df))
    
    @patch("data_scraper.cboe.url", new="http://www.aldkfjaskldfjsa.com")
    @patch("data_scraper.cboe.retry_failure", return_value=None)
    def test_retry(self, mocked_retry):
        """Raise ConnectionError and send notification when host is unreachable"""
        with self.assertRaises(ConnectionError):
            cboe.fetch_data(["SPX"])
            self.assertTrue(mocked_retry.called)
            self.assertTrue(mocked_retry.call_count == 10)


    @patch("data_scraper.cboe.utils.remove_file", return_value=None)
    @patch("data_scraper.cboe.slack_notification", return_value=None)
    def test_aggregate_missing_days(self, mocked_notification, mocked_remove):
        """Data aggregation should send notification when there are missing days"""
        cboe.aggregate_monthly_data(["GOOG"])
        self.assertTrue(mocked_notification.called)
        self.assertFalse(mocked_remove.called)

    @patch("data_scraper.cboe.utils.remove_file", return_value=None)
    @patch("data_scraper.cboe.slack_notification", return_value=None)
    def test_aggregate_invalid_symbol(self, mocked_notification,
                                      mocked_remove):
        """Data aggregation should fail and send notification on invalid symbol"""
        cboe.aggregate_monthly_data(["FOOBAR"])
        self.assertTrue(mocked_notification.called)
        self.assertFalse(mocked_remove.called)

    def remove_files(self, file_path):
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


if __name__ == "__main__":
    unittest.main()
