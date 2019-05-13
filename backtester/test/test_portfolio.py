import unittest
import os
import backtester as bt
from backtester.utils import get_data_dir


class TestPortfolio(unittest.TestCase):
    """Tests benchmark strategy using synthetic data"""

    def setUp(self):
        data_dir = get_data_dir()
        synth_file = os.path.join(data_dir, "synthetic_data.csv")
        self.port = bt.run(synth_file)
        self.report = self.port.create_report()

    def test_last_pct_price(self):
        self.assertAlmostEqual(self.report["% Price"].iloc[-1], 1999)

    def test_mean_10_day_return(self):
        self.assertAlmostEqual(
            self.report["Interval Change"][:10].mean(), 0.31432, delta=0.0001)

    def test_mean_100_day_return(self):
        self.assertAlmostEqual(
            self.report["Interval Change"][:100].mean(), 0.05229, delta=0.0001)

    def test_last_total(self):
        self.assertEqual(self.report["Total Portfolio"].iloc[-1], 2000000000)


if __name__ == "__main__":
    unittest.main()
