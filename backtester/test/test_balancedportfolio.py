import unittest
import os
import backtester as bt
from backtester.utils import get_data_dir


class TestBalancedPortfolio(unittest.TestCase):
    """Tests benchmark strategy using synthetic data"""

    @classmethod
    def setUpClass(cls):
        data_dir = get_data_dir()
        balanced_file = os.path.join(data_dir, "balanced_2015.csv")
        cls.port = bt.run(balanced_file)
        cls.report = cls.port.create_report()
        cls.initial_capital = 1000000
        cls.symbols = [
            "VOO", "GLD", "VNQ", "VNQI", "TLT", "TIP", "BNDX", "EEM", "RJI"
        ]

    def test_first_day_allocation(self):
        """First day allocation of cash should be equal to initial capital"""
        self.assertEqual(TestBalancedPortfolio.report["Cash"].iloc[0],
                         TestBalancedPortfolio.initial_capital)

    def test_total_return(self):
        self.assertAlmostEqual(
            TestBalancedPortfolio.report["% Price"][-1], -0.041, delta=0.001)

    def test_voo_allocation(self):
        """Default VOO allocation should equal 30%"""
        self.assertAlmostEqual(
            TestBalancedPortfolio.report["VOO Exposure"].iloc[1],
            TestBalancedPortfolio.initial_capital * 0.3,
            delta=TestBalancedPortfolio.initial_capital * 0.01)

    def test_voo_amount_constant(self):
        """Default VOO amount should remain constant throughout backtest"""
        voo_amounts = TestBalancedPortfolio.report["VOO Amount"]
        self.assertTrue(all(voo_amounts[1:] == voo_amounts.iloc[1]))

    def test_sum_of_all_symbols(self):
        """Sum of all allocations should equal initial capital"""
        total_allocations = sum([
            TestBalancedPortfolio.report[symbol + " Exposure"].iloc[2]
            for symbol in TestBalancedPortfolio.symbols
        ])
        self.assertAlmostEqual(
            total_allocations,
            TestBalancedPortfolio.initial_capital,
            delta=TestBalancedPortfolio.initial_capital * 0.1)


if __name__ == "__main__":
    unittest.main()
