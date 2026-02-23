"""Tests for position sizing models."""

from options_backtester.execution.sizer import (
    CapitalBased, FixedQuantity, FixedDollar, PercentOfPortfolio,
)


class TestCapitalBased:
    def test_basic_sizing(self):
        s = CapitalBased()
        assert s.size(250.0, 1000.0, 100_000.0) == 4

    def test_zero_cost(self):
        s = CapitalBased()
        assert s.size(0.0, 1000.0, 100_000.0) == 0

    def test_insufficient_capital(self):
        s = CapitalBased()
        assert s.size(500.0, 100.0, 100_000.0) == 0


class TestFixedQuantity:
    def test_fixed(self):
        s = FixedQuantity(quantity=5)
        assert s.size(100.0, 10_000.0, 100_000.0) == 5

    def test_insufficient_capital_reduces(self):
        s = FixedQuantity(quantity=10)
        assert s.size(100.0, 500.0, 100_000.0) == 5


class TestFixedDollar:
    def test_fixed_amount(self):
        s = FixedDollar(amount=1000.0)
        assert s.size(250.0, 10_000.0, 100_000.0) == 4

    def test_amount_capped_by_available(self):
        s = FixedDollar(amount=10_000.0)
        assert s.size(250.0, 500.0, 100_000.0) == 2


class TestPercentOfPortfolio:
    def test_one_percent(self):
        s = PercentOfPortfolio(pct=0.01)
        # 1% of 100k = 1000, 1000 // 250 = 4
        assert s.size(250.0, 10_000.0, 100_000.0) == 4

    def test_capped_by_available(self):
        s = PercentOfPortfolio(pct=0.10)
        # 10% of 100k = 10000, but only 500 available
        assert s.size(250.0, 500.0, 100_000.0) == 2
