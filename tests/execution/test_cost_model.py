"""Tests for transaction cost models."""

from options_portfolio_backtester.execution.cost_model import (
    NoCosts, PerContractCommission, TieredCommission, SpreadSlippage,
)


class TestNoCosts:
    def test_option_cost_is_zero(self):
        m = NoCosts()
        assert m.option_cost(2.50, 10, 100) == 0.0

    def test_stock_cost_is_zero(self):
        m = NoCosts()
        assert m.stock_cost(150.0, 100) == 0.0


class TestPerContractCommission:
    def test_default_rate(self):
        m = PerContractCommission()
        assert m.option_cost(2.50, 10, 100) == 6.50

    def test_custom_rate(self):
        m = PerContractCommission(rate=1.00)
        assert m.option_cost(2.50, 10, 100) == 10.00

    def test_stock_rate(self):
        m = PerContractCommission(stock_rate=0.01)
        assert m.stock_cost(150.0, 100) == 1.00

    def test_negative_qty_uses_abs(self):
        m = PerContractCommission(rate=0.65)
        assert m.option_cost(2.50, -10, 100) == 6.50


class TestTieredCommission:
    def test_default_tiers_small_qty(self):
        m = TieredCommission()
        cost = m.option_cost(2.50, 100, 100)
        assert cost == 100 * 0.65

    def test_default_tiers_large_qty(self):
        m = TieredCommission()
        # 10000 @ 0.65 + 5000 @ 0.50
        cost = m.option_cost(2.50, 15000, 100)
        assert cost == 10000 * 0.65 + 5000 * 0.50


class TestSpreadSlippage:
    def test_zero_pct(self):
        m = SpreadSlippage(pct=0.0)
        assert m.slippage(1.0, 1.10, 10, 100) == 0.0

    def test_half_spread(self):
        m = SpreadSlippage(pct=0.5)
        # spread = 0.10, half = 0.05, * 10 * 100 = 50.0
        assert abs(m.slippage(1.0, 1.10, 10, 100) - 50.0) < 1e-8

    def test_full_spread(self):
        m = SpreadSlippage(pct=1.0)
        assert abs(m.slippage(1.0, 1.10, 10, 100) - 100.0) < 1e-8

    def test_option_cost_is_zero(self):
        """SpreadSlippage models slippage separately, not via option_cost."""
        m = SpreadSlippage(pct=0.5)
        assert m.option_cost(2.50, 10, 100) == 0.0
