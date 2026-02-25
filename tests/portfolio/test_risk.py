"""Tests for risk management."""

from options_portfolio_backtester.core.types import Greeks
from options_portfolio_backtester.portfolio.risk import (
    RiskManager, MaxDelta, MaxVega, MaxDrawdown,
)


class TestMaxDelta:
    def test_within_limit(self):
        c = MaxDelta(limit=100.0)
        assert c.check(Greeks(delta=50.0), Greeks(delta=30.0), 1e6, 1e6) is True

    def test_exceeds_limit(self):
        c = MaxDelta(limit=100.0)
        assert c.check(Greeks(delta=80.0), Greeks(delta=30.0), 1e6, 1e6) is False

    def test_negative_delta(self):
        c = MaxDelta(limit=100.0)
        assert c.check(Greeks(delta=-80.0), Greeks(delta=-30.0), 1e6, 1e6) is False


class TestMaxVega:
    def test_within_limit(self):
        c = MaxVega(limit=50.0)
        assert c.check(Greeks(vega=20.0), Greeks(vega=10.0), 1e6, 1e6) is True

    def test_exceeds_limit(self):
        c = MaxVega(limit=50.0)
        assert c.check(Greeks(vega=40.0), Greeks(vega=20.0), 1e6, 1e6) is False


class TestMaxDrawdown:
    def test_no_drawdown(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        assert c.check(Greeks(), Greeks(), 1e6, 1e6) is True

    def test_within_limit(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        assert c.check(Greeks(), Greeks(), 900_000, 1_000_000) is True

    def test_exceeds_limit(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        assert c.check(Greeks(), Greeks(), 790_000, 1_000_000) is False


class TestRiskManager:
    def test_empty_allows(self):
        rm = RiskManager()
        allowed, reason = rm.is_allowed(Greeks(), Greeks(), 1e6, 1e6)
        assert allowed is True
        assert reason == ""

    def test_single_constraint_blocks(self):
        rm = RiskManager([MaxDelta(limit=10.0)])
        allowed, reason = rm.is_allowed(Greeks(delta=8.0), Greeks(delta=5.0), 1e6, 1e6)
        assert allowed is False
        assert "MaxDelta" in reason

    def test_multiple_constraints_first_blocks(self):
        rm = RiskManager([MaxDelta(limit=100.0), MaxDrawdown(max_dd_pct=0.10)])
        allowed, reason = rm.is_allowed(Greeks(delta=50.0), Greeks(delta=10.0),
                                        850_000, 1_000_000)
        assert allowed is False
        assert "MaxDrawdown" in reason

    def test_add_constraint(self):
        rm = RiskManager()
        rm.add_constraint(MaxVega(limit=5.0))
        allowed, _ = rm.is_allowed(Greeks(vega=3.0), Greeks(vega=3.0), 1e6, 1e6)
        assert allowed is False
