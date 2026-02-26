"""Parity tests for every execution model: Rust vs Python must match."""

from __future__ import annotations

import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    buy_put_strategy,
    run_rust,
    run_python,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# ── Cost Model Parity ──────────────────────────────────────────────────

class TestCostModelParity:
    """Every cost model must produce identical results in Rust and Python."""

    def test_per_contract_commission_parity(self):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=0.65, stock_rate=0.005)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        assert_parity(py, rs, label="PerContract(0.65)")

    def test_per_contract_high_rate_parity(self):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=5.0, stock_rate=0.01)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        assert_parity(py, rs, label="PerContract(5.0)")

    def test_tiered_commission_default_parity(self):
        from options_portfolio_backtester.execution.cost_model import (
            TieredCommission,
        )
        cm = TieredCommission()
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        assert_parity(py, rs, label="TieredDefault")

    def test_tiered_commission_custom_tiers_parity(self):
        from options_portfolio_backtester.execution.cost_model import (
            TieredCommission,
        )
        tiers = [(5_000, 0.80), (20_000, 0.50), (100_000, 0.20)]
        cm = TieredCommission(tiers=tiers, stock_rate=0.01)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm)
        assert_parity(py, rs, label="TieredCustom")


# ── Fill Model Parity ──────────────────────────────────────────────────

class TestFillModelParity:
    """Every fill model must produce identical results in Rust and Python."""

    def test_midprice_parity(self):
        from options_portfolio_backtester.execution.fill_model import MidPrice
        fm = MidPrice()
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      fill_model=fm)
        assert_parity(py, rs, label="MidPrice")

    def test_volume_aware_parity(self):
        from options_portfolio_backtester.execution.fill_model import (
            VolumeAwareFill,
        )
        fm = VolumeAwareFill(full_volume_threshold=100)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      fill_model=fm)
        assert_parity(py, rs, label="VolumeAware(100)")

    def test_volume_aware_low_threshold_parity(self):
        from options_portfolio_backtester.execution.fill_model import (
            VolumeAwareFill,
        )
        fm = VolumeAwareFill(full_volume_threshold=10)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      fill_model=fm)
        assert_parity(py, rs, label="VolumeAware(10)")


# ── Signal Selector Parity ─────────────────────────────────────────────

class TestSignalSelectorParity:
    """Every signal selector must produce identical results."""

    def test_nearest_delta_parity(self):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        ss = NearestDelta(target_delta=-0.30)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      signal_selector=ss)
        assert_parity(py, rs, label="NearestDelta(-0.30)")

    def test_nearest_delta_deep_otm_parity(self):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        ss = NearestDelta(target_delta=-0.10)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      signal_selector=ss)
        assert_parity(py, rs, label="NearestDelta(-0.10)")

    def test_max_open_interest_parity(self):
        from options_portfolio_backtester.execution.signal_selector import (
            MaxOpenInterest,
        )
        ss = MaxOpenInterest()
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      signal_selector=ss)
        assert_parity(py, rs, label="MaxOpenInterest")


# ── Risk Constraint Parity ─────────────────────────────────────────────

class TestRiskConstraintParity:
    """Every risk constraint must produce identical results."""

    def _rm(self, *constraints):
        from options_portfolio_backtester.portfolio.risk import RiskManager
        return RiskManager(list(constraints))

    def test_max_delta_parity(self):
        from options_portfolio_backtester.portfolio.risk import MaxDelta
        rm = self._rm(MaxDelta(limit=100.0))
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label="MaxDelta(100)")

    def test_max_delta_tight_parity(self):
        from options_portfolio_backtester.portfolio.risk import MaxDelta
        rm = self._rm(MaxDelta(limit=0.01))
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label="MaxDelta(0.01)")

    def test_max_vega_parity(self):
        from options_portfolio_backtester.portfolio.risk import MaxVega
        rm = self._rm(MaxVega(limit=50.0))
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label="MaxVega(50)")

    def test_max_drawdown_parity(self):
        from options_portfolio_backtester.portfolio.risk import MaxDrawdown
        rm = self._rm(MaxDrawdown(max_dd_pct=0.20))
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label="MaxDrawdown(0.20)")

    def test_combined_constraints_parity(self):
        from options_portfolio_backtester.portfolio.risk import MaxDelta, MaxVega
        rm = self._rm(MaxDelta(limit=100.0), MaxVega(limit=50.0))
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        risk_manager=rm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      risk_manager=rm)
        assert_parity(py, rs, label="MaxDelta+MaxVega")


# ── Exit Threshold Parity ──────────────────────────────────────────────

class TestExitThresholdParity:
    """Exit thresholds must trigger identically in Rust and Python."""

    def _strat_with_thresholds(self, profit_pct=None, loss_pct=None):
        """Return a strategy factory that adds exit thresholds."""
        import math

        def factory(schema):
            strat = buy_put_strategy(schema)
            strat.add_exit_thresholds(
                profit_pct if profit_pct is not None else math.inf,
                loss_pct if loss_pct is not None else math.inf,
            )
            return strat
        return factory

    def test_profit_threshold_parity(self):
        sf = self._strat_with_thresholds(profit_pct=0.5)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        assert_parity(py, rs, label="profit_pct=0.5")

    def test_loss_threshold_parity(self):
        sf = self._strat_with_thresholds(loss_pct=0.3)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        assert_parity(py, rs, label="loss_pct=0.3")

    def test_both_thresholds_parity(self):
        sf = self._strat_with_thresholds(profit_pct=0.5, loss_pct=0.3)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        assert_parity(py, rs, label="profit+loss")

    def test_tight_profit_threshold_parity(self):
        sf = self._strat_with_thresholds(profit_pct=0.01)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, sf)
        assert_parity(py, rs, label="profit_pct=0.01")
