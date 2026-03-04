"""Tests for full options liquidation at rebalance.

Verifies that at every rebalance:
1. All existing option positions are sold (liquidated)
2. Fresh options matching entry criteria are purchased
3. No ghost positions carry over between rebalances
4. Cash accounting is clean (no money creation)
5. Max drawdown never exceeds 100%
"""

import os

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts, PerContractCommission
from options_portfolio_backtester.execution.signal_selector import NearestDelta
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.core.types import Stock, OptionType as Type, Direction

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
STOCKS_FILE = os.path.join(TEST_DIR, "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "options_data.csv")


def _ivy_stocks():
    return [Stock("VTI", 0.2), Stock("VEU", 0.2), Stock("BND", 0.2),
            Stock("VNQ", 0.2), Stock("DBC", 0.2)]


def _stocks_data():
    data = TiingoData(STOCKS_FILE)
    data._data["adjClose"] = 10
    return data


def _options_data():
    data = HistoricalOptionsData(OPTIONS_FILE)
    data._data.at[2, "ask"] = 1
    data._data.at[2, "bid"] = 0.5
    data._data.at[51, "ask"] = 1.5
    data._data.at[50, "bid"] = 0.5
    data._data.at[130, "bid"] = 0.5
    data._data.at[131, "bid"] = 1.5
    data._data.at[206, "bid"] = 0.5
    data._data.at[207, "bid"] = 1.5
    return data


def _build_strategy(schema, direction=Direction.BUY):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=direction)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _run(cost_model=None, direction=Direction.BUY, signal_selector=None):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=cost_model or NoCosts(),
        signal_selector=signal_selector or NearestDelta(target_delta=-0.30),
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _build_strategy(schema, direction=direction)
    engine.run(rebalance_freq=1, monthly=False)
    return engine


# ---------------------------------------------------------------------------
# Liquidation trade pattern
# ---------------------------------------------------------------------------

class TestLiquidationTradePattern:
    """Verify trade log shows sell-then-rebuy pattern at rebalances."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _run()

    def test_more_than_initial_entry(self):
        """Full liquidation generates more trades than just the initial entry."""
        assert len(self.engine.trade_log) > 2

    def test_liquidation_events_logged(self):
        """Engine event log contains liquidate_all_options events (Python path)."""
        engine = _make_python_engine()
        logs = engine.events_dataframe()
        assert (logs["event"] == "liquidate_all_options").any()

    def test_sell_trades_alternate_with_buy(self):
        """Trade log alternates: BUY entry, STC liquidation, BUY entry, ..."""
        tl = self.engine.trade_log
        orders = tl["leg_1"]["order"].values
        # First trade should be entry (BTO for BUY direction)
        assert orders[0] == "BTO"
        # Subsequent trades alternate between STC (liquidation) and BTO (new entry)
        for i in range(1, len(orders) - 1, 2):
            assert orders[i] == "STC", f"Trade {i} should be STC, got {orders[i]}"
            if i + 1 < len(orders):
                assert orders[i + 1] == "BTO", f"Trade {i+1} should be BTO, got {orders[i+1]}"


# ---------------------------------------------------------------------------
# Cash accounting
# ---------------------------------------------------------------------------

class TestCashAccounting:
    """Verify no money creation or destruction."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _run()

    def test_max_drawdown_under_100_pct(self):
        """Max drawdown must never exceed 100% — portfolio can't go negative."""
        bal = self.engine.balance["total capital"]
        running_max = bal.cummax()
        dd = (running_max - bal) / running_max
        assert dd.max() < 1.0, f"Max drawdown {dd.max():.4f} >= 100%"

    def test_total_capital_always_positive(self):
        """Total capital should stay positive."""
        bal = self.engine.balance["total capital"]
        assert (bal > 0).all(), f"Negative capital found: min={bal.min()}"

    def test_total_capital_equals_sum_of_parts(self):
        """total capital = cash + stocks capital + options capital on every row."""
        bal = self.engine.balance
        computed = bal["cash"] + bal["stocks capital"] + bal["options capital"]
        # Allow small floating point differences
        diff = (bal["total capital"] - computed).abs()
        assert diff.max() < 0.01, f"Max capital discrepancy: {diff.max()}"

    def test_initial_capital_preserved(self):
        """First row has the initial capital."""
        first_total = self.engine.balance["total capital"].iloc[0]
        assert abs(first_total - 1_000_000) < 1.0

    def test_no_capital_inflation(self):
        """Final capital should not exceed initial by an unreasonable amount.

        With a 3% put allocation on flat stock data ($10 fixed), capital should
        decrease (put premiums are a cost) or stay roughly the same.
        """
        final = self.engine.balance["total capital"].iloc[-1]
        assert final <= 1_050_000, f"Capital inflated to {final} — possible money creation"


# ---------------------------------------------------------------------------
# Direction variants
# ---------------------------------------------------------------------------

class TestDirectionVariants:
    def test_buy_put_cash_stays_positive(self):
        engine = _run(direction=Direction.BUY)
        assert (engine.balance["cash"] >= -0.01).all()

    def test_sell_put_has_credit_entries(self):
        engine = _run(direction=Direction.SELL)
        tl = engine.trade_log
        sto_mask = tl["leg_1"]["order"] == "STO"
        sto_costs = tl.loc[sto_mask, ("leg_1", "cost")].values
        assert all(c < 0 for c in sto_costs if c != 0), (
            f"STO costs should be negative (credit), got: {sto_costs}"
        )

    def test_sell_put_max_dd_under_100(self):
        engine = _run(direction=Direction.SELL)
        bal = engine.balance["total capital"]
        dd = ((bal.cummax() - bal) / bal.cummax()).max()
        assert dd < 1.0, f"Sell-put max drawdown {dd:.4f} >= 100%"


# ---------------------------------------------------------------------------
# Commission impact
# ---------------------------------------------------------------------------

class TestCommissionImpact:
    def test_commission_reduces_capital(self):
        """More trades from liquidation means commission impact is larger."""
        no_cost = _run()
        with_cost = _run(cost_model=PerContractCommission(0.65))
        no_cost_final = no_cost.balance["total capital"].iloc[-1]
        cost_final = with_cost.balance["total capital"].iloc[-1]
        assert cost_final < no_cost_final

    def test_high_commission_still_positive(self):
        """Even with high commissions, capital stays positive."""
        engine = _run(cost_model=PerContractCommission(5.00))
        assert (engine.balance["total capital"] > 0).all()


# ---------------------------------------------------------------------------
# Rust vs Python parity on liquidation
# ---------------------------------------------------------------------------

class TestRustPythonLiquidationParity:
    """Verify Rust and Python paths both produce valid results with full liquidation."""

    def test_trade_count_matches(self):
        """Rust and Python should produce the same number of trades."""
        rust_engine = _run()
        py_engine = _make_python_engine()
        assert len(rust_engine.trade_log) == len(py_engine.trade_log)

    def test_trade_costs_match(self):
        """Rust and Python trade costs should match exactly."""
        rust_engine = _run()
        py_engine = _make_python_engine()
        r_costs = rust_engine.trade_log["totals"]["cost"].values
        p_costs = py_engine.trade_log["totals"]["cost"].values
        assert np.allclose(r_costs, p_costs, rtol=1e-4)

    def test_both_paths_positive_capital(self):
        """Both paths should maintain positive total capital."""
        rust_engine = _run()
        py_engine = _make_python_engine()
        assert (rust_engine.balance["total capital"] > 0).all()
        assert (py_engine.balance["total capital"] > 0).all()


class _NoOpAlgo:
    """Trivial algo that does nothing — used to force Python path."""
    def __call__(self, ctx):
        from options_portfolio_backtester.engine.algo_adapters import EngineStepDecision
        return EngineStepDecision()
    def reset(self):
        pass


def _make_python_engine():
    """Force Python path by using a no-op algo (blocks Rust dispatch)."""
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
        signal_selector=NearestDelta(target_delta=-0.30),
        algos=[_NoOpAlgo()],
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _build_strategy(schema)
    engine.run(rebalance_freq=1, monthly=False)
    return engine
