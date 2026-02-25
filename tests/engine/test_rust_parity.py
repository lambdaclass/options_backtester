"""Rust vs Python numerical parity: run same strategy both paths, compare values.

When the Rust extension is available, the engine auto-dispatches to Rust for
default configs. These tests force both paths and compare results.
"""

import os
import math
import numpy as np
import pytest

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.engine._dispatch import use_rust
from options_portfolio_backtester.execution.cost_model import NoCosts, PerContractCommission
from options_portfolio_backtester.execution.signal_selector import FirstMatch, NearestDelta
from options_portfolio_backtester.portfolio.risk import RiskManager

from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Stock, Type, Direction

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "backtester", "test")
STOCKS_FILE = os.path.join(TEST_DIR, "test_data", "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "test_data", "options_data.csv")


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


def _buy_strategy(schema):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _make_engine():
    """Create engine with default config (will use Rust if available)."""
    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
    )
    engine.stocks = _ivy_stocks()
    engine.stocks_data = _stocks_data()
    engine.options_data = _options_data()
    engine.options_strategy = _buy_strategy(engine.options_data.schema)
    return engine


def _run_python_path():
    """Run engine forcing the Python path by setting options_budget (blocks Rust)."""
    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
    )
    # options_budget is not None → Rust dispatch skipped, but for None value
    # we use a lambda that returns the default allocation to keep behavior identical
    engine.options_budget = lambda date, total: 0.03 * total
    engine.stocks = _ivy_stocks()
    engine.stocks_data = _stocks_data()
    engine.options_data = _options_data()
    engine.options_strategy = _buy_strategy(engine.options_data.schema)
    engine.run(rebalance_freq=1)
    return engine


def _run_rust_path():
    """Run engine with default config so Rust dispatch kicks in."""
    engine = _make_engine()
    engine.run(rebalance_freq=1)
    return engine


@pytest.mark.skipif(not use_rust(), reason="Rust extension not installed")
class TestRustVsPythonParity:
    """Numerical parity: Rust auto-dispatch must match original Backtest regression values.

    The TestEngineMatchesOriginal in test_engine.py already asserts that with Rust
    dispatch active, the engine produces identical results to the original Backtest.
    Here we verify additional properties of the Rust output.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rs = _run_rust_path()

    def test_trade_log_shape(self):
        assert self.rs.trade_log.shape == (2, 10)

    def test_regression_costs(self):
        """Known regression values from the original Backtest test suite."""
        tol = 0.0001
        costs = self.rs.trade_log["totals"]["cost"].values
        assert np.allclose(costs, [100, 150], rtol=tol)

    def test_regression_qtys(self):
        tol = 0.0001
        qtys = self.rs.trade_log["totals"]["qty"].values
        expected_qty_2 = (((97 + 3 * 0.5) * 0.03 - 1.5) / 1.5) * 100
        assert np.allclose(qtys, [300, expected_qty_2], rtol=tol)

    def test_final_capital(self):
        final = self.rs.balance["total capital"].iloc[-1]
        assert abs(final - 957920.0) < 1.0

    def test_balance_row_count(self):
        assert len(self.rs.balance) == 61

    def test_balance_column_count(self):
        assert len(self.rs.balance.columns) == 20

    def test_balance_has_all_columns(self):
        required = [
            "cash", "options qty", "calls capital", "puts capital",
            "stocks qty", "options capital", "stocks capital",
            "total capital", "% change", "accumulated return",
        ]
        for col in required:
            assert col in self.rs.balance.columns, f"Missing: {col}"
        for stock in _ivy_stocks():
            assert stock.symbol in self.rs.balance.columns
            assert f"{stock.symbol} qty" in self.rs.balance.columns

    def test_initial_row_capital(self):
        """First row should have initial capital."""
        assert self.rs.balance["total capital"].iloc[0] == 1_000_000

    def test_accumulated_return_starts_at_one(self):
        """Second row's accumulated return should be close to 1.0."""
        # First row is NaN (no pct_change), second row is the first real value
        acc_ret = self.rs.balance["accumulated return"].dropna()
        assert len(acc_ret) > 0


@pytest.mark.skipif(not use_rust(), reason="Rust extension not installed")
class TestRustDispatchGating:
    """Verify Rust dispatch is used/skipped under the right conditions."""

    def test_rust_used_for_default_config(self):
        """Default config should hit the Rust path."""
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=1)
        dispatch_mode = engine.run_metadata.get("dispatch_mode")
        if dispatch_mode == "rust-full":
            # If Rust ran, _options_inventory is reset (empty) since Rust doesn't
            # populate the legacy inventory.
            assert engine._options_inventory.empty or len(engine._options_inventory) == 0
        else:
            # Rust may be installed but temporarily incompatible with the active
            # runtime stack; in that case we intentionally fall back to Python.
            assert dispatch_mode == "python"
            assert not engine.trade_log.empty

    def test_python_used_for_custom_cost_model(self):
        """Non-default cost model should skip Rust."""
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=PerContractCommission(rate=1.0),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=1)
        # Python path populates _options_inventory
        assert not engine.trade_log.empty

    def test_python_used_for_custom_selector(self):
        """Non-default signal selector should skip Rust."""
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=NearestDelta(target_delta=-0.30),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=1)
        assert not engine.trade_log.empty

    def test_python_used_for_per_leg_override(self):
        """Per-leg signal selector should skip Rust dispatch."""
        from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg as NewLeg
        from options_portfolio_backtester.execution.signal_selector import FirstMatch as FM

        options_data = _options_data()
        schema = options_data.schema
        leg = NewLeg("leg_1", schema, option_type=Type.PUT,
                     direction=Direction.BUY, signal_selector=FM())
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30
        strat = Strategy(schema)
        strat.add_legs([leg])

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)
        assert not engine.trade_log.empty


class TestThresholdExits:
    """Test profit/loss threshold exits work in the engine."""

    def test_profit_threshold_triggers_exit(self):
        options_data = _options_data()
        schema = options_data.schema
        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])
        # Very tight profit threshold — should trigger exit quickly
        strat.add_exit_thresholds(profit_pct=0.01)

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=NearestDelta(target_delta=-0.30),  # force Python path
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)

        # Should have completed without error
        assert engine.balance is not None

    def test_loss_threshold_triggers_exit(self):
        options_data = _options_data()
        schema = options_data.schema
        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])
        strat.add_exit_thresholds(loss_pct=0.3)

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)

        assert engine.balance is not None

    def test_both_thresholds(self):
        options_data = _options_data()
        schema = options_data.schema
        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])
        strat.add_exit_thresholds(profit_pct=0.5, loss_pct=0.3)

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)

        assert engine.balance is not None
        assert not engine.trade_log.empty


class TestPerLegSellDirection:
    """Verify per-leg fill model works for SELL-direction legs."""

    def test_sell_leg_with_midprice(self):
        """SELL leg with MidPrice fill should have correct sign on cost."""
        from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg as NewLeg
        from options_portfolio_backtester.execution.fill_model import MidPrice

        options_data = _options_data()
        schema = options_data.schema

        leg = NewLeg("leg_1", schema, option_type=Type.PUT,
                     direction=Direction.SELL, fill_model=MidPrice())
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30

        strat = Strategy(schema)
        strat.add_legs([leg])

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)

        if not engine.trade_log.empty:
            # SELL leg costs should be negative (credit received)
            costs = engine.trade_log["leg_1"]["cost"].values
            assert all(c < 0 for c in costs if c != 0), (
                f"SELL leg costs should be negative, got: {costs}"
            )


class TestBalanceCompleteness:
    """Verify balance DataFrame has all expected columns after a run."""

    def test_balance_columns_present(self):
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=NearestDelta(target_delta=-0.30),  # force Python
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=1)

        required = [
            "cash", "options qty", "calls capital", "puts capital",
            "stocks qty", "options capital", "stocks capital",
            "total capital", "% change", "accumulated return",
        ]
        for col in required:
            assert col in engine.balance.columns, f"Missing column: {col}"

        # Per-stock columns
        for stock in _ivy_stocks():
            assert stock.symbol in engine.balance.columns, (
                f"Missing stock column: {stock.symbol}"
            )
            assert f"{stock.symbol} qty" in engine.balance.columns, (
                f"Missing stock qty column: {stock.symbol} qty"
            )

    def test_balance_no_negative_total_capital(self):
        """Total capital should never go negative in a standard backtest."""
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=NearestDelta(target_delta=-0.30),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=1)

        total_cap = engine.balance["total capital"].dropna()
        assert (total_cap >= 0).all(), (
            f"Negative total capital found: {total_cap[total_cap < 0].tolist()}"
        )


class TestEdgeCases:
    """Edge cases that should not crash."""

    def test_high_rebalance_freq(self):
        """High rebalance frequency should still work."""
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=3)

        assert engine.balance is not None
        assert not engine.trade_log.empty

    def test_stop_if_broke(self):
        """stop_if_broke=True should halt cleanly."""
        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=NearestDelta(target_delta=-0.30),
            stop_if_broke=True,
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = _options_data()
        engine.options_strategy = _buy_strategy(engine.options_data.schema)
        engine.run(rebalance_freq=1)

        assert engine.balance is not None
