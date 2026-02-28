"""Tests for multi-strategy support within BacktestEngine.

Verifies that add_strategy() + run() produces correct results when
multiple strategies share a single capital pool and balance sheet.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.engine.engine import BacktestEngine, _StrategySlot
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.core.types import (
    Stock, OptionType as Type, Direction,
)

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
STOCKS_FILE = os.path.join(TEST_DIR, "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "options_data.csv")


def _ivy_stocks():
    return [
        Stock("VTI", 0.2), Stock("VEU", 0.2), Stock("BND", 0.2),
        Stock("VNQ", 0.2), Stock("DBC", 0.2),
    ]


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


def _buy_put_strategy(schema):
    """Single BUY PUT leg."""
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _sell_call_strategy(schema):
    """Single SELL CALL leg."""
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.SELL)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _make_engine(**kwargs):
    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
        **kwargs,
    )
    engine.stocks = _ivy_stocks()
    engine.stocks_data = _stocks_data()
    engine.options_data = _options_data()
    return engine


# ---------------------------------------------------------------------------
# _StrategySlot unit tests
# ---------------------------------------------------------------------------

class TestStrategySlot:
    def test_dataclass_fields(self):
        schema = _options_data().schema
        strat = _buy_put_strategy(schema)
        slot = _StrategySlot(
            strategy=strat, weight=0.5, rebalance_freq=1,
            name="test_slot",
        )
        assert slot.weight == 0.5
        assert slot.rebalance_freq == 1
        assert slot.rebalance_unit == "BMS"
        assert slot.check_exits_daily is False
        assert slot.name == "test_slot"
        assert slot.inventory is None
        assert slot.rebalance_dates is None


# ---------------------------------------------------------------------------
# add_strategy() API tests
# ---------------------------------------------------------------------------

class TestAddStrategy:
    def test_adds_slot(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        strat = _buy_put_strategy(schema)
        engine.add_strategy(strat, weight=1.0, rebalance_freq=1)
        assert engine._is_multi_strategy
        assert len(engine._strategy_slots) == 1

    def test_auto_names(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(_buy_put_strategy(schema), weight=0.5, rebalance_freq=1)
        engine.add_strategy(_sell_call_strategy(schema), weight=0.5, rebalance_freq=1)
        assert engine._strategy_slots[0].name == "strategy_0"
        assert engine._strategy_slots[1].name == "strategy_1"

    def test_custom_names(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="hedge"
        )
        engine.add_strategy(
            _sell_call_strategy(schema), weight=0.5, rebalance_freq=1, name="income"
        )
        assert engine._strategy_slots[0].name == "hedge"
        assert engine._strategy_slots[1].name == "income"

    def test_not_multi_strategy_by_default(self):
        engine = _make_engine()
        assert not engine._is_multi_strategy


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_weights_must_sum_to_one(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(_buy_put_strategy(schema), weight=0.5, rebalance_freq=1)
        engine.add_strategy(_sell_call_strategy(schema), weight=0.3, rebalance_freq=1)
        with pytest.raises(AssertionError, match="weights must sum to 1.0"):
            engine.run(rebalance_freq=1)


# ---------------------------------------------------------------------------
# Two strategies, same frequency
# ---------------------------------------------------------------------------

class TestSameFrequency:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _make_engine()
        schema = self.engine.options_data.schema
        self.engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="hedge"
        )
        self.engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="hedge2"
        )
        self.engine.run()

    def test_balance_not_empty(self):
        assert not self.engine.balance.empty

    def test_balance_has_required_columns(self):
        cols = self.engine.balance.columns
        for c in ["cash", "calls capital", "puts capital",
                   "options capital", "stocks capital",
                   "total capital", "% change", "accumulated return"]:
            assert c in cols, f"Missing column: {c}"

    def test_dispatch_mode_is_python_multi(self):
        assert self.engine.run_metadata["dispatch_mode"] == "python-multi"

    def test_trade_log_type(self):
        # May be empty if no candidates, but must be a DataFrame
        assert isinstance(self.engine.trade_log, pd.DataFrame)


# ---------------------------------------------------------------------------
# Two strategies, different frequencies
# ---------------------------------------------------------------------------

class TestDifferentFrequency:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _make_engine()
        schema = self.engine.options_data.schema
        # Strategy A: rebalance every 1 BMS
        self.engine.add_strategy(
            _buy_put_strategy(schema), weight=0.7, rebalance_freq=1, name="monthly"
        )
        # Strategy B: rebalance every 2 BMS (less frequent)
        self.engine.add_strategy(
            _buy_put_strategy(schema), weight=0.3, rebalance_freq=2, name="bimonthly"
        )
        self.engine.run()

    def test_balance_not_empty(self):
        assert not self.engine.balance.empty

    def test_total_capital_computed(self):
        assert "total capital" in self.engine.balance.columns
        # Total capital should exist and be positive
        assert self.engine.balance["total capital"].iloc[-1] > 0


# ---------------------------------------------------------------------------
# Single strategy via old API is unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_single_strategy_api_unchanged(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.options_strategy = _buy_put_strategy(schema)
        engine.run(rebalance_freq=1)
        assert not engine.balance.empty
        assert engine.run_metadata["dispatch_mode"] in {"python", "rust-full"}


# ---------------------------------------------------------------------------
# Shared cash pool
# ---------------------------------------------------------------------------

class TestSharedCash:
    def test_cash_flows_into_shared_pool(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="a"
        )
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="b"
        )
        engine.run()
        # After running, current_cash should be a single float (shared pool)
        assert isinstance(engine.current_cash, float)


# ---------------------------------------------------------------------------
# Per-strategy exit thresholds
# ---------------------------------------------------------------------------

class TestPerStrategyExitThresholds:
    def test_different_exit_thresholds(self):
        engine = _make_engine()
        schema = engine.options_data.schema

        strat_tight = _buy_put_strategy(schema)
        strat_tight.add_exit_thresholds(profit_pct=0.1, loss_pct=0.1)

        strat_loose = _buy_put_strategy(schema)
        strat_loose.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

        engine.add_strategy(strat_tight, weight=0.5, rebalance_freq=1, name="tight")
        engine.add_strategy(strat_loose, weight=0.5, rebalance_freq=1, name="loose")
        # Should not crash — exit thresholds are read from each strategy via context
        engine.run()
        assert not engine.balance.empty


# ---------------------------------------------------------------------------
# check_exits_daily per-strategy
# ---------------------------------------------------------------------------

class TestPerStrategyDailyExits:
    def test_daily_exits_per_slot(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1,
            check_exits_daily=True, name="daily_exits"
        )
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1,
            check_exits_daily=False, name="no_daily_exits"
        )
        engine.run()
        assert not engine.balance.empty

    def test_global_check_exits_daily(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="a"
        )
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="b"
        )
        # Global check_exits_daily should apply to all slots
        engine.run(check_exits_daily=True)
        assert not engine.balance.empty


# ---------------------------------------------------------------------------
# stop_if_broke halts entire engine
# ---------------------------------------------------------------------------

class TestStopIfBroke:
    def test_stop_halts_multi_strategy(self):
        engine = _make_engine(stop_if_broke=True)
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="a"
        )
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="b"
        )
        # Should not crash; stop_if_broke is checked in multi-strategy loop
        engine.run()
        assert isinstance(engine.balance, pd.DataFrame)


# ---------------------------------------------------------------------------
# Rust full-loop is NOT used in multi-strategy mode
# ---------------------------------------------------------------------------

class TestRustGate:
    def test_multi_strategy_uses_python_dispatch(self):
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=1.0, rebalance_freq=1
        )
        engine.run()
        assert engine.run_metadata["dispatch_mode"] == "python-multi"


# ---------------------------------------------------------------------------
# Comparison: multi-strategy with single slot vs single-strategy
# ---------------------------------------------------------------------------

class TestSingleSlotEquivalence:
    def test_single_slot_produces_balance(self):
        """A single add_strategy() call should produce a valid balance."""
        engine = _make_engine()
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=1.0, rebalance_freq=1
        )
        engine.run()
        assert not engine.balance.empty
        assert engine.balance["total capital"].iloc[-1] > 0


# ---------------------------------------------------------------------------
# options_budget compatibility
# ---------------------------------------------------------------------------

class TestOptionsBudget:
    def test_fixed_options_budget(self):
        engine = _make_engine()
        engine.options_budget = 5000.0
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="a"
        )
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="b"
        )
        engine.run()
        assert not engine.balance.empty

    def test_callable_options_budget(self):
        engine = _make_engine()
        engine.options_budget = lambda date, capital: 3000.0
        schema = engine.options_data.schema
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="a"
        )
        engine.add_strategy(
            _buy_put_strategy(schema), weight=0.5, rebalance_freq=1, name="b"
        )
        engine.run()
        assert not engine.balance.empty
