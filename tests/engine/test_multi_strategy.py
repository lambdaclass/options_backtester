"""Tests for MultiStrategyEngine."""

import os
import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.engine.multi_strategy import (
    StrategyAllocation, MultiStrategyEngine,
)
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.data.providers import (
    TiingoData, HistoricalOptionsData,
)
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.core.types import Direction, OptionType, Stock


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _make_engine(data_dir):
    stocks_path = os.path.join(data_dir, "test_stocks.csv")
    options_path = os.path.join(data_dir, "test_options.csv")
    if not os.path.exists(stocks_path) or not os.path.exists(options_path):
        pytest.skip("test data files not available")

    stocks_data = TiingoData(stocks_path)
    options_data = HistoricalOptionsData(options_path)
    schema = options_data.schema

    strategy = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=OptionType.CALL, direction=Direction.SELL)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= 30)
        & (schema.dte <= 60)
    )
    leg.exit_filter = schema.dte <= 7
    strategy.add_leg(leg)

    engine = BacktestEngine(
        allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
        initial_capital=500_000,
    )
    engine.stocks = [Stock("SPY", 1.0)]
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = strategy
    return engine


class TestStrategyAllocation:
    def test_fields(self):
        engine = BacktestEngine(allocation={"stocks": 1.0})
        sa = StrategyAllocation(name="test", engine=engine, weight=0.5)
        assert sa.name == "test"
        assert sa.weight == 0.5
        assert sa.engine is engine


class TestMultiStrategyEngine:
    def test_weight_normalization(self):
        e1 = BacktestEngine(allocation={"stocks": 1.0})
        e2 = BacktestEngine(allocation={"stocks": 1.0})
        mse = MultiStrategyEngine(
            strategies=[
                StrategyAllocation("a", e1, weight=3.0),
                StrategyAllocation("b", e2, weight=1.0),
            ],
            initial_capital=1_000_000,
        )
        assert abs(mse._weights["a"] - 0.75) < 1e-10
        assert abs(mse._weights["b"] - 0.25) < 1e-10

    def test_equal_weights(self):
        engines = [BacktestEngine(allocation={"stocks": 1.0}) for _ in range(3)]
        mse = MultiStrategyEngine(
            strategies=[
                StrategyAllocation(f"s{i}", e) for i, e in enumerate(engines)
            ],
        )
        for w in mse._weights.values():
            assert abs(w - 1.0 / 3) < 1e-10

    def test_run_with_mocked_engines(self):
        """Test run() and _build_combined_balance() without real data."""
        dates = pd.bdate_range("2020-01-01", periods=5)

        class FakeEngine:
            def __init__(self):
                self.initial_capital = 100_000
                self.balance = pd.DataFrame({
                    "total capital": [100000, 101000, 102000, 101500, 103000],
                    "% change": [0.0, 0.01, 0.0099, -0.0049, 0.0148],
                }, index=dates)

            def run(self, **kwargs):
                return pd.DataFrame()  # empty trade log

        e1, e2 = FakeEngine(), FakeEngine()
        mse = MultiStrategyEngine(
            strategies=[
                StrategyAllocation("a", e1, weight=0.6),
                StrategyAllocation("b", e2, weight=0.4),
            ],
            initial_capital=1_000_000,
        )
        results = mse.run(rebalance_freq=1)
        assert "a" in results
        assert "b" in results
        assert hasattr(mse, "balance")
        assert "total capital" in mse.balance.columns
        assert "% change" in mse.balance.columns
        assert "accumulated return" in mse.balance.columns
        assert len(mse.balance) == 5
        # Capital share should be updated
        assert e1.initial_capital == 600_000
        assert e2.initial_capital == 400_000

    def test_run_engine_without_balance(self):
        """Engines that don't produce a balance still work."""
        class NoBalanceEngine:
            def __init__(self):
                self.initial_capital = 0
            def run(self, **kwargs):
                return pd.DataFrame()

        e1 = NoBalanceEngine()
        mse = MultiStrategyEngine(
            strategies=[StrategyAllocation("x", e1, weight=1.0)],
            initial_capital=500_000,
        )
        results = mse.run()
        assert "x" in results
        assert mse.balance.empty

    def test_run_with_data(self, data_dir):
        e1 = _make_engine(data_dir)
        e2 = _make_engine(data_dir)
        mse = MultiStrategyEngine(
            strategies=[
                StrategyAllocation("strat_a", e1, weight=0.6),
                StrategyAllocation("strat_b", e2, weight=0.4),
            ],
            initial_capital=1_000_000,
        )
        results = mse.run(rebalance_freq=1)
        assert "strat_a" in results
        assert "strat_b" in results
        assert hasattr(mse, "balance")
        assert "total capital" in mse.balance.columns
        assert "% change" in mse.balance.columns
        assert "accumulated return" in mse.balance.columns
