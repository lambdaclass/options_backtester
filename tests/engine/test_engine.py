"""Tests for BacktestEngine — verifies it produces identical results to the original Backtest."""

import os
import math
import pytest
import numpy as np

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts, PerContractCommission
from options_portfolio_backtester.execution.signal_selector import FirstMatch
from options_portfolio_backtester.portfolio.risk import RiskManager

from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Stock, Type, Direction
from backtester import Backtest as OriginalBacktest

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
    """Create test options data with known bid/ask values (same as conftest.options_data_2puts_buy)."""
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


def _run_original():
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    bt = OriginalBacktest({"stocks": 0.97, "options": 0.03, "cash": 0})
    bt.stocks = stocks
    bt.stocks_data = stocks_data
    bt.options_data = options_data
    bt.options_strategy = _buy_strategy(schema)
    bt.run(rebalance_freq=1)
    return bt


def _run_engine(cost_model=None):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=cost_model or NoCosts(),
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _buy_strategy(schema)
    engine.run(rebalance_freq=1)
    return engine


class TestEngineMatchesOriginal:
    """The new engine with NoCosts must produce identical results to the original."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.original = _run_original()
        self.engine = _run_engine()

    def test_trade_log_same_shape(self):
        assert self.original.trade_log.shape == self.engine.trade_log.shape

    def test_balance_same_shape(self):
        assert self.original.balance.shape == self.engine.balance.shape

    def test_final_capital_matches(self):
        orig_final = self.original.balance["total capital"].iloc[-1]
        engine_final = self.engine.balance["total capital"].iloc[-1]
        assert abs(orig_final - engine_final) < 0.01, f"{orig_final} != {engine_final}"

    def test_cash_matches(self):
        assert abs(self.original.current_cash - self.engine.current_cash) < 0.01

    def test_trade_log_costs_match(self):
        orig_costs = self.original.trade_log["totals"]["cost"].values
        engine_costs = self.engine.trade_log["totals"]["cost"].values
        assert np.allclose(orig_costs, engine_costs, rtol=1e-4)

    def test_regression_values(self):
        """Same regression assertions as the original test suite."""
        tol = 0.0001
        bt = self.engine
        assert np.allclose(bt.trade_log["totals"]["cost"].values, [100, 150], rtol=tol)
        assert np.allclose(bt.trade_log["leg_1"]["cost"].values, [100, 150], rtol=tol)
        assert np.allclose(
            bt.trade_log["totals"]["qty"].values,
            [300, (((97 + 3 * 0.5) * 0.03 - 1.5) / 1.5) * 100],
            rtol=tol,
        )


class TestEngineWithCosts:
    """Test that adding costs changes the result (proves costs are wired in)."""

    def test_commission_reduces_final_capital(self):
        no_cost = _run_engine()
        with_cost = _run_engine(cost_model=PerContractCommission(rate=5.00, stock_rate=0.01))

        no_cost_final = no_cost.balance["total capital"].iloc[-1]
        with_cost_final = with_cost.balance["total capital"].iloc[-1]
        assert with_cost_final < no_cost_final


class TestRunMetadata:
    """Ensure reproducibility metadata is attached to outputs."""

    def test_metadata_attached_to_trade_log_and_balance(self):
        engine = _run_engine()
        meta = engine.run_metadata

        assert meta["framework"] == "options_portfolio_backtester.engine.BacktestEngine"
        assert meta["dispatch_mode"] in {"python", "rust-full"}
        assert isinstance(meta["git_sha"], str)
        assert len(meta["config_hash"]) == 64
        assert len(meta["data_snapshot_hash"]) == 64
        assert meta["data_snapshot"]["options_rows"] > 0
        assert meta["data_snapshot"]["stocks_rows"] > 0
        assert engine.trade_log.attrs["run_metadata"] == meta
        assert engine.balance.attrs["run_metadata"] == meta


class TestEngineInit:
    """Test engine initialization without running backtests."""

    def test_default_allocation_normalized(self):
        e = BacktestEngine({"stocks": 60, "options": 30, "cash": 10})
        assert abs(e.allocation["stocks"] - 0.6) < 1e-10
        assert abs(e.allocation["options"] - 0.3) < 1e-10
        assert abs(e.allocation["cash"] - 0.1) < 1e-10

    def test_default_components(self):
        e = BacktestEngine({"stocks": 1.0})
        assert isinstance(e.cost_model, NoCosts)
        assert isinstance(e.signal_selector, FirstMatch)
        assert isinstance(e.risk_manager, RiskManager)
        assert e.stop_if_broke is False

    def test_stop_if_broke_flag(self):
        e = BacktestEngine({"stocks": 1.0}, stop_if_broke=True)
        assert e.stop_if_broke is True
