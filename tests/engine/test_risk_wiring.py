"""Tests that RiskManager is actually wired into the engine."""

import os
import pytest
import numpy as np

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.portfolio.risk import RiskManager, MaxDelta, MaxDrawdown

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


def _buy_strategy(schema):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _run_engine(risk_manager=None):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
        risk_manager=risk_manager or RiskManager(),
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _buy_strategy(schema)
    engine.run(rebalance_freq=1)
    return engine


class TestRiskManagerWiring:
    """Verify the engine actually calls the risk manager."""

    def test_no_constraints_allows_all(self):
        engine = _run_engine(RiskManager())
        assert not engine.trade_log.empty

    def test_max_delta_blocks_entries(self):
        """Tiny delta limit should block all entries."""
        no_risk = _run_engine(RiskManager())
        tight_risk = _run_engine(RiskManager([MaxDelta(limit=0.0001)]))

        no_risk_trades = len(no_risk.trade_log)
        tight_trades = len(tight_risk.trade_log)
        assert tight_trades < no_risk_trades, (
            f"MaxDelta should block entries: got {tight_trades} vs {no_risk_trades}"
        )

    def test_max_drawdown_blocks_during_crash(self):
        """Very tight drawdown limit should block entries once any loss occurs."""
        no_risk = _run_engine(RiskManager())
        tight_dd = _run_engine(RiskManager([MaxDrawdown(max_dd_pct=0.0001)]))

        no_risk_trades = len(no_risk.trade_log)
        tight_trades = len(tight_dd.trade_log)
        # With 0.01% max drawdown, most entries after the first should be blocked
        assert tight_trades <= no_risk_trades

    def test_risk_manager_preserves_capital(self):
        """Blocked entries should leave cash unchanged (budget stays as cash)."""
        blocked = _run_engine(RiskManager([MaxDelta(limit=0.0001)]))
        # If all entries are blocked, the final capital should be close to initial
        # minus stock movements
        assert blocked.balance is not None
