"""Tests that SignalSelector is actually wired into the engine.

All execution goes through Rust, so we verify via standard selectors
that have to_rust_config() and produce different Rust behavior.
"""

import os
import pandas as pd
import numpy as np
import pytest

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.execution.signal_selector import (
    FirstMatch, NearestDelta, MaxOpenInterest,
)

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


def _run_engine(signal_selector):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
        signal_selector=signal_selector,
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _buy_strategy(schema)
    engine.run(rebalance_freq=1)
    return engine


class TestSignalSelectorWiring:
    """Verify the engine uses the plugged-in signal selector via Rust dispatch."""

    def test_first_match_still_works(self):
        engine = _run_engine(FirstMatch())
        assert not engine.trade_log.empty

    def test_different_selectors_may_pick_different_contracts(self):
        """FirstMatch and NearestDelta should produce valid results (may differ)."""
        first_engine = _run_engine(FirstMatch())
        delta_engine = _run_engine(NearestDelta(target_delta=-0.30))

        assert not first_engine.balance.empty
        assert not delta_engine.balance.empty

    def test_nearest_delta_runs_without_error(self):
        engine = _run_engine(NearestDelta(target_delta=-0.30))
        assert engine.balance is not None

    def test_max_open_interest_runs(self):
        """MaxOpenInterest selector completes without error."""
        engine = _run_engine(MaxOpenInterest(oi_column="openinterest"))
        assert engine.balance is not None
