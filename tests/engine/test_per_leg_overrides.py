"""Tests for per-leg signal_selector and fill_model overrides."""

import os
import numpy as np
import pandas as pd
import pytest

from options_backtester.engine.engine import BacktestEngine
from options_backtester.execution.cost_model import NoCosts
from options_backtester.execution.fill_model import MarketAtBidAsk, MidPrice
from options_backtester.execution.signal_selector import FirstMatch, NearestDelta, SignalSelector

from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy
from backtester.enums import Stock, Type, Direction

# Use new StrategyLeg that supports signal_selector/fill_model
from options_backtester.strategy.strategy_leg import StrategyLeg

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


class TestPerLegSignalSelector:
    """Verify per-leg signal selector overrides the engine-level one."""

    def test_leg_selector_is_used(self):
        """A per-leg selector should be called instead of the engine-level one."""
        call_count = 0

        class CountingSelector(SignalSelector):
            def select(self, candidates):
                nonlocal call_count
                call_count += 1
                return candidates.iloc[0]

        options_data = _options_data()
        schema = options_data.schema

        # Create leg with per-leg selector
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                          direction=Direction.BUY, signal_selector=CountingSelector())
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30

        strat = Strategy(schema)
        strat.add_legs([leg])

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=FirstMatch(),  # engine-level, should be ignored
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)

        assert call_count > 0, "Per-leg signal selector was never called"

    def test_engine_selector_used_when_leg_has_none(self):
        """When leg has no signal_selector attr (legacy leg), engine-level is used."""
        from backtester.strategy.strategy_leg import StrategyLeg as LegacyLeg

        call_count = 0

        class CountingSelector(SignalSelector):
            def select(self, candidates):
                nonlocal call_count
                call_count += 1
                return candidates.iloc[0]

        options_data = _options_data()
        schema = options_data.schema

        leg = LegacyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30

        strat = Strategy(schema)
        strat.add_legs([leg])

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            signal_selector=CountingSelector(),
        )
        engine.stocks = _ivy_stocks()
        engine.stocks_data = _stocks_data()
        engine.options_data = options_data
        engine.options_strategy = strat
        engine.run(rebalance_freq=1)

        assert call_count > 0, "Engine-level selector was never called for legacy leg"


class TestPerLegFillModel:
    """Verify per-leg fill model overrides the engine-level one."""

    def test_midprice_differs_from_market(self):
        """MidPrice fill model should produce different costs than MarketAtBidAsk."""
        options_data = _options_data()
        schema = options_data.schema

        # Run with default MarketAtBidAsk
        leg_market = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                                 direction=Direction.BUY, fill_model=MarketAtBidAsk())
        leg_market.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
        leg_market.exit_filter = schema.dte <= 30
        strat_market = Strategy(schema)
        strat_market.add_legs([leg_market])

        engine_market = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine_market.stocks = _ivy_stocks()
        engine_market.stocks_data = _stocks_data()
        engine_market.options_data = _options_data()
        engine_market.options_strategy = strat_market
        engine_market.run(rebalance_freq=1)

        # Run with MidPrice fill model
        options_data2 = _options_data()
        schema2 = options_data2.schema
        leg_mid = StrategyLeg("leg_1", schema2, option_type=Type.PUT,
                              direction=Direction.BUY, fill_model=MidPrice())
        leg_mid.entry_filter = (schema2.underlying == "SPX") & (schema2.dte >= 60)
        leg_mid.exit_filter = schema2.dte <= 30
        strat_mid = Strategy(schema2)
        strat_mid.add_legs([leg_mid])

        engine_mid = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine_mid.stocks = _ivy_stocks()
        engine_mid.stocks_data = _stocks_data()
        engine_mid.options_data = options_data2
        engine_mid.options_strategy = strat_mid
        engine_mid.run(rebalance_freq=1)

        # Both should have trades
        assert not engine_market.trade_log.empty
        assert not engine_mid.trade_log.empty

        # Costs should differ because MidPrice uses (bid+ask)/2 instead of ask
        market_costs = engine_market.trade_log["leg_1"]["cost"].values
        mid_costs = engine_mid.trade_log["leg_1"]["cost"].values

        if len(market_costs) > 0 and len(mid_costs) > 0:
            # MidPrice for BUY should be cheaper than MarketAtBidAsk (which uses ask)
            # Different fill models may produce different numbers of trades
            if len(market_costs) != len(mid_costs):
                pass  # Different lengths means different results
            else:
                assert not np.allclose(market_costs, mid_costs, rtol=1e-6), \
                    "MidPrice should produce different costs than MarketAtBidAsk"
