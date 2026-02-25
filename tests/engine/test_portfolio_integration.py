"""Tests verifying Portfolio dataclass is kept in sync with legacy MultiIndex inventory."""

import os
import pytest
import numpy as np

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.portfolio.portfolio import Portfolio
from options_portfolio_backtester.portfolio.position import OptionPosition

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


def _run_engine():
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _buy_strategy(schema)
    engine.run(rebalance_freq=1)
    return engine


class TestPortfolioIntegration:
    """Verify _portfolio dataclass is maintained alongside legacy DataFrames."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _run_engine()

    def test_portfolio_exists(self):
        assert hasattr(self.engine, '_portfolio')
        assert isinstance(self.engine._portfolio, Portfolio)

    def test_portfolio_position_count_matches_inventory(self):
        """After backtest, portfolio positions should match remaining inventory rows."""
        inv_count = len(self.engine._options_inventory)
        port_count = len(self.engine._portfolio.option_positions)
        assert inv_count == port_count, (
            f"Inventory has {inv_count} rows but Portfolio has {port_count} positions"
        )

    def test_positions_have_correct_legs(self):
        """Each position should have legs matching strategy leg names."""
        for pid, pos in self.engine._portfolio.option_positions.items():
            assert isinstance(pos, OptionPosition)
            assert len(pos.legs) > 0
            for leg_name in pos.legs:
                assert leg_name.startswith("leg_")

    def test_trade_log_not_empty(self):
        """Backtest should produce trades."""
        assert not self.engine.trade_log.empty

    def test_portfolio_contracts_match_inventory(self):
        """Portfolio contract IDs should match inventory contract IDs."""
        for idx, inv_row in self.engine._options_inventory.iterrows():
            if idx in self.engine._portfolio.option_positions:
                pos = self.engine._portfolio.option_positions[idx]
                for leg in self.engine._options_strategy.legs:
                    inv_contract = inv_row[leg.name]["contract"]
                    pos_contract = pos.legs[leg.name].contract_id
                    assert inv_contract == pos_contract, (
                        f"Contract mismatch at {idx}/{leg.name}: "
                        f"inventory={inv_contract}, portfolio={pos_contract}"
                    )
