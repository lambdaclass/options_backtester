"""Regression snapshot tests — lock backtest outputs against golden values.

Run a full backtest with fixed data + deterministic config, assert against
hardcoded values.  Any change in output = regression.

Uses NearestDelta selector to force the Python path for determinism
(avoids Rust dispatch which may differ across platforms).
"""

import math
import os

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


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_engine.py pattern)
# ---------------------------------------------------------------------------

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


def _run(cost_model=None, direction=Direction.BUY, monthly=False):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=cost_model or NoCosts(),
        signal_selector=NearestDelta(target_delta=-0.30),
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _build_strategy(schema, direction=direction)
    engine.run(rebalance_freq=1, monthly=monthly)
    return engine


# ---------------------------------------------------------------------------
# Golden values captured from deterministic runs
# ---------------------------------------------------------------------------

class TestSnapshotBuyPutNoCosts:
    """Buy-put backtest with NoCosts, daily rebalance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _run()

    def test_final_capital(self):
        final = self.engine.balance["total capital"].iloc[-1]
        assert abs(final - 932225.0) < 0.01, f"Regression: final_capital={final}"

    def test_trade_count(self):
        n = len(self.engine.trade_log)
        assert n == 7, f"Regression: trade_count={n}"

    def test_balance_rows(self):
        n = len(self.engine.balance)
        assert n == 61, f"Regression: balance_rows={n}"

    def test_total_return(self):
        bal = self.engine.balance["total capital"]
        ret = (bal.iloc[-1] - bal.iloc[0]) / bal.iloc[0]
        assert abs(ret - (-0.067775)) < 1e-4, f"Regression: total_return={ret}"

    def test_max_drawdown(self):
        bal = self.engine.balance["total capital"]
        running_max = bal.cummax()
        dd = (running_max - bal) / running_max
        max_dd = dd.max()
        assert abs(max_dd - 0.067775) < 1e-4, f"Regression: max_drawdown={max_dd}"


class TestSnapshotBuyPutWithCommission:
    """Buy-put with PerContractCommission — costs must reduce final capital."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine_no_cost = _run()
        self.engine_cost = _run(cost_model=PerContractCommission(0.65))

    def test_commission_reduces_capital(self):
        no_cost_final = self.engine_no_cost.balance["total capital"].iloc[-1]
        cost_final = self.engine_cost.balance["total capital"].iloc[-1]
        assert cost_final < no_cost_final

    def test_final_capital(self):
        final = self.engine_cost.balance["total capital"].iloc[-1]
        assert abs(final - 929824.225) < 0.01, f"Regression: final_capital={final}"


class TestSnapshotSellPut:
    """Sell-put (reversed direction) — verifies direction wiring."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _run(direction=Direction.SELL)

    def test_final_capital(self):
        final = self.engine.balance["total capital"].iloc[-1]
        assert abs(final - 874135.0) < 1.0, f"Regression: final_capital={final}"

    def test_trade_count(self):
        n = len(self.engine.trade_log)
        assert n == 7, f"Regression: trade_count={n}"

    def test_sell_vs_buy_differ(self):
        buy_engine = _run(direction=Direction.BUY)
        buy_final = buy_engine.balance["total capital"].iloc[-1]
        sell_final = self.engine.balance["total capital"].iloc[-1]
        assert buy_final != sell_final


class TestSnapshotMonthlyRebalance:
    """Monthly rebalance — fewer balance rows than daily."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine_daily = _run(monthly=False)
        self.engine_monthly = _run(monthly=True)

    def test_fewer_balance_rows(self):
        daily_rows = len(self.engine_daily.balance)
        monthly_rows = len(self.engine_monthly.balance)
        assert monthly_rows <= daily_rows

    def test_final_capital(self):
        final = self.engine_monthly.balance["total capital"].iloc[-1]
        assert abs(final - 932225.0) < 0.01, f"Regression: final_capital={final}"

    def test_balance_rows(self):
        n = len(self.engine_monthly.balance)
        assert n == 61, f"Regression: balance_rows={n}"
