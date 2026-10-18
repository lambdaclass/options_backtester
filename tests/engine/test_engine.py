"""Tests for BacktestEngine — verifies regression values and engine behavior."""

import os

import numpy as np
import pytest

from options_portfolio_backtester.core.types import Direction, Stock
from options_portfolio_backtester.core.types import OptionType as Type
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts, PerContractCommission
from options_portfolio_backtester.execution.signal_selector import FirstMatch
from options_portfolio_backtester.portfolio.risk import RiskManager
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
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


class TestEngineRegressionValues:
    """Verify the engine produces known regression values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = _run_engine()

    def test_trade_log_not_empty(self):
        assert not self.engine.trade_log.empty

    def test_balance_not_empty(self):
        assert not self.engine.balance.empty

    def test_regression_costs(self):
        tol = 0.0001
        bt = self.engine
        # Positions persist across rebalances — only new entries, no liquidation churn.
        assert np.allclose(bt.trade_log["totals"]["cost"].values,
                           [100, 150], rtol=tol)
        assert np.allclose(bt.trade_log["leg_1"]["cost"].values,
                           [100, 150], rtol=tol)

    def test_regression_qtys(self):
        tol = 0.0001
        bt = self.engine
        assert np.allclose(
            bt.trade_log["totals"]["qty"].values,
            [300, 97],
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


class TestRebalanceDatesOverride:
    """run(rebalance_dates=...) gates entry to a caller-supplied set of dates
    (used for signal-gated tail-hedge research). Verifies the override is
    honored: a single rebalance date yields far fewer entries than the
    default per-period schedule, and entries occur only on/after it."""

    def _engine(self):
        opts = _options_data()
        eng = BacktestEngine({"stocks": 0.97, "options": 0.03, "cash": 0.0},
                             cost_model=NoCosts())
        eng.stocks = _ivy_stocks()
        eng.stocks_data = _stocks_data()
        eng.options_data = opts
        eng.options_strategy = _buy_strategy(opts.schema)
        return eng, opts

    def test_override_reduces_and_gates_entries(self):
        import pandas as pd
        eng_default, opts = self._engine()
        eng_default.run(rebalance_freq=1)
        default_entries = eng_default.trade_log[
            eng_default.trade_log[("leg_1", "order")] == "BTO"
        ]

        dates = sorted(pd.to_datetime(opts._data["quotedate"].unique()))
        one_date = [dates[len(dates) // 2]]
        eng_gated, _ = self._engine()
        eng_gated.run(rebalance_freq=1, rebalance_dates=one_date)
        gated_entries = eng_gated.trade_log[
            eng_gated.trade_log[("leg_1", "order")] == "BTO"
        ]

        # Gating to one date must not produce more entries than the full schedule,
        # and any entry must be on/after the single supplied rebalance date.
        assert len(gated_entries) <= len(default_entries)
        if not gated_entries.empty:
            entry_dates = pd.to_datetime(gated_entries[("totals", "date")])
            assert (entry_dates >= one_date[0]).all()


class TestHedgeFillDiagnostics:
    """The engine reports a hedge fill-rate so a requested option leg that the
    data cannot supply (e.g. a deep-OTM strike band absent from old chains) is
    surfaced instead of silently degrading the overlay to buy-and-hold."""

    def _engine(self, entry_filter_fn=None):
        opts = _options_data()
        eng = BacktestEngine({"stocks": 0.97, "options": 0.03, "cash": 0.0},
                             cost_model=NoCosts())
        eng.stocks = _ivy_stocks()
        eng.stocks_data = _stocks_data()
        eng.options_data = opts
        schema = opts.schema
        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (entry_filter_fn(schema) if entry_filter_fn
                            else (schema.underlying == "SPX") & (schema.dte >= 60))
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])
        eng.options_strategy = strat
        return eng

    def test_normal_run_fills_and_is_silent(self):
        import warnings

        from options_portfolio_backtester.engine.engine import HedgeFillWarning
        eng = self._engine()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eng.run(rebalance_freq=1)
            fired = [x for x in w if issubclass(x.category, HedgeFillWarning)]
        assert eng.option_entry_attempts >= 1
        assert eng.option_fill_rate >= 0.9   # a satisfiable filter fills
        assert not fired

    def test_unsatisfiable_filter_warns_and_reports_zero_fill(self):
        import warnings

        from options_portfolio_backtester.engine.engine import HedgeFillWarning
        # A strike far above any contract in the chain can never match, so every
        # rebalance attempts an entry and none fill.
        eng = self._engine(lambda s: (s.underlying == "SPX") & (s.strike > 1e9))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eng.run(rebalance_freq=1)
            fired = [x for x in w if issubclass(x.category, HedgeFillWarning)]
        assert eng.option_entry_attempts >= 1
        assert eng.option_entry_unfilled == eng.option_entry_attempts
        assert eng.option_fill_rate == 0.0
        assert fired, "expected a HedgeFillWarning when no contract can match"
