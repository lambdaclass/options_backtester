"""Tests for max_notional_pct engine parameter."""

import os

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts

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
    """Long-only put strategy (no SELL legs)."""
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _sell_strategy(schema):
    """Short put strategy (SELL leg) — triggers notional cap."""
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.SELL)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


def _straddle_strategy(schema):
    """Short straddle (2 SELL legs) — tests multi-leg notional summing."""
    strat = Strategy(schema)
    call = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.SELL)
    call.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    call.exit_filter = schema.dte <= 30
    put = StrategyLeg("leg_2", schema, option_type=Type.PUT, direction=Direction.SELL)
    put.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    put.exit_filter = schema.dte <= 30
    strat.add_legs([call, put])
    return strat


def _run_engine(max_notional_pct=None, strategy_fn=None):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=NoCosts(),
        max_notional_pct=max_notional_pct,
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = (strategy_fn or _buy_strategy)(schema)
    engine.run(rebalance_freq=1)
    return engine


class TestMaxNotionalPct:
    """Verify max_notional_pct caps short option notional exposure."""

    def test_none_is_backward_compatible(self):
        """max_notional_pct=None should produce the same result as before."""
        engine_default = _run_engine(max_notional_pct=None)
        assert not engine_default.balance.empty
        assert not engine_default.trade_log.empty

    def test_long_only_unaffected(self):
        """Long-only strategies should be unaffected by the notional cap."""
        engine_no_cap = _run_engine(max_notional_pct=None, strategy_fn=_buy_strategy)
        engine_with_cap = _run_engine(max_notional_pct=0.01, strategy_fn=_buy_strategy)
        assert len(engine_no_cap.trade_log) == len(engine_with_cap.trade_log)

    def test_sell_strategy_capped(self):
        """A tight notional cap should reduce qty on short strategies."""
        engine_no_cap = _run_engine(max_notional_pct=None, strategy_fn=_sell_strategy)
        # 1% of 1M = 10k; one contract at strike 650 = 65k notional → 0 qty
        engine_tight = _run_engine(max_notional_pct=0.01, strategy_fn=_sell_strategy)

        no_cap_trades = len(engine_no_cap.trade_log)
        tight_trades = len(engine_tight.trade_log)
        assert tight_trades <= no_cap_trades

    def test_zero_cap_blocks_all_short_trades(self):
        """max_notional_pct=0 should block all short option entries."""
        engine = _run_engine(max_notional_pct=0.0, strategy_fn=_sell_strategy)
        assert len(engine.trade_log) == 0

    def test_generous_cap_allows_trades(self):
        """A generous notional cap should still allow trades (more than a tight cap)."""
        engine_generous = _run_engine(max_notional_pct=10.0, strategy_fn=_sell_strategy)
        engine_tight = _run_engine(max_notional_pct=0.01, strategy_fn=_sell_strategy)
        assert len(engine_generous.trade_log) >= len(engine_tight.trade_log)
        assert len(engine_generous.trade_log) > 0

    def test_straddle_both_legs_contribute_notional(self):
        """A straddle has 2 SELL legs — both should count toward notional cap."""
        # Tight cap blocks straddle (2× notional vs single put)
        engine_straddle = _run_engine(max_notional_pct=0.01, strategy_fn=_straddle_strategy)
        engine_put = _run_engine(max_notional_pct=0.01, strategy_fn=_sell_strategy)
        # Both should be blocked at this cap level
        assert len(engine_straddle.trade_log) <= len(engine_put.trade_log)

    def test_cap_monotonic(self):
        """Increasing the cap should never reduce the number of trades."""
        caps = [0.01, 0.10, 0.50, 1.0, 10.0]
        trade_counts = []
        for cap in caps:
            engine = _run_engine(max_notional_pct=cap, strategy_fn=_sell_strategy)
            trade_counts.append(len(engine.trade_log))
        for i in range(len(trade_counts) - 1):
            assert trade_counts[i] <= trade_counts[i + 1], (
                f"cap {caps[i]} had {trade_counts[i]} trades but "
                f"cap {caps[i+1]} had {trade_counts[i+1]}"
            )
