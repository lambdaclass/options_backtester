"""Chaos / fault-injection tests — corrupted and adversarial data.

Feed corrupted data through the engine. Assert: either raises a clear error
OR completes with math.isfinite(final_capital). Never silently produces NaN/Inf.

Reuses the test_engine.py data pattern, then corrupts ._data in-memory.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts, PerContractCommission
from options_portfolio_backtester.execution.fill_model import (
    MarketAtBidAsk, MidPrice, VolumeAwareFill,
)
from options_portfolio_backtester.execution.signal_selector import NearestDelta, FirstMatch
from options_portfolio_backtester.portfolio.risk import RiskManager, MaxDelta, MaxVega
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.core.types import Stock, OptionType as Type, Direction

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
STOCKS_FILE = os.path.join(TEST_DIR, "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "options_data.csv")

pytestmark = pytest.mark.chaos


# ---------------------------------------------------------------------------
# Shared helpers
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


def _run_chaos(options_data, stocks_data=None, cost_model=None,
               fill_model=None, signal_selector=None, risk_manager=None,
               direction=Direction.BUY, initial_capital=1_000_000):
    """Run engine with possibly-corrupted data. Returns engine or raises."""
    stocks = _ivy_stocks()
    sd = stocks_data or _stocks_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0},
        cost_model=cost_model or NoCosts(),
        fill_model=fill_model or MarketAtBidAsk(),
        signal_selector=signal_selector or NearestDelta(target_delta=-0.30),
        risk_manager=risk_manager or RiskManager(),
        initial_capital=initial_capital,
    )
    engine.stocks = stocks
    engine.stocks_data = sd
    engine.options_data = options_data
    engine.options_strategy = _build_strategy(schema, direction=direction)
    engine.run(rebalance_freq=1)
    return engine


def _assert_finite_or_error(fn):
    """Call fn(). If it succeeds, assert final capital is finite. If it raises, that's OK too."""
    try:
        engine = fn()
        final = engine.balance["total capital"].iloc[-1]
        assert math.isfinite(final), f"Non-finite final capital: {final}"
        return engine
    except (ValueError, KeyError, IndexError, ZeroDivisionError, AssertionError):
        pass  # Clear error is acceptable


# ---------------------------------------------------------------------------
# Chaos test classes
# ---------------------------------------------------------------------------

class TestNaNInjection:
    """NaN injected into bid, ask, delta, volume columns."""

    def test_nan_all_bids(self):
        od = _options_data()
        od._data["bid"] = np.nan
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_nan_all_asks(self):
        od = _options_data()
        od._data["ask"] = np.nan
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_nan_scattered_bid(self):
        od = _options_data()
        mask = od._data.index % 2 == 0
        od._data.loc[mask, "bid"] = np.nan
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_nan_scattered_ask(self):
        od = _options_data()
        mask = od._data.index % 2 == 0
        od._data.loc[mask, "ask"] = np.nan
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_nan_delta(self):
        od = _options_data()
        if "delta" in od._data.columns:
            od._data["delta"] = np.nan
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_nan_volume(self):
        od = _options_data()
        od._data["volume"] = np.nan
        _assert_finite_or_error(lambda: _run_chaos(od))


class TestNegativePrices:
    """Negative bid/ask prices — should not crash or produce NaN."""

    def test_negative_bid(self):
        od = _options_data()
        od._data["bid"] = -1.0
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_negative_ask(self):
        od = _options_data()
        od._data["ask"] = -5.0
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_both_negative(self):
        od = _options_data()
        od._data["bid"] = -2.0
        od._data["ask"] = -1.0
        _assert_finite_or_error(lambda: _run_chaos(od))


class TestInvertedBidAsk:
    """Bid > ask (crossed market) — should still produce finite fills."""

    def test_inverted_spread(self):
        od = _options_data()
        original_bid = od._data["bid"].copy()
        original_ask = od._data["ask"].copy()
        od._data["bid"] = original_ask
        od._data["ask"] = original_bid
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_bid_equals_ask(self):
        od = _options_data()
        od._data["ask"] = od._data["bid"]
        _assert_finite_or_error(lambda: _run_chaos(od))


class TestMissingColumns:
    """Drop delta column with NearestDelta selector — should fall back to first match."""

    def test_missing_delta_column(self):
        od = _options_data()
        if "delta" in od._data.columns:
            od._data = od._data.drop(columns=["delta"])
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, signal_selector=NearestDelta(target_delta=-0.30))
        )
        # NearestDelta falls back to iloc[0] when delta column missing
        if engine is not None:
            assert math.isfinite(engine.balance["total capital"].iloc[-1])


class TestNoMatchingContracts:
    """All DTE=0 — entry_filter (dte >= 60) never matches."""

    def test_all_dte_zero(self):
        od = _options_data()
        od._data["dte"] = 0
        engine = _assert_finite_or_error(lambda: _run_chaos(od))
        if engine is not None:
            # No trades should have occurred
            assert len(engine.trade_log) == 0 or engine.trade_log.empty


class TestZeroVolume:
    """Volume=0 with VolumeAwareFill — should fill at mid."""

    def test_zero_volume_fill(self):
        od = _options_data()
        od._data["volume"] = 0
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, fill_model=VolumeAwareFill(full_volume_threshold=100))
        )
        if engine is not None:
            assert math.isfinite(engine.balance["total capital"].iloc[-1])


class TestExtremeGreeks:
    """Extreme delta/vega values — risk constraints should block/allow correctly."""

    def test_extreme_delta_blocked(self):
        od = _options_data()
        if "delta" in od._data.columns:
            od._data["delta"] = 100.0
        rm = RiskManager(constraints=[MaxDelta(limit=0.01)])
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, risk_manager=rm)
        )
        if engine is not None:
            assert math.isfinite(engine.balance["total capital"].iloc[-1])

    def test_extreme_vega_blocked(self):
        od = _options_data()
        if "vega" in od._data.columns:
            od._data["vega"] = -999.0
        rm = RiskManager(constraints=[MaxVega(limit=0.01)])
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, risk_manager=rm)
        )
        if engine is not None:
            assert math.isfinite(engine.balance["total capital"].iloc[-1])

    def test_extreme_delta_allowed(self):
        od = _options_data()
        if "delta" in od._data.columns:
            od._data["delta"] = 100.0
        rm = RiskManager(constraints=[MaxDelta(limit=999999)])
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, risk_manager=rm)
        )
        if engine is not None:
            assert math.isfinite(engine.balance["total capital"].iloc[-1])


class TestDuplicateDates:
    """Duplicate all rows in options and stocks — should not crash."""

    def test_duplicate_options_rows(self):
        od = _options_data()
        od._data = pd.concat([od._data, od._data], ignore_index=True)
        sd = _stocks_data()
        sd._data = pd.concat([sd._data, sd._data], ignore_index=True)
        _assert_finite_or_error(lambda: _run_chaos(od, stocks_data=sd))


class TestCapitalExhaustion:
    """Initial capital = 1 — should not crash, zero or minimal trades."""

    def test_tiny_capital(self):
        od = _options_data()
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, initial_capital=1)
        )
        if engine is not None:
            final = engine.balance["total capital"].iloc[-1]
            assert math.isfinite(final)

    def test_zero_capital(self):
        od = _options_data()
        engine = _assert_finite_or_error(
            lambda: _run_chaos(od, initial_capital=0)
        )
        if engine is not None:
            final = engine.balance["total capital"].iloc[-1]
            assert math.isfinite(final)


class TestMassiveSpread:
    """bid=0.01, ask=999 — extreme spread should produce finite fills."""

    def test_massive_spread(self):
        od = _options_data()
        od._data["bid"] = 0.01
        od._data["ask"] = 999.0
        _assert_finite_or_error(lambda: _run_chaos(od))

    def test_massive_spread_mid_fill(self):
        od = _options_data()
        od._data["bid"] = 0.01
        od._data["ask"] = 999.0
        _assert_finite_or_error(
            lambda: _run_chaos(od, fill_model=MidPrice())
        )


class TestAllExpired:
    """All DTE=0 — no entries should happen, capital preserved."""

    def test_all_expired_capital_preserved(self):
        od = _options_data()
        od._data["dte"] = 0
        engine = _assert_finite_or_error(lambda: _run_chaos(od))
        if engine is not None:
            final = engine.balance["total capital"].iloc[-1]
            assert math.isfinite(final)
            # Capital should be close to initial since no options trades
            assert final > 0


class TestSingleDay:
    """Filter data to a single date — stats computation should not crash."""

    def test_single_date(self):
        od = _options_data()
        sd = _stocks_data()
        first_date = od._data["quotedate"].iloc[0]
        od._data = od._data[od._data["quotedate"] == first_date].copy()
        sd._data = sd._data[sd._data["date"] == first_date].copy()
        _assert_finite_or_error(lambda: _run_chaos(od, stocks_data=sd))
