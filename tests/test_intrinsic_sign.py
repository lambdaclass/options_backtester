"""Tests that intrinsic-value fallback produces correct sign in _current_options_capital.

BUY legs are assets  → positive capital
SELL legs are liabilities → negative capital
OTM options (zero intrinsic) → zero capital
"""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.core.types import Direction, OptionType
from options_portfolio_backtester.data.schema import Schema
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg


def _make_engine_with_position(direction: Direction, option_type: OptionType,
                               strike: float, spot: float, qty: int = 1):
    """Build a minimal engine with one inventory position and NaN option quotes.

    Returns (engine, empty_options_df, stocks_df) ready for _current_options_capital().
    """
    opt_schema = Schema.options()
    stk_schema = Schema.stocks()

    leg = StrategyLeg("leg_1", opt_schema, option_type=option_type, direction=direction)
    strategy = Strategy(opt_schema)
    strategy.legs.append(leg)

    engine = BacktestEngine(
        allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
        initial_capital=1_000_000,
        shares_per_contract=100,
    )
    engine.options_strategy = strategy
    engine._stocks_schema = stk_schema._mappings
    engine._options_schema = opt_schema._mappings

    # Build a one-row inventory with the position
    inventory = pd.DataFrame({
        ("leg_1", "contract"): ["SPY_TEST_001"],
        ("leg_1", "type"): [option_type.value],
        ("leg_1", "strike"): [strike],
        ("leg_1", "underlying"): ["SPY"],
        ("totals", "qty"): [qty],
    })
    inventory.columns = pd.MultiIndex.from_tuples(inventory.columns)
    engine._options_inventory = inventory

    # Empty options frame → all quotes will be NaN after left-merge
    options = pd.DataFrame(columns=[
        "underlying", "underlying_last", "date", "contract",
        "type", "expiration", "strike", "bid", "ask",
        "volume", "open_interest",
    ])

    # Stock frame with known spot
    stocks = pd.DataFrame({"symbol": ["SPY"], "adjClose": [spot]})

    return engine, options, stocks


class TestBuyPutItmPositiveCapital:
    """BUY put ITM: the position is an asset → capital > 0."""

    def test_buy_put_itm_positive_capital(self):
        engine, options, stocks = _make_engine_with_position(
            Direction.BUY, OptionType.PUT, strike=400.0, spot=380.0,
        )
        capital = engine._current_options_capital(options, stocks)
        # intrinsic = 400 - 380 = 20, scaled by 100 spc, qty=1, BUY = asset
        assert capital > 0
        assert capital == pytest.approx(20.0 * 100)


class TestSellPutItmNegativeCapital:
    """SELL put ITM: the position is a liability → capital < 0."""

    def test_sell_put_itm_negative_capital(self):
        engine, options, stocks = _make_engine_with_position(
            Direction.SELL, OptionType.PUT, strike=400.0, spot=380.0,
        )
        capital = engine._current_options_capital(options, stocks)
        # intrinsic = 20 * 100, SELL = liability → negative
        assert capital < 0
        assert capital == pytest.approx(-20.0 * 100)


class TestBuyCallItmPositiveCapital:
    """BUY call ITM: asset → capital > 0."""

    def test_buy_call_itm_positive_capital(self):
        engine, options, stocks = _make_engine_with_position(
            Direction.BUY, OptionType.CALL, strike=380.0, spot=400.0,
        )
        capital = engine._current_options_capital(options, stocks)
        assert capital > 0
        assert capital == pytest.approx(20.0 * 100)


class TestSellCallItmNegativeCapital:
    """SELL call ITM: liability → capital < 0."""

    def test_sell_call_itm_negative_capital(self):
        engine, options, stocks = _make_engine_with_position(
            Direction.SELL, OptionType.CALL, strike=380.0, spot=400.0,
        )
        capital = engine._current_options_capital(options, stocks)
        assert capital < 0
        assert capital == pytest.approx(-20.0 * 100)


class TestOtmCapitalIsZero:
    """OTM options have zero intrinsic → capital = 0."""

    def test_otm_put(self):
        engine, options, stocks = _make_engine_with_position(
            Direction.SELL, OptionType.PUT, strike=380.0, spot=400.0,
        )
        capital = engine._current_options_capital(options, stocks)
        assert capital == pytest.approx(0.0)

    def test_otm_call(self):
        engine, options, stocks = _make_engine_with_position(
            Direction.BUY, OptionType.CALL, strike=400.0, spot=380.0,
        )
        capital = engine._current_options_capital(options, stocks)
        assert capital == pytest.approx(0.0)
