"""Tests for strategy preset constructors."""

import math

from options_portfolio_backtester.core.types import Direction, OptionType
from options_portfolio_backtester.data.schema import Schema
from options_portfolio_backtester.strategy.presets import (
    strangle, iron_condor, covered_call, cash_secured_put, collar, butterfly,
)


def _options_schema():
    s = Schema.options()
    s.update({
        "contract": "optionroot",
        "date": "quotedate",
        "dte": "dte",
        "last": "last",
        "open_interest": "openinterest",
        "impliedvol": "impliedvol",
        "delta": "delta",
        "gamma": "gamma",
        "theta": "theta",
        "vega": "vega",
    })
    return s


class TestStrangleFunction:
    def test_creates_two_legs(self):
        s = strangle(_options_schema(), "SPY", Direction.SELL,
                     dte_range=(30, 60), dte_exit=7)
        assert len(s.legs) == 2

    def test_leg_types(self):
        s = strangle(_options_schema(), "SPY", Direction.BUY,
                     dte_range=(30, 60), dte_exit=7)
        assert s.legs[0].type == OptionType.CALL
        assert s.legs[1].type == OptionType.PUT

    def test_leg_directions_match_input(self):
        s = strangle(_options_schema(), "SPY", Direction.SELL,
                     dte_range=(30, 60), dte_exit=7)
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[1].direction == Direction.SELL

    def test_exit_thresholds(self):
        s = strangle(_options_schema(), "SPY", Direction.SELL,
                     dte_range=(30, 60), dte_exit=7,
                     exit_thresholds=(0.5, 0.3))
        assert s.exit_thresholds == (0.5, 0.3)

    def test_default_exit_thresholds_are_inf(self):
        s = strangle(_options_schema(), "SPY", Direction.SELL,
                     dte_range=(30, 60), dte_exit=7)
        assert s.exit_thresholds == (math.inf, math.inf)


class TestIronCondor:
    def test_creates_four_legs(self):
        s = iron_condor(_options_schema(), "SPY",
                        dte_range=(30, 60), dte_exit=7)
        assert len(s.legs) == 4

    def test_leg_directions(self):
        s = iron_condor(_options_schema(), "SPY",
                        dte_range=(30, 60), dte_exit=7)
        # short call, long call, short put, long put
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[1].direction == Direction.BUY
        assert s.legs[2].direction == Direction.SELL
        assert s.legs[3].direction == Direction.BUY

    def test_leg_types(self):
        s = iron_condor(_options_schema(), "SPY",
                        dte_range=(30, 60), dte_exit=7)
        assert s.legs[0].type == OptionType.CALL
        assert s.legs[1].type == OptionType.CALL
        assert s.legs[2].type == OptionType.PUT
        assert s.legs[3].type == OptionType.PUT


class TestCoveredCall:
    def test_creates_one_leg(self):
        s = covered_call(_options_schema(), "SPY",
                         dte_range=(30, 60), dte_exit=7)
        assert len(s.legs) == 1

    def test_leg_is_sell_call(self):
        s = covered_call(_options_schema(), "SPY",
                         dte_range=(30, 60), dte_exit=7)
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[0].type == OptionType.CALL


class TestCashSecuredPut:
    def test_creates_one_leg(self):
        s = cash_secured_put(_options_schema(), "SPY",
                             dte_range=(30, 60), dte_exit=7)
        assert len(s.legs) == 1

    def test_leg_is_sell_put(self):
        s = cash_secured_put(_options_schema(), "SPY",
                             dte_range=(30, 60), dte_exit=7)
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[0].type == OptionType.PUT


class TestCollar:
    def test_creates_two_legs(self):
        s = collar(_options_schema(), "SPY",
                   dte_range=(30, 60), dte_exit=7)
        assert len(s.legs) == 2

    def test_leg_types_and_directions(self):
        s = collar(_options_schema(), "SPY",
                   dte_range=(30, 60), dte_exit=7)
        # short call + long put
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[0].type == OptionType.CALL
        assert s.legs[1].direction == Direction.BUY
        assert s.legs[1].type == OptionType.PUT


class TestButterfly:
    def test_creates_three_legs(self):
        s = butterfly(_options_schema(), "SPY",
                      dte_range=(30, 60), dte_exit=7)
        assert len(s.legs) == 3

    def test_default_type_is_call(self):
        s = butterfly(_options_schema(), "SPY",
                      dte_range=(30, 60), dte_exit=7)
        for leg in s.legs:
            assert leg.type == OptionType.CALL

    def test_put_butterfly(self):
        s = butterfly(_options_schema(), "SPY",
                      dte_range=(30, 60), dte_exit=7,
                      option_type=OptionType.PUT)
        for leg in s.legs:
            assert leg.type == OptionType.PUT

    def test_directions(self):
        s = butterfly(_options_schema(), "SPY",
                      dte_range=(30, 60), dte_exit=7)
        # buy lower, sell middle, buy upper
        assert s.legs[0].direction == Direction.BUY
        assert s.legs[1].direction == Direction.SELL
        assert s.legs[2].direction == Direction.BUY

    def test_entry_sort_on_wings(self):
        s = butterfly(_options_schema(), "SPY",
                      dte_range=(30, 60), dte_exit=7)
        assert s.legs[0].entry_sort == ("strike", True)   # ascending
        assert s.legs[2].entry_sort == ("strike", False)   # descending
