"""Tests for Strategy class: adding/removing legs, thresholds."""

import math

import pytest
import numpy as np
import pandas as pd

from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.data.schema import Schema
from options_portfolio_backtester.core.types import OptionType as Type, Direction


@pytest.fixture
def schema():
    return Schema.options()


@pytest.fixture
def strategy(schema):
    return Strategy(schema)


@pytest.fixture
def make_leg(schema):
    def _make(option_type=Type.CALL, direction=Direction.BUY):
        return StrategyLeg("leg", schema, option_type=option_type, direction=direction)
    return _make


class TestAddLeg:
    def test_add_one_leg(self, strategy, make_leg):
        leg = make_leg()
        strategy.add_leg(leg)
        assert len(strategy.legs) == 1
        assert strategy.legs[0].name == "leg_1"

    def test_add_two_legs(self, strategy, make_leg):
        strategy.add_leg(make_leg())
        strategy.add_leg(make_leg(Type.PUT))
        assert len(strategy.legs) == 2
        assert strategy.legs[0].name == "leg_1"
        assert strategy.legs[1].name == "leg_2"

    def test_add_legs_batch(self, strategy, make_leg):
        legs = [make_leg(), make_leg(Type.PUT)]
        strategy.add_legs(legs)
        assert len(strategy.legs) == 2

    def test_add_leg_schema_mismatch_asserts(self, strategy):
        other_schema = Schema.options()
        other_schema.update({"underlying": "different_col"})
        leg = StrategyLeg("leg", other_schema)
        with pytest.raises(AssertionError):
            strategy.add_leg(leg)


class TestRemoveLeg:
    def test_remove_leg(self, strategy, make_leg):
        strategy.add_legs([make_leg(), make_leg(Type.PUT)])
        strategy.remove_leg(0)
        assert len(strategy.legs) == 1

    def test_clear_legs(self, strategy, make_leg):
        strategy.add_legs([make_leg(), make_leg(Type.PUT)])
        strategy.clear_legs()
        assert len(strategy.legs) == 0


class TestExitThresholds:
    def test_default_thresholds(self, strategy):
        assert strategy.exit_thresholds == (math.inf, math.inf)

    def test_set_valid_thresholds(self, strategy):
        strategy.add_exit_thresholds(0.5, 0.3)
        assert strategy.exit_thresholds == (0.5, 0.3)

    def test_negative_profit_asserts(self, strategy):
        with pytest.raises(AssertionError):
            strategy.add_exit_thresholds(-0.1, 0.3)

    def test_negative_loss_asserts(self, strategy):
        with pytest.raises(AssertionError):
            strategy.add_exit_thresholds(0.5, -0.1)


class TestFilterThresholds:
    def test_within_bounds_no_exit(self, strategy):
        strategy.add_exit_thresholds(0.5, 0.5)
        entry_cost = pd.Series([100.0])
        current_cost = pd.Series([-110.0])
        result = strategy.filter_thresholds(entry_cost, current_cost)
        assert not result.any()

    def test_profit_exceeded(self, strategy):
        strategy.add_exit_thresholds(0.5, 0.5)
        entry_cost = pd.Series([100.0])
        current_cost = pd.Series([-200.0])
        result = strategy.filter_thresholds(entry_cost, current_cost)
        assert result.all()

    def test_loss_exceeded(self, strategy):
        strategy.add_exit_thresholds(0.5, 0.5)
        entry_cost = pd.Series([100.0])
        current_cost = pd.Series([-10.0])
        result = strategy.filter_thresholds(entry_cost, current_cost)
        assert result.all()
