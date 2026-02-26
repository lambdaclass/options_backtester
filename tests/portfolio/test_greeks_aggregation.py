"""Tests for portfolio-level Greeks aggregation."""

from options_portfolio_backtester.core.types import (
    Direction, OptionType, Order, Greeks,
)
from options_portfolio_backtester.portfolio.position import (
    OptionPosition, PositionLeg,
)
from options_portfolio_backtester.portfolio.greeks import aggregate_greeks


def _make_position(pid, direction=Direction.BUY, qty=10):
    pos = OptionPosition(position_id=pid, quantity=qty, entry_cost=100.0)
    pos.add_leg(PositionLeg(
        name="leg_1",
        contract_id=f"SPY_C_{pid}",
        underlying="SPY",
        expiration="2024-01-01",
        option_type=OptionType.CALL,
        strike=450.0,
        entry_price=5.0,
        direction=direction,
        order=Order.BTO if direction == Direction.BUY else Order.STO,
    ))
    return pos


def test_aggregate_single_position():
    pos = _make_position(1, Direction.BUY, qty=10)
    leg_greeks = {1: {"leg_1": Greeks(delta=0.5, gamma=0.05, theta=-0.1, vega=0.3)}}
    total = aggregate_greeks({1: pos}, leg_greeks)
    # BUY direction: sign=+1, qty=10
    assert total.delta == 0.5 * 10
    assert total.gamma == 0.05 * 10
    assert total.theta == -0.1 * 10
    assert total.vega == 0.3 * 10


def test_aggregate_sell_position():
    pos = _make_position(1, Direction.SELL, qty=5)
    leg_greeks = {1: {"leg_1": Greeks(delta=0.5, gamma=0.05, theta=-0.1, vega=0.3)}}
    total = aggregate_greeks({1: pos}, leg_greeks)
    # SELL direction: sign=-1, qty=5
    assert total.delta == 0.5 * (-1) * 5
    assert total.vega == 0.3 * (-1) * 5


def test_aggregate_multiple_positions():
    pos1 = _make_position(1, Direction.BUY, qty=10)
    pos2 = _make_position(2, Direction.SELL, qty=5)
    leg_greeks = {
        1: {"leg_1": Greeks(delta=0.5, gamma=0.05, theta=-0.1, vega=0.3)},
        2: {"leg_1": Greeks(delta=0.4, gamma=0.04, theta=-0.08, vega=0.2)},
    }
    total = aggregate_greeks({1: pos1, 2: pos2}, leg_greeks)
    expected_delta = 0.5 * 10 + 0.4 * (-1) * 5
    assert abs(total.delta - expected_delta) < 1e-10


def test_aggregate_empty():
    total = aggregate_greeks({}, {})
    assert total.delta == 0.0
    assert total.gamma == 0.0
    assert total.theta == 0.0
    assert total.vega == 0.0


def test_aggregate_missing_greeks_for_position():
    pos = _make_position(1, Direction.BUY, qty=10)
    # No greeks provided for position 1
    total = aggregate_greeks({1: pos}, {})
    assert total.delta == 0.0
    assert total.vega == 0.0
