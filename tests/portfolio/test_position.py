"""Tests for option position and position leg."""

from options_portfolio_backtester.core.types import Direction, OptionType, Order, Greeks
from options_portfolio_backtester.portfolio.position import PositionLeg, OptionPosition


class TestPositionLeg:
    def test_exit_order_inverts(self):
        leg = PositionLeg(
            name="leg_1", contract_id="SPY_C", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        )
        assert leg.exit_order == Order.STC

    def test_buy_leg_value(self):
        leg = PositionLeg(
            name="leg_1", contract_id="SPY_C", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        )
        # BUY: +1 * 6.0 * 10 * 100 = 6000
        assert leg.current_value(6.0, 10, 100) == 6000.0

    def test_sell_leg_value(self):
        leg = PositionLeg(
            name="leg_1", contract_id="SPY_P", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.PUT,
            strike=400.0, entry_price=3.0, direction=Direction.SELL,
            order=Order.STO,
        )
        # SELL: -1 * 4.0 * 10 * 100 = -4000
        assert leg.current_value(4.0, 10, 100) == -4000.0


class TestOptionPosition:
    def _make_position(self) -> OptionPosition:
        pos = OptionPosition(position_id=0, quantity=5, entry_cost=-1500.0)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="SPY_C_500", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=3.0, direction=Direction.BUY,
            order=Order.BTO,
        ))
        return pos

    def test_current_value(self):
        pos = self._make_position()
        # BUY leg: +1 * 4.0 * 5 * 100 = 2000
        val = pos.current_value({"leg_1": 4.0}, 100)
        assert val == 2000.0

    def test_current_value_missing_price(self):
        pos = self._make_position()
        # Missing price defaults to 0
        assert pos.current_value({}, 100) == 0.0

    def test_greeks(self):
        pos = self._make_position()
        leg_greeks = {"leg_1": Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1)}
        result = pos.greeks(leg_greeks)
        # BUY direction, qty=5: delta = 0.5 * 5 = 2.5
        assert abs(result.delta - 2.5) < 1e-10
        assert abs(result.gamma - 0.05) < 1e-10
        assert abs(result.theta - (-0.1)) < 1e-10
        assert abs(result.vega - 0.5) < 1e-10

    def test_multi_leg_greeks(self):
        pos = OptionPosition(position_id=1, quantity=10)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="C1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=3.0, direction=Direction.BUY,
            order=Order.BTO,
        ))
        pos.add_leg(PositionLeg(
            name="leg_2", contract_id="P1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.PUT,
            strike=480.0, entry_price=2.0, direction=Direction.SELL,
            order=Order.STO,
        ))
        leg_greeks = {
            "leg_1": Greeks(delta=0.6, gamma=0.02, theta=-0.03, vega=0.15),
            "leg_2": Greeks(delta=-0.4, gamma=0.01, theta=-0.02, vega=0.10),
        }
        result = pos.greeks(leg_greeks)
        # leg_1 BUY: +1*10 * (0.6, 0.02, -0.03, 0.15) = (6, 0.2, -0.3, 1.5)
        # leg_2 SELL: -1*10 * (-0.4, 0.01, -0.02, 0.10) = (4, -0.1, 0.2, -1.0)
        # total: (10, 0.1, -0.1, 0.5)
        assert abs(result.delta - 10.0) < 1e-10
        assert abs(result.gamma - 0.1) < 1e-10
        assert abs(result.theta - (-0.1)) < 1e-10
        assert abs(result.vega - 0.5) < 1e-10
