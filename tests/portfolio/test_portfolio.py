"""Tests for Portfolio class."""

from options_backtester.core.types import Direction, OptionType, Order, Greeks
from options_backtester.portfolio.portfolio import Portfolio, StockHolding
from options_backtester.portfolio.position import OptionPosition, PositionLeg


def _make_portfolio() -> Portfolio:
    p = Portfolio(initial_cash=100_000.0)

    pos = OptionPosition(position_id=p.next_position_id(), quantity=10, entry_cost=-5000.0)
    pos.add_leg(PositionLeg(
        name="leg_1", contract_id="SPY_C_500", underlying="SPY",
        expiration="2024-01-19", option_type=OptionType.CALL,
        strike=500.0, entry_price=5.0, direction=Direction.BUY,
        order=Order.BTO,
    ))
    p.add_option_position(pos)
    p.set_stock_holding("SPY", 100, 480.0)
    return p


class TestPortfolio:
    def test_initial_cash(self):
        p = Portfolio(initial_cash=50_000.0)
        assert p.cash == 50_000.0

    def test_add_remove_position(self):
        p = Portfolio()
        pos = OptionPosition(position_id=0, quantity=1)
        p.add_option_position(pos)
        assert 0 in p.option_positions
        removed = p.remove_option_position(0)
        assert removed is pos
        assert 0 not in p.option_positions

    def test_remove_nonexistent(self):
        p = Portfolio()
        assert p.remove_option_position(999) is None

    def test_next_position_id_increments(self):
        p = Portfolio()
        assert p.next_position_id() == 0
        assert p.next_position_id() == 1
        assert p.next_position_id() == 2

    def test_options_value(self):
        p = _make_portfolio()
        # BUY leg: +1 * 6.0 * 10 * 100 = 6000
        val = p.options_value({0: {"leg_1": 6.0}}, 100)
        assert val == 6000.0

    def test_stocks_value(self):
        p = _make_portfolio()
        val = p.stocks_value({"SPY": 490.0})
        assert val == 100 * 490.0

    def test_total_value(self):
        p = _make_portfolio()
        total = p.total_value(
            stock_prices={"SPY": 490.0},
            option_prices={0: {"leg_1": 6.0}},
            shares_per_contract=100,
        )
        # cash=100000, stocks=49000, options=6000
        assert total == 155_000.0

    def test_clear_stock_holdings(self):
        p = _make_portfolio()
        p.clear_stock_holdings()
        assert len(p.stock_holdings) == 0
        assert p.stocks_value({"SPY": 490.0}) == 0.0

    def test_portfolio_greeks(self):
        p = _make_portfolio()
        greeks_map = {0: {"leg_1": Greeks(delta=0.5, gamma=0.01)}}
        result = p.portfolio_greeks(greeks_map)
        # BUY, qty=10: delta = 0.5 * 10 = 5.0
        assert abs(result.delta - 5.0) < 1e-10


class TestPortfolioInvariant:
    """Test: cash + stocks + options == total on every operation."""

    def test_invariant_after_stock_buy(self):
        p = Portfolio(initial_cash=100_000.0)
        price = 150.0
        qty = 100
        p.cash -= price * qty
        p.set_stock_holding("SPY", qty, price)
        total = p.total_value({"SPY": price}, {}, 100)
        assert abs(total - 100_000.0) < 1e-10

    def test_invariant_after_option_buy(self):
        p = Portfolio(initial_cash=100_000.0)
        cost = 5.0 * 10 * 100  # 5000
        p.cash -= cost
        pos = OptionPosition(position_id=0, quantity=10, entry_cost=-cost)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="C1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        ))
        p.add_option_position(pos)
        total = p.total_value({}, {0: {"leg_1": 5.0}}, 100)
        assert abs(total - 100_000.0) < 1e-10
