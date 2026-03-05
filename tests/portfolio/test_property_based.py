"""Property-based tests for portfolio position and portfolio invariants."""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from options_portfolio_backtester.core.types import Direction, OptionType, Order
from options_portfolio_backtester.portfolio.portfolio import Portfolio
from options_portfolio_backtester.portfolio.position import PositionLeg, OptionPosition


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

price = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)
quantity_int = st.integers(min_value=0, max_value=10000)
cash_amount = st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False)
spc = st.sampled_from([1, 10, 100])


# ---------------------------------------------------------------------------
# Position properties
# ---------------------------------------------------------------------------

class TestPositionProperties:
    @given(price, spc)
    @settings(max_examples=50)
    def test_zero_quantity_zero_value(self, p, shares_per_contract):
        """Position with quantity 0 has value 0."""
        pos = OptionPosition(position_id=0, quantity=0)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="C1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        ))
        val = pos.current_value({"leg_1": p}, shares_per_contract)
        assert val == 0.0

    @given(quantity_int, price, spc)
    @settings(max_examples=50)
    def test_buy_leg_value_formula(self, qty, p, shares_per_contract):
        """BUY leg value = +1 * price * quantity * shares_per_contract."""
        pos = OptionPosition(position_id=0, quantity=qty)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="C1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        ))
        val = pos.current_value({"leg_1": p}, shares_per_contract)
        expected = p * qty * shares_per_contract
        assert abs(val - expected) < 1e-6

    @given(quantity_int, price, spc)
    @settings(max_examples=50)
    def test_sell_leg_value_formula(self, qty, p, shares_per_contract):
        """SELL leg value = -1 * price * quantity * shares_per_contract."""
        pos = OptionPosition(position_id=0, quantity=qty)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="P1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.PUT,
            strike=400.0, entry_price=3.0, direction=Direction.SELL,
            order=Order.STO,
        ))
        val = pos.current_value({"leg_1": p}, shares_per_contract)
        expected = -p * qty * shares_per_contract
        assert abs(val - expected) < 1e-6


# ---------------------------------------------------------------------------
# Portfolio properties
# ---------------------------------------------------------------------------

class TestPortfolioProperties:
    @given(cash_amount, price, quantity_int)
    @settings(max_examples=50)
    def test_total_value_is_sum(self, cash, stock_price, qty):
        """Total value = cash + sum of stock values."""
        p = Portfolio(initial_cash=cash)
        if qty > 0:
            p.set_stock_holding("SPY", qty, stock_price)
        total = p.total_value({"SPY": stock_price}, {}, 100)
        expected = cash + qty * stock_price
        assert abs(total - expected) < 1e-4

    @given(cash_amount, quantity_int, price, spc)
    @settings(max_examples=50)
    def test_add_remove_position_roundtrip(self, cash, qty, p, shares_per_contract):
        """Adding then removing a position returns to original state."""
        portfolio = Portfolio(initial_cash=cash)
        original_total = portfolio.total_value({}, {}, shares_per_contract)

        pos = OptionPosition(position_id=portfolio.next_position_id(), quantity=qty)
        pos.add_leg(PositionLeg(
            name="leg_1", contract_id="C1", underlying="SPY",
            expiration="2024-01-19", option_type=OptionType.CALL,
            strike=500.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        ))
        portfolio.add_option_position(pos)
        portfolio.remove_option_position(pos.position_id)

        after_total = portfolio.total_value({}, {}, shares_per_contract)
        assert abs(after_total - original_total) < 1e-10

    @given(cash_amount, price, quantity_int)
    @settings(max_examples=50)
    def test_portfolio_value_non_negative(self, cash, stock_price, qty):
        """Portfolio value >= 0 when all inputs are non-negative."""
        p = Portfolio(initial_cash=cash)
        if qty > 0:
            p.set_stock_holding("SPY", qty, stock_price)
        total = p.total_value({"SPY": stock_price}, {}, 100)
        assert total >= -1e-10
