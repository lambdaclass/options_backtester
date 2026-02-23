"""Tests for core domain types."""

from options_backtester.core.types import (
    Direction, OptionType, Order, Signal, Greeks, Fill, OptionContract,
    StockAllocation, Stock, get_order,
)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

class TestDirection:
    def test_buy_price_column_is_ask(self):
        assert Direction.BUY.price_column == "ask"

    def test_sell_price_column_is_bid(self):
        assert Direction.SELL.price_column == "bid"

    def test_invert_buy(self):
        assert ~Direction.BUY == Direction.SELL

    def test_invert_sell(self):
        assert ~Direction.SELL == Direction.BUY

    def test_decoupled_from_column_name(self):
        """Direction.value is 'buy'/'sell', NOT 'ask'/'bid'."""
        assert Direction.BUY.value == "buy"
        assert Direction.SELL.value == "sell"


# ---------------------------------------------------------------------------
# OptionType
# ---------------------------------------------------------------------------

class TestOptionType:
    def test_invert_call(self):
        assert ~OptionType.CALL == OptionType.PUT

    def test_invert_put(self):
        assert ~OptionType.PUT == OptionType.CALL


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class TestOrder:
    def test_invert_bto(self):
        assert ~Order.BTO == Order.STC

    def test_invert_stc(self):
        assert ~Order.STC == Order.BTO

    def test_invert_sto(self):
        assert ~Order.STO == Order.BTC

    def test_invert_btc(self):
        assert ~Order.BTC == Order.STO


# ---------------------------------------------------------------------------
# get_order
# ---------------------------------------------------------------------------

class TestGetOrder:
    def test_buy_entry(self):
        assert get_order(Direction.BUY, Signal.ENTRY) == Order.BTO

    def test_buy_exit(self):
        assert get_order(Direction.BUY, Signal.EXIT) == Order.STC

    def test_sell_entry(self):
        assert get_order(Direction.SELL, Signal.ENTRY) == Order.STO

    def test_sell_exit(self):
        assert get_order(Direction.SELL, Signal.EXIT) == Order.BTC


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_default_zeros(self):
        g = Greeks()
        assert g.delta == 0.0
        assert g.gamma == 0.0
        assert g.theta == 0.0
        assert g.vega == 0.0

    def test_addition(self):
        a = Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1)
        b = Greeks(delta=-0.3, gamma=0.02, theta=-0.01, vega=0.05)
        result = a + b
        assert abs(result.delta - 0.2) < 1e-10
        assert abs(result.gamma - 0.03) < 1e-10
        assert abs(result.theta - (-0.03)) < 1e-10
        assert abs(result.vega - 0.15) < 1e-10

    def test_scalar_multiply(self):
        g = Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1)
        result = g * 10
        assert abs(result.delta - 5.0) < 1e-10
        assert abs(result.gamma - 0.1) < 1e-10

    def test_rmul(self):
        g = Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1)
        result = 10 * g
        assert abs(result.delta - 5.0) < 1e-10

    def test_negation(self):
        g = Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1)
        neg = -g
        assert abs(neg.delta - (-0.5)) < 1e-10
        assert abs(neg.vega - (-0.1)) < 1e-10

    def test_as_dict(self):
        g = Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1)
        d = g.as_dict
        assert d == {"delta": 0.5, "gamma": 0.01, "theta": -0.02, "vega": 0.1}

    def test_frozen(self):
        g = Greeks(delta=0.5)
        try:
            g.delta = 1.0  # type: ignore
            assert False, "Should have raised"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

class TestFill:
    def test_buy_fill_notional(self):
        f = Fill(price=2.50, quantity=10, direction=Direction.BUY, shares_per_contract=100)
        # -1 * 2.50 * 10 * 100 = -2500
        assert f.notional == -2500.0

    def test_sell_fill_notional(self):
        f = Fill(price=2.50, quantity=10, direction=Direction.SELL, shares_per_contract=100)
        # 1 * 2.50 * 10 * 100 = 2500
        assert f.notional == 2500.0

    def test_fill_with_commission(self):
        f = Fill(price=2.50, quantity=10, direction=Direction.BUY,
                 shares_per_contract=100, commission=6.50)
        # -2500 - 6.50 = -2506.50
        assert f.notional == -2506.50

    def test_fill_with_slippage(self):
        f = Fill(price=2.50, quantity=10, direction=Direction.BUY,
                 shares_per_contract=100, slippage=5.0)
        assert f.notional == -2505.0

    def test_fill_with_commission_and_slippage(self):
        f = Fill(price=2.50, quantity=10, direction=Direction.BUY,
                 shares_per_contract=100, commission=6.50, slippage=5.0)
        assert f.notional == -2511.50


# ---------------------------------------------------------------------------
# OptionContract
# ---------------------------------------------------------------------------

class TestOptionContract:
    def test_creation(self):
        c = OptionContract(
            contract_id="SPY240119C00500000",
            underlying="SPY",
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            strike=500.0,
        )
        assert c.contract_id == "SPY240119C00500000"
        assert c.option_type == OptionType.CALL
        assert c.strike == 500.0


# ---------------------------------------------------------------------------
# StockAllocation / Stock
# ---------------------------------------------------------------------------

class TestStockAllocation:
    def test_creation(self):
        s = StockAllocation("SPY", 0.60)
        assert s.symbol == "SPY"
        assert s.percentage == 0.60

    def test_stock_alias(self):
        s = Stock("VOO", 1.0)
        assert s.symbol == "VOO"
        assert isinstance(s, StockAllocation)
