from backtester.enums import Type, Direction, Order, Signal, get_order


class TestTypeInvert:
    def test_invert_call_gives_put(self):
        assert ~Type.CALL == Type.PUT

    def test_invert_put_gives_call(self):
        assert ~Type.PUT == Type.CALL

    def test_double_invert_call(self):
        assert ~(~Type.CALL) == Type.CALL

    def test_double_invert_put(self):
        assert ~(~Type.PUT) == Type.PUT


class TestDirectionInvert:
    def test_invert_buy_gives_sell(self):
        assert ~Direction.BUY == Direction.SELL

    def test_invert_sell_gives_buy(self):
        assert ~Direction.SELL == Direction.BUY


class TestOrderInvert:
    def test_invert_bto_gives_stc(self):
        assert ~Order.BTO == Order.STC

    def test_invert_stc_gives_bto(self):
        assert ~Order.STC == Order.BTO

    def test_invert_sto_gives_btc(self):
        assert ~Order.STO == Order.BTC

    def test_invert_btc_gives_sto(self):
        assert ~Order.BTC == Order.STO


class TestGetOrder:
    def test_buy_entry_is_bto(self):
        assert get_order(Direction.BUY, Signal.ENTRY) == Order.BTO

    def test_buy_exit_is_stc(self):
        assert get_order(Direction.BUY, Signal.EXIT) == Order.STC

    def test_sell_entry_is_sto(self):
        assert get_order(Direction.SELL, Signal.ENTRY) == Order.STO

    def test_sell_exit_is_btc(self):
        assert get_order(Direction.SELL, Signal.EXIT) == Order.BTC
