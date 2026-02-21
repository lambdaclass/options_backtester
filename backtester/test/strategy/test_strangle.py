import pytest

from backtester.strategy.strangle import Strangle
from backtester.datahandler.schema import Schema
from backtester.enums import Type, Direction


@pytest.fixture
def schema():
    s = Schema.options()
    s.update({'dte': 'dte'})
    return s


class TestStrangle:
    def test_long_strangle_creates_buy_call_and_buy_put(self, schema):
        s = Strangle(schema, "long", "SPX", (30, 60), 10)
        assert len(s.legs) == 2
        assert s.legs[0].type == Type.CALL
        assert s.legs[0].direction == Direction.BUY
        assert s.legs[1].type == Type.PUT
        assert s.legs[1].direction == Direction.BUY

    def test_short_strangle_creates_sell_call_and_sell_put(self, schema):
        s = Strangle(schema, "short", "SPX", (30, 60), 10)
        assert len(s.legs) == 2
        assert s.legs[0].type == Type.CALL
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[1].type == Type.PUT
        assert s.legs[1].direction == Direction.SELL

    def test_invalid_name_asserts(self, schema):
        with pytest.raises(AssertionError):
            Strangle(schema, "invalid", "SPX", (30, 60), 10)

    def test_exit_thresholds_propagate(self, schema):
        s = Strangle(schema, "long", "SPX", (30, 60), 10, exit_thresholds=(0.5, 0.3))
        assert s.exit_thresholds == (0.5, 0.3)

    def test_case_insensitive_name(self, schema):
        s = Strangle(schema, "Long", "SPX", (30, 60), 10)
        assert s.legs[0].direction == Direction.BUY
        s2 = Strangle(schema, "SHORT", "SPX", (30, 60), 10)
        assert s2.legs[0].direction == Direction.SELL
