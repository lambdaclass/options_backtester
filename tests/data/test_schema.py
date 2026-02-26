"""Tests for Schema, Field, and Filter DSL."""

import pandas as pd
import pytest

from options_portfolio_backtester.data.schema import Schema, Field, Filter


class TestSchema:
    def test_stocks_factory(self):
        s = Schema.stocks()
        assert "symbol" in s
        assert "date" in s
        assert "adjClose" in s

    def test_options_factory(self):
        s = Schema.options()
        assert "underlying" in s
        assert "strike" in s
        assert "bid" in s
        assert "ask" in s

    def test_getitem(self):
        s = Schema.stocks()
        assert s["symbol"] == "symbol"

    def test_getattr_returns_field(self):
        s = Schema.stocks()
        f = s.symbol
        assert isinstance(f, Field)
        assert f.mapping == "symbol"

    def test_update(self):
        s = Schema.stocks()
        s.update({"custom": "custom_col"})
        assert s["custom"] == "custom_col"

    def test_contains(self):
        s = Schema.stocks()
        assert "symbol" in s
        assert "nonexistent" not in s

    def test_setitem(self):
        s = Schema.stocks()
        s["new_field"] = "new_col"
        assert s["new_field"] == "new_col"

    def test_iter(self):
        s = Schema.stocks()
        pairs = list(s)
        assert any(k == "symbol" for k, _ in pairs)

    def test_repr(self):
        s = Schema.stocks()
        r = repr(s)
        assert "Schema" in r

    def test_equality(self):
        s1 = Schema.stocks()
        s2 = Schema.stocks()
        assert s1 == s2

    def test_inequality_different_schema(self):
        s1 = Schema.stocks()
        s2 = Schema.options()
        assert s1 != s2

    def test_equality_with_non_schema(self):
        s = Schema.stocks()
        assert s != "not a schema"


class TestField:
    def test_repr(self):
        f = Field("strike", "strike")
        assert "Field" in repr(f)
        assert "strike" in repr(f)

    def test_comparison_operators(self):
        s = Schema.options()
        f = s.strike > 100
        assert isinstance(f, Filter)
        assert "100" in f.query

    def test_equality_operator_string(self):
        s = Schema.options()
        f = s.underlying == "SPY"
        assert isinstance(f, Filter)
        assert "'SPY'" in f.query

    def test_arithmetic_field_field(self):
        s = Schema.options()
        combined = s.strike + s.bid
        assert isinstance(combined, Field)
        assert "+" in combined.mapping

    def test_arithmetic_field_scalar(self):
        s = Schema.options()
        combined = s.strike * 1.05
        assert isinstance(combined, Field)
        assert "*" in combined.mapping

    def test_radd(self):
        s = Schema.options()
        combined = 100 + s.strike
        assert isinstance(combined, Field)

    def test_rsub(self):
        s = Schema.options()
        combined = 100 - s.strike
        assert isinstance(combined, Field)

    def test_rtruediv(self):
        s = Schema.options()
        combined = 1 / s.strike
        assert isinstance(combined, Field)

    def test_rmul(self):
        s = Schema.options()
        combined = 2 * s.strike
        assert isinstance(combined, Field)

    def test_ne_operator(self):
        s = Schema.options()
        f = s.underlying != "SPY"
        assert isinstance(f, Filter)
        assert "!=" in f.query


class TestFilter:
    def test_and(self):
        s = Schema.options()
        f = (s.strike > 100) & (s.strike < 200)
        assert isinstance(f, Filter)
        assert "&" in f.query

    def test_or(self):
        s = Schema.options()
        f = (s.strike > 100) | (s.strike < 50)
        assert isinstance(f, Filter)
        assert "|" in f.query

    def test_invert(self):
        s = Schema.options()
        f = ~(s.strike > 100)
        assert isinstance(f, Filter)
        assert "!" in f.query

    def test_call_on_dataframe(self):
        df = pd.DataFrame({"strike": [100, 200, 300]})
        s = Schema.options()
        f = s.strike > 150
        result = f(df)
        assert isinstance(result, pd.Series)
        assert result.sum() == 2

    def test_repr(self):
        f = Filter("strike > 100")
        assert "Filter" in repr(f)
        assert "strike > 100" in repr(f)
