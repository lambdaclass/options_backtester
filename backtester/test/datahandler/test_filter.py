from backtester.datahandler import Field


def test_strike_eq_100():
    """Test filter for 'strike' == 100"""
    strike_field = Field("strike", "strike")
    ft = strike_field == 100
    assert ft.query == "strike == 100"


def test_strike_lt_100():
    """Test filter for 'strike' < 100"""
    strike_field = Field("strike", "strike")
    ft = strike_field < 100
    assert ft.query == "strike < 100"


def test_strike_ge_100():
    """Test filter for 'strike' >= 100"""
    strike_field = Field("strike", "strike")
    ft = strike_field >= 100
    assert ft.query == "strike >= 100"


def test_negate_filter():
    """Test negations of a filter"""
    symbol_field = Field("underlying", "underlying")
    ft = symbol_field == "SPX"
    negated = ~ft
    assert negated.query == "!(underlying == 'SPX')"


def test_compose_filters_with_and():
    """Test composition of two filters with and"""
    symbol_field = Field("underlying", "underlying")
    strike_field = Field("strike", "strike")
    ft1 = symbol_field == "SPX"
    ft2 = strike_field < 200
    composed = ft1 & ft2
    assert composed.query == "(underlying == 'SPX') & (strike < 200)"


def test_compose_filters_with_or():
    """Test composition of two filters with or"""
    strike_field = Field("strike", "strike")
    ft1 = strike_field >= 200
    ft2 = strike_field < 100
    composed = ft1 | ft2
    assert composed.query == "((strike >= 200) | (strike < 100))"


def test_compose_many_filters():
    symbol_field = Field("underlying", "underlying")
    strike_field = Field("strike", "strike")
    ft1 = symbol_field == "SPX"
    ft2 = strike_field >= 200
    ft3 = strike_field < 100
    composed = ft1 & (ft2 | ft3)
    assert composed.query == "(underlying == 'SPX') & (((strike >= 200) | (strike < 100)))"
