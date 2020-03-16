from backtester.datahandler.schema import Field


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
    """Test composition of three filters mixing and + or"""
    symbol_field = Field("underlying", "underlying")
    strike_field = Field("strike", "strike")
    ft1 = symbol_field == "SPX"
    ft2 = strike_field >= 200
    ft3 = strike_field < 100
    composed = ft1 & (ft2 | ft3)
    assert composed.query == "(underlying == 'SPX') & (((strike >= 200) | (strike < 100)))"


def test_add_number_to_field():
    """Test addition of a number to a field"""
    strike_field = Field("strike", "strike")
    field = strike_field + 10
    assert field.name == "strike + 10"
    assert field.mapping == "strike + 10"


def test_subtract_number_from_field():
    """Test subtraction of a number from a field"""
    strike_field = Field("strike", "strike")
    field = strike_field - 10
    assert field.name == "strike - 10"
    assert field.mapping == "strike - 10"


def test_multiply_field_by_number():
    """Test multiplication of a field by a number"""
    underlying_last = Field("last", "underlying_last")
    field = underlying_last * 1.5
    assert field.name == "last * 1.5"
    assert field.mapping == "underlying_last * 1.5"


def test_multiply_on_left():
    """Test multiplication of a field by a number on the *left*"""
    underlying_last = Field("last", "underlying_last")
    field = 1.5 * underlying_last
    assert field.name == "1.5 * last"
    assert field.mapping == "1.5 * underlying_last"


def test_filter_from_combined_field():
    """Test filter from a linear combination of fields"""
    underlying_last = Field("last", "underlying_last")
    strike_field = Field("strike", "strike")
    combined_filter = underlying_last == strike_field * 1.2
    assert combined_filter.query == "underlying_last == strike * 1.2"
