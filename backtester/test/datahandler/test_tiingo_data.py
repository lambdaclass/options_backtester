def test_sma(sample_stocks_datahandler):
    data = sample_stocks_datahandler
    data.sma(30)
    for symbol in data['symbol'].unique():
        symbol_data = data.query('symbol == "{}"'.format(symbol))
        sma = symbol_data.rolling(30)['adjClose'].mean().fillna(0)
        assert (symbol_data['sma'] == sma).all()


def test_constant_price_sma(constant_price_stocks):
    fixed_price = constant_price_stocks['adjClose'].iloc[0]
    for window in range(1, 20):
        constant_price_stocks.sma(window)
        for _symbol, data in constant_price_stocks.groupby('symbol'):
            assert (data['sma'][window:] == fixed_price).all()
