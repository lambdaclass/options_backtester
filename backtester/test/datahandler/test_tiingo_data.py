from backtester.datahandler import TiingoData


def test_sma(sample_stocks_datahandler):
    data = sample_stocks_datahandler
    data.sma(30)
    for symbol in data['symbol'].unique():
        symbol_data = data.query('symbol == "{}"'.format(symbol))
        sma = symbol_data.rolling(30)['adjClose'].mean().fillna(0)
        assert (symbol_data['sma'] == sma).all()