"""Worker function for Taleb barbell parallel sweep.

Extracted from taleb_barbell.ipynb so ProcessPoolExecutor can pickle it
(cell-defined functions aren't picklable when run via nbconvert).
"""

import math
import warnings


def run_config(args):
    """Run a single barbell backtest configuration.

    Args is a 13-element tuple:
        (name, opt_pct, rebal_freq,
         atm_dte_min, atm_dte_max, atm_exit_dte, atm_strike_width,
         otm_delta_min, otm_delta_max, otm_dte_min, otm_dte_max, otm_exit_dte,
         max_notional_pct)

    Returns a dict with performance metrics.
    """
    (name, opt_pct, rebal_freq,
     atm_dte_min, atm_dte_max, atm_exit_dte, atm_strike_width,
     otm_delta_min, otm_delta_max, otm_dte_min, otm_dte_max, otm_exit_dte,
     max_notional_pct) = args

    warnings.filterwarnings('ignore')

    from options_portfolio_backtester import BacktestEngine, Stock, Direction
    from options_portfolio_backtester.core.types import OptionType as Type
    from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

    options_data = HistoricalOptionsData('data/processed/options.csv')
    stocks_data = TiingoData('data/processed/stocks.csv')
    schema = options_data.schema

    spy_prices = (
        stocks_data._data[stocks_data._data['symbol'] == 'SPY']
        .set_index('date')['adjClose'].sort_index()
    )
    bt_years = (spy_prices.index[-1] - spy_prices.index[0]).days / 365.25

    # Build strategy inline (Schema/Filter objects aren't picklable)
    lo = 1.0 - atm_strike_width
    hi = 1.0 + atm_strike_width

    atm_call = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.SELL)
    atm_call.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= atm_dte_min) & (schema.dte <= atm_dte_max)
        & (schema.strike >= schema.underlying_last * lo)
        & (schema.strike <= schema.underlying_last * hi)
    )
    atm_call.entry_sort = ('delta', False)
    atm_call.exit_filter = schema.dte <= atm_exit_dte

    atm_put = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=Direction.SELL)
    atm_put.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= atm_dte_min) & (schema.dte <= atm_dte_max)
        & (schema.strike >= schema.underlying_last * lo)
        & (schema.strike <= schema.underlying_last * hi)
    )
    atm_put.entry_sort = ('delta', True)
    atm_put.exit_filter = schema.dte <= atm_exit_dte

    otm_put = StrategyLeg('leg_3', schema, option_type=Type.PUT, direction=Direction.BUY)
    otm_put.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= otm_dte_min) & (schema.dte <= otm_dte_max)
        & (schema.delta >= otm_delta_min) & (schema.delta <= otm_delta_max)
    )
    otm_put.entry_sort = ('delta', False)
    otm_put.exit_filter = schema.dte <= otm_exit_dte

    strat = Strategy(schema)
    strat.add_legs([atm_call, atm_put, otm_put])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    # Run backtest — SPY equity base + options overlay
    stock_pct = 1.0 - opt_pct
    bt = BacktestEngine(
        {'stocks': stock_pct, 'options': opt_pct, 'cash': 0.0},
        initial_capital=1_000_000,
        max_notional_pct=max_notional_pct,
    )
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strat
    bt.options_data = options_data
    bt.run(rebalance_freq=rebal_freq)

    balance = bt.balance
    total_cap = balance['total capital']
    total_ret = (balance['accumulated return'].iloc[-1] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / bt_years) - 1) * 100
    cummax = total_cap.cummax()
    drawdown = (total_cap - cummax) / cummax
    max_dd = drawdown.min() * 100

    daily_rets = balance['% change'].dropna()
    vol = daily_rets.std() * (252 ** 0.5) * 100
    sharpe = (annual_ret - 4.0) / vol if vol > 0 else 0

    # Crisis returns
    crisis_rets = {}
    for label, cs, ce in [
        ('2008 GFC',          '2007-10-01', '2009-03-09'),
        ('2011 US Downgrade',  '2011-07-22', '2011-10-03'),
        ('2015 China Deval',   '2015-08-10', '2015-08-25'),
        ('2018 Volmageddon',   '2018-01-26', '2018-02-08'),
        ('2018 Q4 Selloff',    '2018-10-01', '2018-12-24'),
        ('2020 COVID',         '2020-02-19', '2020-03-23'),
        ('2022 Bear',          '2022-01-03', '2022-10-12'),
    ]:
        sl = total_cap[(total_cap.index >= cs) & (total_cap.index <= ce)]
        if len(sl) > 1:
            crisis_rets[label] = (sl.iloc[-1] / sl.iloc[0] - 1) * 100
        else:
            crisis_rets[label] = float('nan')

    return {
        'name': name,
        'opt_pct': opt_pct,
        'rebal_freq': rebal_freq,
        'max_notional_pct': max_notional_pct,
        'atm_dte_min': atm_dte_min, 'atm_dte_max': atm_dte_max,
        'atm_exit_dte': atm_exit_dte, 'atm_strike_width': atm_strike_width,
        'otm_delta_min': otm_delta_min, 'otm_delta_max': otm_delta_max,
        'otm_dte_min': otm_dte_min, 'otm_dte_max': otm_dte_max,
        'otm_exit_dte': otm_exit_dte,
        'annual_ret': annual_ret,
        'total_ret': total_ret,
        'max_dd': max_dd,
        'vol': vol,
        'sharpe': sharpe,
        'trades': len(bt.trade_log),
        'balance_values': total_cap.to_dict(),
        **crisis_rets,
    }
