"""Smoke tests: verify all modules import and a tiny backtest runs."""

import math

import pytest


def test_import_all_modules():
    """Every public module should import without error."""
    import backtester
    from backtester import Backtest, Stock, Type, Direction
    from backtester.datahandler import HistoricalOptionsData, TiingoData
    from backtester.datahandler.schema import Schema, Field, Filter
    from backtester.strategy import Strategy, StrategyLeg
    from backtester.strategy.strangle import Strangle
    from backtester.statistics.stats import summary
    from backtester.statistics.charts import returns_chart, returns_histogram, monthly_returns_heatmap
    from backtester.enums import Order, Signal, get_order

    # Sanity: enums work
    assert Type.CALL.value == 'call'
    assert Direction.BUY.value == 'ask'
    assert ~Direction.BUY == Direction.SELL


def test_tiny_backtest(sample_stocks_datahandler, sample_options_datahandler):
    """Run a minimal backtest with test data to verify the pipeline doesn't crash."""
    from backtester import Backtest, Stock, Type, Direction
    from backtester.strategy import Strategy, StrategyLeg

    schema = sample_options_datahandler.schema

    # Test data uses VOO as the stock symbol, not SPY
    stock_symbols = sample_stocks_datahandler._data['symbol'].unique()
    test_symbol = stock_symbols[0]

    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.dte >= 20) & (schema.dte <= 180)
    leg.exit_filter = (schema.dte <= 10)
    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    # Use equal weight across all stocks in the test data
    n_stocks = len(stock_symbols)
    bt = Backtest({'stocks': 0.99, 'options': 0.01, 'cash': 0.0}, initial_capital=100_000)
    bt.stocks = [Stock(sym, 1.0 / n_stocks) for sym in stock_symbols]
    bt.stocks_data = sample_stocks_datahandler
    bt.options_strategy = strategy
    bt.options_data = sample_options_datahandler
    bt.run(rebalance_freq=1)

    assert 'total capital' in bt.balance.columns
    assert len(bt.balance) > 0
    assert bt.balance['total capital'].iloc[-1] > 0


def test_backtest_runner_imports():
    """backtest_runner module should import and expose expected functions."""
    from backtest_runner import (
        load_data,
        make_puts_strategy,
        make_calls_strategy,
        make_straddle_strategy,
        make_strangle_strategy,
        run_backtest,
        print_results_table,
        plot_results,
    )
    # Just verify they're callable
    assert callable(load_data)
    assert callable(make_puts_strategy)
    assert callable(run_backtest)
    assert callable(print_results_table)
    assert callable(plot_results)
