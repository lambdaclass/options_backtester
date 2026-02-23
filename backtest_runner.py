#!/usr/bin/env python3
"""Shared helpers for sweep scripts.

Provides data loading, backtest execution, result formatting,
and 4-panel charting so individual sweeps stay short.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000
REBAL_MONTHS = 1

# OTM put defaults
DELTA_MIN = -0.25
DELTA_MAX = -0.10
DTE_MIN = 60
DTE_MAX = 120
EXIT_DTE = 30

# OTM call defaults
CALL_DELTA_MIN = 0.10
CALL_DELTA_MAX = 0.25


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    options_path: str = 'data/processed/options.csv',
    stocks_path: str = 'data/processed/stocks.csv',
    signals_path: str = 'data/processed/signals.csv',
) -> dict[str, Any]:
    """Load options, stocks, signals and compute SPY baseline stats.

    Returns dict with keys:
        options_data, stocks_data, schema, spy_prices, years,
        spy_total_ret, spy_annual_ret, spy_dd, spy_cummax,
        signals_df (or None), vix, vix_median, buffett, buffett_median,
        tobin, tobin_median
    """
    print("Loading data...")
    options_data = HistoricalOptionsData(options_path)
    stocks_data = TiingoData(stocks_path)
    schema = options_data.schema

    spy_prices = (
        stocks_data._data[stocks_data._data['symbol'] == 'SPY']
        .set_index('date')['adjClose']
        .sort_index()
    )
    years = (spy_prices.index[-1] - spy_prices.index[0]).days / 365.25

    spy_total_ret = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100
    spy_annual_ret = ((1 + spy_total_ret / 100) ** (1 / years) - 1) * 100
    spy_cummax = spy_prices.cummax()
    spy_dd = ((spy_prices - spy_cummax) / spy_cummax).min() * 100

    print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date} ({years:.1f} years)")
    print(f"SPY B&H: {spy_total_ret:.1f}% total, {spy_annual_ret:.2f}% annual, {spy_dd:.1f}% max DD\n")

    # Signals
    signals_df = None
    vix = vix_median = buffett = buffett_median = tobin = tobin_median = None

    if os.path.exists(signals_path):
        signals_df = pd.read_csv(signals_path, parse_dates=['date'], index_col='date')
        vix = signals_df['vix'].dropna()
        vix_median = vix.rolling(252, min_periods=60).median()
        if 'buffett_indicator' in signals_df.columns:
            buffett = signals_df['buffett_indicator']
            buffett_median = buffett.rolling(252, min_periods=60).median()
        if 'tobin_q' in signals_df.columns:
            tobin = signals_df['tobin_q']
            tobin_median = tobin.rolling(252, min_periods=60).median()
        print(f"Loaded macro signals: {list(signals_df.columns)}")
    else:
        print("No signals.csv — skipping macro filters. Run data/fetch_signals.py first.")

    return {
        'options_data': options_data,
        'stocks_data': stocks_data,
        'schema': schema,
        'spy_prices': spy_prices,
        'years': years,
        'spy_total_ret': spy_total_ret,
        'spy_annual_ret': spy_annual_ret,
        'spy_dd': spy_dd,
        'spy_cummax': spy_cummax,
        'signals_df': signals_df,
        'vix': vix,
        'vix_median': vix_median,
        'buffett': buffett,
        'buffett_median': buffett_median,
        'tobin': tobin,
        'tobin_median': tobin_median,
    }


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------
def make_puts_strategy(schema: Any) -> Strategy:
    """OTM put hedge: buy delta -0.25 to -0.10 puts."""
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.delta >= DELTA_MIN) & (schema.delta <= DELTA_MAX)
    )
    leg.entry_sort = ('delta', False)
    leg.exit_filter = (schema.dte <= EXIT_DTE)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_calls_strategy(schema: Any) -> Strategy:
    """OTM call momentum: buy delta 0.10 to 0.25 calls."""
    leg = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.delta >= CALL_DELTA_MIN) & (schema.delta <= CALL_DELTA_MAX)
    )
    leg.entry_sort = ('delta', True)
    leg.exit_filter = (schema.dte <= EXIT_DTE)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_straddle_strategy(schema: Any, direction: Direction = Direction.BUY) -> Strategy:
    """ATM straddle: buy (or sell) ATM call + put.

    Uses strike within 1% of underlying_last as ATM proxy.
    """
    call_leg = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=direction)
    call_leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.strike >= schema.underlying_last * 0.99)
        & (schema.strike <= schema.underlying_last * 1.01)
    )
    call_leg.entry_sort = ('delta', False)  # closest to ATM
    call_leg.exit_filter = (schema.dte <= EXIT_DTE)

    put_leg = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=direction)
    put_leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.strike >= schema.underlying_last * 0.99)
        & (schema.strike <= schema.underlying_last * 1.01)
    )
    put_leg.entry_sort = ('delta', True)  # closest to ATM
    put_leg.exit_filter = (schema.dte <= EXIT_DTE)

    s = Strategy(schema)
    s.add_leg(call_leg)
    s.add_leg(put_leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_strangle_strategy(schema: Any, direction: Direction = Direction.SELL) -> Strategy:
    """OTM strangle: sell (or buy) OTM call + OTM put."""
    call_leg = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=direction)
    call_leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.delta >= CALL_DELTA_MIN) & (schema.delta <= CALL_DELTA_MAX)
    )
    call_leg.entry_sort = ('delta', True)
    call_leg.exit_filter = (schema.dte <= EXIT_DTE)

    put_leg = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=direction)
    put_leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.delta >= DELTA_MIN) & (schema.delta <= DELTA_MAX)
    )
    put_leg.entry_sort = ('delta', False)
    put_leg.exit_filter = (schema.dte <= EXIT_DTE)

    s = Strategy(schema)
    s.add_leg(call_leg)
    s.add_leg(put_leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_covered_call_strategy(schema: Any) -> Strategy:
    """Covered call (BXM replica): sell ATM calls monthly.

    Sell call with delta 0.40-0.60 (near ATM), DTE 30-60, exit at DTE <= 7.
    Paper ref: Whaley (2002), Feldman & Roy (2005) — comparable returns, 2/3 vol.
    """
    leg = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.SELL)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= 30) & (schema.dte <= 60)
        & (schema.delta >= 0.40) & (schema.delta <= 0.60)
    )
    leg.entry_sort = ('delta', False)  # closest to 0.50
    leg.exit_filter = (schema.dte <= 7)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_cash_secured_put_strategy(schema: Any) -> Strategy:
    """Cash-secured put writing (PUT index replica): sell OTM puts.

    Sell puts with delta -0.30 to -0.15, DTE 30-60, exit at DTE <= 7.
    Paper ref: Neuberger Berman — PUT index beats BXM by ~1%/yr.
    """
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.SELL)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= 30) & (schema.dte <= 60)
        & (schema.delta >= -0.30) & (schema.delta <= -0.15)
    )
    leg.entry_sort = ('delta', True)  # closest to ATM first
    leg.exit_filter = (schema.dte <= 7)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_collar_strategy(schema: Any) -> Strategy:
    """Collar: buy OTM put + sell OTM call (zero-cost hedge).

    Paper ref: Israelov & Klein (2015) — collars reduce equity premium capture.
    Braun et al. (2023) — hidden rebalance timing luck costs ~400bps.
    """
    # Sell OTM call
    call_leg = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.SELL)
    call_leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.delta >= 0.20) & (schema.delta <= 0.35)
    )
    call_leg.entry_sort = ('delta', False)
    call_leg.exit_filter = (schema.dte <= EXIT_DTE)

    # Buy OTM put
    put_leg = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=Direction.BUY)
    put_leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX)
        & (schema.delta >= -0.35) & (schema.delta <= -0.20)
    )
    put_leg.entry_sort = ('delta', True)
    put_leg.exit_filter = (schema.dte <= EXIT_DTE)

    s = Strategy(schema)
    s.add_leg(call_leg)
    s.add_leg(put_leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_deep_otm_put_strategy(
    schema: Any,
    delta_min: float = -0.10,
    delta_max: float = -0.02,
    dte_min: int = 90,
    dte_max: int = 180,
    exit_dte: int = 14,
) -> Strategy:
    """Deep OTM tail hedge (Universa-style): buy far-OTM puts.

    Paper ref: Spitznagel (2021) — 3.3% allocation claims 12.3% CAGR.
    AQR (Ilmanen & Israelov 2018) — argues cost exceeds benefit.
    """
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.delta >= delta_min) & (schema.delta <= delta_max)
    )
    leg.entry_sort = ('delta', False)  # deepest OTM first
    leg.exit_filter = (schema.dte <= exit_dte)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------
def run_backtest(
    name: str,
    stock_pct: float,
    opt_pct: float,
    strategy_fn: Callable[[], Strategy],
    data: dict[str, Any],
    budget_fn: Callable[[pd.Timestamp, float], float] | None = None,
    initial_capital: int = INITIAL_CAPITAL,
    rebal_months: int = REBAL_MONTHS,
    rebal_unit: str = 'BMS',
) -> dict[str, Any]:
    """Run a single backtest configuration and return results dict.

    rebal_unit: pandas frequency unit. 'BMS' = business month start (default),
                'B' = every business day, 'W-MON' = weekly Monday.
    """
    bt = Backtest(
        {'stocks': stock_pct, 'options': opt_pct, 'cash': 0.0},
        initial_capital=initial_capital,
    )
    if budget_fn is not None:
        bt.options_budget = budget_fn
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = data['stocks_data']
    bt.options_strategy = strategy_fn()
    bt.options_data = data['options_data']
    bt.run(rebalance_freq=rebal_months, rebalance_unit=rebal_unit)

    balance = bt.balance
    total_cap = balance['total capital']
    total_ret = (balance['accumulated return'].iloc[-1] - 1) * 100
    years = data['years']
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    cummax = total_cap.cummax()
    drawdown = (total_cap - cummax) / cummax
    max_dd = drawdown.min() * 100

    return {
        'name': name,
        'stock_pct': stock_pct,
        'opt_pct': opt_pct,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'trades': len(bt.trade_log),
        'excess_annual': annual_ret - data['spy_annual_ret'],
        'balance': balance,
        'drawdown': drawdown,
        'trade_log': bt.trade_log,
    }


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
def print_results_table(
    results: list[dict[str, Any]],
    spy_annual: float,
    spy_total: float | None = None,
    spy_dd: float | None = None,
    title: str = 'Results',
) -> None:
    """Pretty-print a comparison table to the console."""
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    print(f"{'Config':<30} {'Stk%':>5} {'Opt%':>5} {'Annual%':>9} {'Total%':>10} "
          f"{'MaxDD%':>8} {'Trades':>7} {'Excess/yr':>10}")
    print("-" * 110)

    if spy_total is not None and spy_dd is not None:
        print(f"{'SPY Buy & Hold':<30} {'100':>5} {'0':>5} {spy_annual:>8.2f}% "
              f"{spy_total:>9.1f}% {spy_dd:>7.1f}%")
        print("-" * 110)

    for r in results:
        marker = " ***" if r['excess_annual'] > 0 else ""
        print(f"{r['name']:<30} {r['stock_pct']*100:>4.1f}% {r['opt_pct']*100:>4.1f}% "
              f"{r['annual_ret']:>8.2f}% {r['total_ret']:>9.1f}% "
              f"{r['max_dd']:>7.1f}% {r['trades']:>7} {r['excess_annual']:>+9.2f}%{marker}")

    print("=" * 110)


# ---------------------------------------------------------------------------
# 4-panel chart
# ---------------------------------------------------------------------------
def plot_results(
    results: list[dict[str, Any]],
    spy_prices: pd.Series,
    title: str,
    filename: str,
    initial_capital: int = INITIAL_CAPITAL,
) -> None:
    """Generate a 4-panel PNG: capital curves, drawdowns, annual return scatter, max DD scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=11)

    spy_norm = spy_prices / spy_prices.iloc[0] * initial_capital
    spy_cummax = spy_prices.cummax()
    spy_dd_s = (spy_prices - spy_cummax) / spy_cummax * 100

    colors = plt.colormaps['tab10'](np.linspace(0, 1, max(len(results), 10)))

    # Top-left: capital curves
    ax = axes[0, 0]
    ax.plot(spy_norm.index, spy_norm.values, 'k--', linewidth=2, label='SPY B&H', alpha=0.7)
    for r, c in zip(results, colors):
        r['balance']['total capital'].plot(ax=ax, label=r['name'], color=c, alpha=0.8)
    ax.set_title('Total Capital')
    ax.set_ylabel('$')
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(fontsize=6, loc='upper left')

    # Top-right: drawdowns
    ax = axes[0, 1]
    ax.plot(spy_dd_s.index, spy_dd_s.values, 'k--', linewidth=2, label='SPY B&H', alpha=0.7)
    for r, c in zip(results, colors):
        (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8)
    ax.set_title('Drawdown')
    ax.set_ylabel('% from peak')
    ax.legend(fontsize=6, loc='lower left')

    # Bottom-left: annual return bar
    ax = axes[1, 0]
    names = [r['name'] for r in results]
    annual_rets = [r['annual_ret'] for r in results]
    bar_colors = ['green' if r['excess_annual'] > 0 else 'salmon' for r in results]
    ax.barh(names, annual_rets, color=bar_colors)
    spy_annual = results[0]['annual_ret'] - results[0]['excess_annual']  # recover from excess
    ax.axvline(x=spy_annual, color='black', linestyle='--', alpha=0.5, label='SPY B&H')
    ax.set_xlabel('Annual Return (%)')
    ax.set_title('Annual Return Comparison')
    ax.legend(fontsize=8)

    # Bottom-right: risk/return scatter
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        ax.scatter(abs(r['max_dd']), r['annual_ret'], color=c, s=80, label=r['name'], zorder=3)
    ax.set_xlabel('Max Drawdown (%, abs)')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Risk / Return')
    ax.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nSaved chart to {filename}")
