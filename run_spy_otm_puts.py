#!/usr/bin/env python3
"""Backtest: long SPY + long deep OTM puts (tail-risk hedge).

Strategy:
  - 99% stocks (SPY), 1% deep OTM puts (~40% below underlying)
  - Entry: ~4 months DTE (100-130 DTE)
  - Exit: after ~2 months (DTE <= 60) OR profit hits 10x
  - Monthly rebalancing

Usage:
    python data/fetch_data.py all --symbols SPY --start 2008-01-01 --end 2025-12-31
    python run_spy_otm_puts.py
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg
from backtester.statistics.stats import summary

print("Loading data...")
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

print(f"Options: {len(options_data._data)} rows")
print(f"Stocks: {len(stocks_data._data)} rows")
print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")

# Strategy: long deep OTM puts on SPY
# Buy puts with strike 38-42% below underlying (centered on 40% OTM)
# Entry: ~4 months DTE (100-130 DTE)
# Exit: after ~2 months (DTE <= 60) OR profit hits 10x
strategy = Strategy(schema)

leg1 = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
leg1.entry_filter = (
    (schema.underlying == 'SPY') &
    (schema.dte >= 100) & (schema.dte <= 130) &
    (schema.strike <= 0.62 * schema.underlying_last) &
    (schema.strike >= 0.58 * schema.underlying_last)
)
leg1.exit_filter = (schema.dte <= 60)

leg2 = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=Direction.BUY)
leg2.entry_filter = (
    (schema.underlying == 'SPY') &
    (schema.dte >= 100) & (schema.dte <= 130) &
    (schema.strike <= 0.58 * schema.underlying_last) &
    (schema.strike >= 0.55 * schema.underlying_last)
)
leg2.exit_filter = (schema.dte <= 60)

strategy.add_legs([leg1, leg2])
strategy.add_exit_thresholds(profit_pct=10.0, loss_pct=math.inf)

# 99% SPY, 1% OTM puts
stocks = [Stock('SPY', 1.0)]
allocation = {'stocks': 0.99, 'options': 0.01, 'cash': 0.0}

bt = Backtest(allocation, initial_capital=1_000_000)
bt.stocks = stocks
bt.stocks_data = stocks_data
bt.options_strategy = strategy
bt.options_data = options_data

print("\nRunning backtest (monthly rebalancing)...")
bt.run(rebalance_freq=1)

balance = bt.balance
trade_log = bt.trade_log

# Max drawdown
cummax = balance['total capital'].cummax()
drawdown = (balance['total capital'] - cummax) / cummax

print("\n=== Results ===")
print(f"Initial capital:  ${bt.initial_capital:,.2f}")
print(f"Final capital:    ${balance['total capital'].iloc[-1]:,.2f}")
print(f"Total return:     {(balance['accumulated return'].iloc[-1] - 1) * 100:.2f}%")
print(f"Max drawdown:     {drawdown.min() * 100:.1f}%")
print(f"Trades executed:  {len(trade_log)}")

# Generate charts
print("\nGenerating charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('99% SPY + 1% Deep OTM Puts (40% OTM, 4mo DTE, exit at 2mo or 10x)', fontsize=13)

ax = axes[0, 0]
balance['total capital'].plot(ax=ax)
ax.set_title('Total Capital')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')

ax = axes[0, 1]
(balance['accumulated return'].dropna() * 100 - 100).plot(ax=ax)
ax.set_title('Accumulated Return')
ax.set_ylabel('% return')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax = axes[1, 0]
(drawdown * 100).plot(ax=ax, color='red', alpha=0.7)
ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')

ax = axes[1, 1]
cols = ['stocks capital', 'options capital', 'cash']
available = [c for c in cols if c in balance.columns]
balance[available].dropna().plot.area(ax=ax, alpha=0.7)
ax.set_title('Portfolio Composition')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=150)
print("Saved chart to backtest_results.png")

if not trade_log.empty:
    print("\n=== Trade Summary ===")
    try:
        s = summary(trade_log, balance)
        print(s.data.to_string())
    except Exception as e:
        print(f"Could not generate summary: {e}")

print("\nDone.")
