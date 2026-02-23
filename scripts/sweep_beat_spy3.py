#!/usr/bin/env python3
"""Sweep #3: diagnose backtester drag, then try to beat SPY.

Key finding from sweep 2: "No puts $0" with 99% stocks returned only 221%.
SPY B&H returned 555%. The backtester's rebalancing mechanism itself is the
biggest drag -- not the puts.

This sweep:
  1. Diagnose: test 100% stock alloc at various rebalance frequencies
  2. Find the backtester's pure-stock ceiling
  3. Then try hedge configs that BEAT that ceiling
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg

print("Loading data...")
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

sd = stocks_data._data
spy = sd[sd['symbol'] == 'SPY'].set_index('date')['adjClose']
spy_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
spy_cummax = spy.cummax()
spy_dd = ((spy - spy_cummax) / spy_cummax).min() * 100
print(f"SPY B&H: {spy_ret:.1f}% return, {spy_dd:.1f}% max drawdown\n")

# Dummy strategy (required by backtester but budget=0 means no trades)
def make_dummy_strategy():
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= 90) & (schema.dte <= 180) &
        (schema.delta >= -0.15) & (schema.delta <= -0.05)
    )
    leg.entry_sort = ('delta', False)
    leg.exit_filter = (schema.dte <= 30)
    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=20.0, loss_pct=math.inf)
    return strategy


def run(stock_alloc, opt_alloc, cash_alloc, budget, rebal, name, profit_x=20, delta_min=-0.15, delta_max=-0.05, dte_min=90, dte_max=180, exit_dte=30):
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= dte_min) & (schema.dte <= dte_max) &
        (schema.delta >= delta_min) & (schema.delta <= delta_max)
    )
    leg.entry_sort = ('delta', False)
    leg.exit_filter = (schema.dte <= exit_dte)
    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=profit_x, loss_pct=math.inf)

    bt = Backtest({'stocks': stock_alloc, 'options': opt_alloc, 'cash': cash_alloc}, initial_capital=1_000_000)
    bt.options_budget = budget
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.run(rebalance_freq=rebal)

    balance = bt.balance
    cummax = balance['total capital'].cummax()
    drawdown = (balance['total capital'] - cummax) / cummax

    return {
        'name': name,
        'final': balance['total capital'].iloc[-1],
        'return': (balance['accumulated return'].iloc[-1] - 1) * 100,
        'max_dd': drawdown.min() * 100,
        'trades': len(bt.trade_log),
        'balance': balance,
        'drawdown': drawdown,
    }


print("=== Part 1: Diagnose backtester stock return ===\n")

diag_configs = [
    # (stock_alloc, opt_alloc, cash_alloc, budget, rebal, name)
    (1.0, 0.0, 0.0, 0, 1,  '100% stk monthly'),
    (1.0, 0.0, 0.0, 0, 3,  '100% stk qtrly'),
    (1.0, 0.0, 0.0, 0, 6,  '100% stk 6mo'),
    (1.0, 0.0, 0.0, 0, 12, '100% stk annual'),
    (0.99, 0.01, 0.0, 0, 1, '99% stk monthly'),
    (0.99, 0.01, 0.0, 0, 3, '99% stk qtrly'),
    (0.99, 0.01, 0.0, 0, 12, '99% stk annual'),
]

diag_results = []
for sa, oa, ca, bud, reb, name in diag_configs:
    print(f"  {name:<22}", end=' ', flush=True)
    try:
        r = run(sa, oa, ca, bud, reb, name)
        diag_results.append(r)
        print(f"Return: {r['return']:>8.1f}%  MaxDD: {r['max_dd']:>7.1f}%")
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\n  SPY B&H benchmark:       Return: {spy_ret:>8.1f}%  MaxDD: {spy_dd:>7.1f}%")


print("\n\n=== Part 2: Hedge configs (using best stock baseline) ===\n")

# Use whichever stock config performs best, then add puts on top
hedge_configs = [
    # Annual rebalance, 100% stocks, vary put budget
    (1.0, 0.0, 0.0, 500,  12, 'Ann 100% $500 puts',  20, -0.15, -0.05, 120, 240, 30),
    (1.0, 0.0, 0.0, 1000, 12, 'Ann 100% $1K puts',   20, -0.15, -0.05, 120, 240, 30),
    (1.0, 0.0, 0.0, 2000, 12, 'Ann 100% $2K puts',   20, -0.15, -0.05, 120, 240, 30),
    (1.0, 0.0, 0.0, 5000, 12, 'Ann 100% $5K puts',   20, -0.15, -0.05, 120, 240, 30),

    # Annual, 100% stocks, close to money for crash payoff
    (1.0, 0.0, 0.0, 1000, 12, 'Ann ATM $1K',         20, -0.30, -0.10, 120, 240, 30),
    (1.0, 0.0, 0.0, 2000, 12, 'Ann ATM $2K',         20, -0.30, -0.10, 120, 240, 30),

    # Annual, 100% stocks, no profit cap
    (1.0, 0.0, 0.0, 1000, 12, 'Ann $1K nox',         math.inf, -0.15, -0.05, 120, 240, 14),
    (1.0, 0.0, 0.0, 2000, 12, 'Ann $2K nox',         math.inf, -0.15, -0.05, 120, 240, 14),
    (1.0, 0.0, 0.0, 1000, 12, 'Ann ATM $1K nox',     math.inf, -0.25, -0.10, 120, 240, 14),

    # 6-month, 100% stocks
    (1.0, 0.0, 0.0, 1000, 6, '6mo 100% $1K',         20, -0.15, -0.05, 90, 180, 30),
    (1.0, 0.0, 0.0, 2000, 6, '6mo 100% $2K',         20, -0.15, -0.05, 90, 180, 30),
    (1.0, 0.0, 0.0, 5000, 6, '6mo 100% $5K',         20, -0.15, -0.05, 90, 180, 30),
]

hedge_results = []
for sa, oa, ca, bud, reb, name, px, dmin, dmax, dtemin, dtemax, exdte in hedge_configs:
    print(f"  {name:<22}", end=' ', flush=True)
    try:
        r = run(sa, oa, ca, bud, reb, name, px, dmin, dmax, dtemin, dtemax, exdte)
        hedge_results.append(r)
        beat = " << BEATS BASELINE" if r['return'] > diag_results[0]['return'] else ""
        print(f"Return: {r['return']:>8.1f}%  MaxDD: {r['max_dd']:>7.1f}%  Trades: {r['trades']:>4}{beat}")
    except Exception as e:
        print(f"FAILED: {e}")

# Full summary
all_results = diag_results + hedge_results
print("\n" + "=" * 90)
print(f"{'Variant':<24} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'vs SPY':>12}")
print("-" * 90)
print(f"{'SPY Buy & Hold':<24} {spy_ret:>9.1f}% {spy_dd:>7.1f}%")
print("-" * 90)
for r in all_results:
    diff = r['return'] - spy_ret
    marker = " ***" if r['return'] > spy_ret else ""
    print(f"{r['name']:<24} {r['return']:>9.1f}% {r['max_dd']:>7.1f}% {r['trades']:>7} {diff:>+11.1f}%{marker}")
print("=" * 90)

best = max(all_results, key=lambda r: r['return'])
print(f"\nBest overall: {best['name']} at {best['return']:.1f}% (SPY: {spy_ret:.1f}%)")

winners = [r for r in all_results if r['return'] > spy_ret]
if winners:
    print(f"\n*** {len(winners)} config(s) BEAT SPY! ***")
    for r in sorted(winners, key=lambda r: r['return'], reverse=True):
        print(f"  {r['name']}: {r['return']:.1f}% return, {r['max_dd']:.1f}% drawdown")

# Plot
top6 = sorted(all_results, key=lambda r: r['return'], reverse=True)[:6]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
spy_norm = spy / spy.iloc[0] * 1_000_000

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Sweep #3: Diagnosing Backtester + Hedge Configs vs SPY', fontsize=13)

ax = axes[0, 0]
spy_norm.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(top6, colors):
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Total Capital')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=8, loc='upper left')

ax = axes[0, 1]
spy_ret_s = (spy / spy.iloc[0] - 1) * 100
spy_ret_s.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(top6, colors):
    ret = r['balance']['accumulated return'].dropna() * 100 - 100
    ret.plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Accumulated Return')
ax.set_ylabel('% return')
ax.legend(fontsize=8, loc='upper left')

ax = axes[1, 0]
spy_dd_s = (spy - spy_cummax) / spy_cummax * 100
spy_dd_s.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(top6, colors):
    (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=8, loc='lower left')

ax = axes[1, 1]
for r, c in zip(top6, colors):
    ax.scatter(abs(r['max_dd']), r['return'], color=c, s=80, zorder=5, label=r['name'])
ax.scatter(abs(spy_dd), spy_ret, color='black', s=120, marker='*', zorder=5, label='SPY B&H')
ax.set_xlabel('Max Drawdown (%, abs)')
ax.set_ylabel('Total Return (%)')
ax.set_title('Risk vs Return')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('beat_spy_sweep3.png', dpi=150)
print(f"\nSaved chart to beat_spy_sweep3.png")
