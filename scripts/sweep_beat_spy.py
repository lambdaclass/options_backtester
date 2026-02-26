#!/usr/bin/env python3
"""Sweep to find tail-risk hedge configs that beat SPY buy-and-hold.

Hypothesis: premium drag is the killer. Test:
  1. Tiny fixed budgets ($500-$2K)
  2. Percentage-of-capital budgets (0.05%-0.2%)
  3. Close-to-money puts with high profit targets
  4. Quarterly rebalance (less frequent = less drag)
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from options_portfolio_backtester import BacktestEngine as Backtest, Stock, OptionType as Type, Direction
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

print("Loading data...")
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema
print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")

# SPY benchmark
sd = stocks_data._data
spy = sd[sd['symbol'] == 'SPY'].set_index('date')['adjClose']
spy_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
spy_cummax = spy.cummax()
spy_dd = ((spy - spy_cummax) / spy_cummax).min() * 100
print(f"SPY B&H: {spy_ret:.1f}% return, {spy_dd:.1f}% max drawdown\n")

VARIANTS = [
    # --- Tiny fixed budgets ---
    {'name': 'Tiny $500',     'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 500,  'rebal': 1},
    {'name': 'Tiny $1K',      'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 1000, 'rebal': 1},
    {'name': 'Tiny $2K',      'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 2000, 'rebal': 1},

    # --- Pct-of-capital budgets ---
    {'name': '0.05% capital',  'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget_pct': 0.0005, 'rebal': 1},
    {'name': '0.1% capital',   'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget_pct': 0.001,  'rebal': 1},
    {'name': '0.2% capital',   'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget_pct': 0.002,  'rebal': 1},

    # --- Close-to-money, high profit target ---
    {'name': 'ATM-ish 20x',   'delta_min': -0.30, 'delta_max': -0.15, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 1000, 'rebal': 1},
    {'name': 'ATM-ish 50x',   'delta_min': -0.30, 'delta_max': -0.15, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 14, 'profit_x': 50, 'budget': 1000, 'rebal': 1},

    # --- Long-dated LEAPS ---
    {'name': 'LEAPS $1K',     'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 270, 'dte_max': 365, 'exit_dte': 90, 'profit_x': 20, 'budget': 1000, 'rebal': 1},
    {'name': 'LEAPS $2K',     'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 270, 'dte_max': 365, 'exit_dte': 90, 'profit_x': 20, 'budget': 2000, 'rebal': 1},

    # --- Quarterly rebalance (less drag) ---
    {'name': 'Qtrly $1K',     'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 1000, 'rebal': 3},
    {'name': 'Qtrly $2K',     'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 2000, 'rebal': 3},

    # --- High profit targets, let crash run ---
    {'name': '$1K 50x exit',   'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 120, 'dte_max': 240, 'exit_dte': 14, 'profit_x': 50,  'budget': 1000, 'rebal': 1},
    {'name': '$1K 100x exit',  'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 120, 'dte_max': 240, 'exit_dte': 14, 'profit_x': 100, 'budget': 1000, 'rebal': 1},
    {'name': '$1K no exit',    'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 120, 'dte_max': 240, 'exit_dte': 14, 'profit_x': math.inf, 'budget': 1000, 'rebal': 1},
]


def run_variant(cfg):
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= cfg['dte_min']) & (schema.dte <= cfg['dte_max']) &
        (schema.delta >= cfg['delta_min']) & (schema.delta <= cfg['delta_max'])
    )
    leg.entry_sort = ('delta', False)
    leg.exit_filter = (schema.dte <= cfg['exit_dte'])

    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=cfg['profit_x'], loss_pct=math.inf)

    bt = Backtest({'stocks': 0.99, 'options': 0.01, 'cash': 0.0}, initial_capital=1_000_000)

    if 'budget_pct' in cfg:
        pct = cfg['budget_pct']
        bt.options_budget = lambda date, capital: capital * pct
    else:
        bt.options_budget = cfg['budget']

    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.run(rebalance_freq=cfg['rebal'])

    balance = bt.balance
    cummax = balance['total capital'].cummax()
    drawdown = (balance['total capital'] - cummax) / cummax

    return {
        'name': cfg['name'],
        'final': balance['total capital'].iloc[-1],
        'return': (balance['accumulated return'].iloc[-1] - 1) * 100,
        'max_dd': drawdown.min() * 100,
        'trades': len(bt.trade_log),
        'balance': balance,
        'drawdown': drawdown,
    }


results = []
for cfg in VARIANTS:
    print(f"Running {cfg['name']}...", end=' ', flush=True)
    try:
        r = run_variant(cfg)
        results.append(r)
        beat = " << BEATS SPY" if r['return'] > spy_ret else ""
        print(f"Return: {r['return']:>8.1f}%  MaxDD: {r['max_dd']:>6.1f}%  Trades: {r['trades']}{beat}")
    except Exception as e:
        print(f"FAILED: {e}")

# Summary table
print("\n" + "=" * 90)
print(f"{'Variant':<22} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'vs SPY':>10} {'DD vs SPY':>10}")
print("-" * 90)
print(f"{'SPY Buy & Hold':<22} {spy_ret:>9.1f}% {spy_dd:>7.1f}% {'---':>7} {'---':>10} {'---':>10}")
print("-" * 90)
for r in results:
    diff = r['return'] - spy_ret
    dd_diff = r['max_dd'] - spy_dd
    marker = " ***" if r['return'] > spy_ret else ""
    print(f"{r['name']:<22} {r['return']:>9.1f}% {r['max_dd']:>7.1f}% {r['trades']:>7} {diff:>+9.1f}% {dd_diff:>+9.1f}%{marker}")
print("=" * 90)

winners = [r for r in results if r['return'] > spy_ret]
if winners:
    print(f"\n*** {len(winners)} variant(s) BEAT SPY's {spy_ret:.1f}% return! ***")
    for r in winners:
        print(f"  {r['name']}: {r['return']:.1f}% return, {r['max_dd']:.1f}% max drawdown")
else:
    print(f"\nNo variant beat SPY's {spy_ret:.1f}% return.")

# Best drawdown
better_dd = [r for r in results if r['max_dd'] > spy_dd]
if better_dd:
    print(f"\nVariants with better drawdown than SPY ({spy_dd:.1f}%):")
    for r in sorted(better_dd, key=lambda x: x['max_dd'], reverse=True):
        print(f"  {r['name']}: {r['max_dd']:.1f}% drawdown, {r['return']:.1f}% return")

# Plot top 6 by return
top6 = sorted(results, key=lambda r: r['return'], reverse=True)[:6]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Beat-SPY Sweep: Top 6 Variants vs SPY Buy & Hold', fontsize=13)

spy_norm = spy / spy.iloc[0] * 1_000_000

ax = axes[0, 0]
spy_norm.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(top6, colors):
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Total Capital ($1M start)')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=8, loc='upper left')

ax = axes[0, 1]
spy_ret_series = (spy / spy.iloc[0] - 1) * 100
spy_ret_series.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(top6, colors):
    ret = r['balance']['accumulated return'].dropna() * 100 - 100
    ret.plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Accumulated Return')
ax.set_ylabel('% return')
ax.legend(fontsize=8, loc='upper left')

ax = axes[1, 0]
spy_dd_series = (spy - spy_cummax) / spy_cummax * 100
spy_dd_series.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(top6, colors):
    (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=8, loc='lower left')

ax = axes[1, 1]
names = [r['name'] for r in results]
rets = [r['return'] for r in results]
bar_colors = ['green' if r > spy_ret else ('gold' if r > 0 else 'red') for r in rets]
bars = ax.barh(names, rets, color=bar_colors, alpha=0.7)
ax.axvline(x=spy_ret, color='black', linestyle='--', linewidth=2, label=f'SPY B&H ({spy_ret:.0f}%)')
ax.set_title('Return Comparison (green = beats SPY)')
ax.set_xlabel('% return')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('beat_spy_sweep.png', dpi=150)
print(f"\nSaved chart to beat_spy_sweep.png")
