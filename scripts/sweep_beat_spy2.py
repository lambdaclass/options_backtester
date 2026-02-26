#!/usr/bin/env python3
"""Second sweep: extreme premium reduction to beat SPY.

Key insight from sweep 1: even $1K/month monthly bleeds too much.
Quarterly $1K was best at 222%. Try:
  - Semi-annual and annual rebalance
  - Budgets of $200-$1000
  - No profit cap (let crash puts run to full value)
  - Also test 100% stock baseline through backtester to see rebalancing overhead
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

sd = stocks_data._data
spy = sd[sd['symbol'] == 'SPY'].set_index('date')['adjClose']
spy_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
spy_cummax = spy.cummax()
spy_dd = ((spy - spy_cummax) / spy_cummax).min() * 100
print(f"SPY B&H: {spy_ret:.1f}% return, {spy_dd:.1f}% max drawdown\n")

VARIANTS = [
    # --- Rebalance frequency sweep with $1K ---
    {'name': 'Monthly $1K',   'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20,       'budget': 1000, 'rebal': 1},
    {'name': 'Qtrly $1K',    'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20,       'budget': 1000, 'rebal': 3},
    {'name': '6mo $1K',       'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20,       'budget': 1000, 'rebal': 6},
    {'name': 'Annual $1K',    'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 180, 'dte_max': 365, 'exit_dte': 30, 'profit_x': 20,      'budget': 1000, 'rebal': 12},

    # --- Tiny budget quarterly ---
    {'name': 'Qtrly $200',   'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20,       'budget': 200,  'rebal': 3},
    {'name': 'Qtrly $500',   'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20,       'budget': 500,  'rebal': 3},

    # --- No profit cap (let winners run to full crash value) ---
    {'name': 'Qtrly $1K nox', 'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': math.inf, 'budget': 1000, 'rebal': 3},
    {'name': '6mo $1K nox',    'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': math.inf, 'budget': 1000, 'rebal': 6},
    {'name': 'Annual $1K nox', 'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 180, 'dte_max': 365, 'exit_dte': 30, 'profit_x': math.inf, 'budget': 1000, 'rebal': 12},

    # --- Close to ATM, quarterly, no cap ---
    {'name': 'Qtrly ATM nox', 'delta_min': -0.30, 'delta_max': -0.10, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 14, 'profit_x': math.inf, 'budget': 1000, 'rebal': 3},

    # --- Minimal puts: $200 quarterly, no profit cap ---
    {'name': 'Min $200 Q nox', 'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': math.inf, 'budget': 200, 'rebal': 3},
    {'name': 'Min $500 Q nox', 'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': math.inf, 'budget': 500, 'rebal': 3},

    # --- 6-month, close to money, no cap ---
    {'name': '6mo ATM nox',    'delta_min': -0.25, 'delta_max': -0.10, 'dte_min': 120, 'dte_max': 240, 'exit_dte': 14, 'profit_x': math.inf, 'budget': 1000, 'rebal': 6},

    # --- Baseline: $0 budget (pure 99% SPY through the backtester to measure overhead) ---
    {'name': 'No puts $0',    'delta_min': -0.15, 'delta_max': -0.05, 'dte_min': 90, 'dte_max': 180, 'exit_dte': 30, 'profit_x': 20, 'budget': 0, 'rebal': 3},
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
    print(f"Running {cfg['name']:<20}", end=' ', flush=True)
    try:
        r = run_variant(cfg)
        results.append(r)
        beat = " << BEATS SPY!" if r['return'] > spy_ret else ""
        print(f"Return: {r['return']:>8.1f}%  MaxDD: {r['max_dd']:>7.1f}%  Trades: {r['trades']:>4}{beat}")
    except Exception as e:
        print(f"FAILED: {e}")

# Summary
print("\n" + "=" * 95)
print(f"{'Variant':<22} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'vs SPY ret':>12} {'vs SPY DD':>12}")
print("-" * 95)
print(f"{'SPY Buy & Hold':<22} {spy_ret:>9.1f}% {spy_dd:>7.1f}%")
print("-" * 95)
for r in results:
    diff = r['return'] - spy_ret
    dd_diff = r['max_dd'] - spy_dd
    marker = " ***" if r['return'] > spy_ret else ""
    dd_marker = " ++" if r['max_dd'] > spy_dd else ""
    print(f"{r['name']:<22} {r['return']:>9.1f}% {r['max_dd']:>7.1f}% {r['trades']:>7} {diff:>+11.1f}% {dd_diff:>+11.1f}%{marker}{dd_marker}")
print("=" * 95)

winners = [r for r in results if r['return'] > spy_ret]
if winners:
    print(f"\n*** {len(winners)} variant(s) BEAT SPY! ***")
    for r in winners:
        print(f"  {r['name']}: {r['return']:.1f}% return, {r['max_dd']:.1f}% drawdown")
else:
    print(f"\nNo variant beat SPY. Best: {max(results, key=lambda r: r['return'])['name']} "
          f"at {max(results, key=lambda r: r['return'])['return']:.1f}%")

# Plot top 6
top6 = sorted(results, key=lambda r: r['return'], reverse=True)[:6]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
spy_norm = spy / spy.iloc[0] * 1_000_000

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Beat-SPY Sweep #2: Top 6 Variants vs SPY Buy & Hold', fontsize=13)

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
plt.savefig('beat_spy_sweep2.png', dpi=150)
print(f"\nSaved chart to beat_spy_sweep2.png")
