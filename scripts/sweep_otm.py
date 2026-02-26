#!/usr/bin/env python3
"""Sweep delta ranges to find the best tail-risk hedge configuration.

Tests delta bands from near-ATM (-0.30) to deep OTM (-0.02), using
entry_sort=('delta', False) to pick the deepest OTM within each band.

Fixed: 99% SPY / 1% puts, 120-180 DTE, exit at 30 DTE or 10x profit, $5K budget.
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

# Each level: (delta_min, delta_max, label)
DELTA_LEVELS = [
    (-0.30, -0.20, '-0.30 to -0.20'),
    (-0.20, -0.10, '-0.20 to -0.10'),
    (-0.15, -0.05, '-0.15 to -0.05'),
    (-0.10, -0.03, '-0.10 to -0.03'),
    (-0.05, -0.01, '-0.05 to -0.01'),
    (-0.03, -0.005, '-0.03 to -0.005'),
]


def run_delta_band(delta_min, delta_max):
    """Run backtest with puts in a given delta band."""
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= 120) & (schema.dte <= 180) &
        (schema.delta >= delta_min) & (schema.delta <= delta_max)
    )
    leg.entry_sort = ('delta', False)  # pick deepest OTM within band
    leg.exit_filter = (schema.dte <= 30)

    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=10.0, loss_pct=math.inf)

    bt = Backtest({'stocks': 0.99, 'options': 0.01, 'cash': 0.0}, initial_capital=1_000_000)
    bt.options_budget = 5000
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.run(rebalance_freq=1)

    balance = bt.balance
    cummax = balance['total capital'].cummax()
    drawdown = (balance['total capital'] - cummax) / cummax

    return {
        'delta_min': delta_min,
        'delta_max': delta_max,
        'final': balance['total capital'].iloc[-1],
        'return': (balance['accumulated return'].iloc[-1] - 1) * 100,
        'max_dd': drawdown.min() * 100,
        'trades': len(bt.trade_log),
        'balance': balance,
    }


results = []

for delta_min, delta_max, label in DELTA_LEVELS:
    print(f"\nRunning delta band {label}...")
    r = run_delta_band(delta_min, delta_max)
    r['label'] = label
    results.append(r)
    print(f"  Return: {r['return']:.1f}%  Max DD: {r['max_dd']:.1f}%  Trades: {r['trades']}")

# Summary
print("\n" + "=" * 80)
print(f"{'Delta Band':<22} {'Final Capital':>15} {'Return':>10} {'Max DD':>10} {'Trades':>8}")
print("-" * 80)
for r in results:
    print(f"{r['label']:<22} ${r['final']:>14,.0f} {r['return']:>9.1f}% {r['max_dd']:>9.1f}% {r['trades']:>8}")
print("=" * 80)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('99% SPY + OTM Puts — Delta Band Sweep (120-180 DTE, exit 30 DTE or 10x)', fontsize=14)

ax = axes[0, 0]
for r in results:
    r['balance']['total capital'].plot(ax=ax, label=r['label'], alpha=0.8)
ax.set_title('Total Capital')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=9)

ax = axes[0, 1]
for r in results:
    ret = r['balance']['accumulated return'].dropna() * 100 - 100
    ret.plot(ax=ax, label=r['label'], alpha=0.8)
ax.set_title('Accumulated Return')
ax.set_ylabel('% return')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=9)

ax = axes[1, 0]
for r in results:
    cummax = r['balance']['total capital'].cummax()
    dd = (r['balance']['total capital'] - cummax) / cummax * 100
    dd.plot(ax=ax, label=r['label'], alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=9)

ax = axes[1, 1]
returns = [r['return'] for r in results]
colors = ['green' if r > 0 else 'red' for r in returns]
bars = ax.bar([r['label'] for r in results], returns, color=colors, alpha=0.7)
ax.set_title('Final Return by Delta Band')
ax.set_ylabel('% return')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=30)
for bar, r in zip(bars, results):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + (5 if y > 0 else -15),
            f'{r["return"]:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('otm_sweep_results.png', dpi=150)
print("\nSaved chart to otm_sweep_results.png")

best = max(results, key=lambda r: r['return'])
print(f"\nBest delta band: {best['label']} — Return: {best['return']:.1f}%, Max DD: {best['max_dd']:.1f}%")
