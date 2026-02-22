#!/usr/bin/env python3
"""Sweep OTM levels to find the best tail-risk hedge configuration.

Tests: 20%, 25%, 30%, 35%, 40%, 45% OTM
Fixed: 99% SPY / 1% puts, 4mo DTE, exit at 2mo or 10x profit
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
print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")


def run_otm(otm_pct):
    """Run backtest with puts at given OTM percentage."""
    upper = 1.0 - otm_pct / 100 + 0.02  # e.g. 35% OTM -> strike <= 0.67
    lower = 1.0 - otm_pct / 100 - 0.02  # e.g. 35% OTM -> strike >= 0.63
    upper2 = lower                        # second leg slightly deeper
    lower2 = lower - 0.03

    strategy = Strategy(schema)

    leg1 = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg1.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= 100) & (schema.dte <= 130) &
        (schema.strike <= upper * schema.underlying_last) &
        (schema.strike >= lower * schema.underlying_last)
    )
    leg1.exit_filter = (schema.dte <= 60)

    leg2 = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg2.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= 100) & (schema.dte <= 130) &
        (schema.strike <= upper2 * schema.underlying_last) &
        (schema.strike >= lower2 * schema.underlying_last)
    )
    leg2.exit_filter = (schema.dte <= 60)

    strategy.add_legs([leg1, leg2])
    strategy.add_exit_thresholds(profit_pct=10.0, loss_pct=math.inf)

    bt = Backtest({'stocks': 0.99, 'options': 0.01, 'cash': 0.0}, initial_capital=1_000_000)
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.run(rebalance_freq=1)

    balance = bt.balance
    cummax = balance['total capital'].cummax()
    drawdown = (balance['total capital'] - cummax) / cummax

    return {
        'otm': otm_pct,
        'final': balance['total capital'].iloc[-1],
        'return': (balance['accumulated return'].iloc[-1] - 1) * 100,
        'max_dd': drawdown.min() * 100,
        'trades': len(bt.trade_log),
        'balance': balance,
    }


otm_levels = [20, 25, 30, 35, 40, 45]
results = []

for otm in otm_levels:
    print(f"\nRunning {otm}% OTM...")
    r = run_otm(otm)
    results.append(r)
    print(f"  Return: {r['return']:.1f}%  Max DD: {r['max_dd']:.1f}%  Trades: {r['trades']}")

# Summary
print("\n" + "=" * 70)
print(f"{'OTM %':<8} {'Final Capital':>15} {'Return':>10} {'Max DD':>10} {'Trades':>8}")
print("-" * 70)
for r in results:
    print(f"{r['otm']}%{'':<5} ${r['final']:>14,.0f} {r['return']:>9.1f}% {r['max_dd']:>9.1f}% {r['trades']:>8}")
print("=" * 70)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('99% SPY + 1% OTM Puts — OTM Level Sweep (4mo DTE, exit 2mo or 10x)', fontsize=14)

ax = axes[0, 0]
for r in results:
    r['balance']['total capital'].plot(ax=ax, label=f"{r['otm']}% OTM", alpha=0.8)
ax.set_title('Total Capital')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=9)

ax = axes[0, 1]
for r in results:
    ret = r['balance']['accumulated return'].dropna() * 100 - 100
    ret.plot(ax=ax, label=f"{r['otm']}% OTM", alpha=0.8)
ax.set_title('Accumulated Return')
ax.set_ylabel('% return')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=9)

ax = axes[1, 0]
for r in results:
    cummax = r['balance']['total capital'].cummax()
    dd = (r['balance']['total capital'] - cummax) / cummax * 100
    dd.plot(ax=ax, label=f"{r['otm']}% OTM", alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=9)

ax = axes[1, 1]
returns = [r['return'] for r in results]
colors = ['green' if r > 0 else 'red' for r in returns]
bars = ax.bar([f"{r['otm']}%" for r in results], returns, color=colors, alpha=0.7)
ax.set_title('Final Return by OTM Level')
ax.set_ylabel('% return')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, r in zip(bars, results):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + (5 if y > 0 else -15),
            f'{r["return"]:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('otm_sweep_results.png', dpi=150)
print("\nSaved chart to otm_sweep_results.png")

best = max(results, key=lambda r: r['return'])
print(f"\nBest OTM level: {best['otm']}% — Return: {best['return']:.1f}%, Max DD: {best['max_dd']:.1f}%")
