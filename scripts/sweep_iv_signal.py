#!/usr/bin/env python3
"""IV-signal-filtered budget sweep.

Sweeps budget % (0.1%-2.0%) with and without an IV filter that only buys
puts when avg put IV is below its rolling 1-year median (cheap convexity).

Outputs:
  - Comparison table: annual return, max drawdown, trades, excess vs SPY
  - 4-panel chart saved to sweep_iv_signal.png
"""

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_portfolio_backtester import BacktestEngine as Backtest, Stock, OptionType as Type, Direction
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000
BUDGET_PCTS = [0.1, 0.2, 0.5, 1.0, 2.0]

# Strategy params (same as analyze_entries_exits.py)
DELTA_MIN = -0.25
DELTA_MAX = -0.10
DTE_MIN = 60
DTE_MAX = 120
EXIT_DTE = 30
REBAL_MONTHS = 1

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema
print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")

# SPY price series
spy_prices = stocks_data._data[stocks_data._data['symbol'] == 'SPY'].set_index('date')['adjClose'].sort_index()
years = (spy_prices.index[-1] - spy_prices.index[0]).days / 365.25

# SPY buy-and-hold stats
spy_total_ret = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100
spy_annual_ret = ((1 + spy_total_ret / 100) ** (1 / years) - 1) * 100
spy_cummax = spy_prices.cummax()
spy_dd = ((spy_prices - spy_cummax) / spy_cummax).min() * 100

# ---------------------------------------------------------------------------
# Compute IV signal: rolling 252-day median of avg put IV
# ---------------------------------------------------------------------------
opts = options_data._data
put_iv = opts[opts['type'] == 'put'].groupby('quotedate')['impliedvol'].mean().sort_index()
iv_rolling_median = put_iv.rolling(252, min_periods=60).median()

print(f"IV series: {len(put_iv)} dates, rolling median from {iv_rolling_median.first_valid_index()}")
print(f"SPY B&H: {spy_total_ret:.1f}% total, {spy_annual_ret:.2f}% annual, {spy_dd:.1f}% max DD")


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------
def make_strategy():
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX) &
        (schema.delta >= DELTA_MIN) & (schema.delta <= DELTA_MAX)
    )
    leg.entry_sort = ('delta', False)
    leg.exit_filter = (schema.dte <= EXIT_DTE)
    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return strategy


def run_config(name, budget_fn):
    bt = Backtest(
        {'stocks': 1.0, 'options': 0.0, 'cash': 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget = budget_fn
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = make_strategy()
    bt.options_data = options_data
    bt.run(rebalance_freq=REBAL_MONTHS)

    balance = bt.balance
    total_cap = balance['total capital']
    total_ret = (balance['accumulated return'].iloc[-1] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    cummax = total_cap.cummax()
    drawdown = (total_cap - cummax) / cummax
    max_dd = drawdown.min() * 100

    return {
        'name': name,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'trades': len(bt.trade_log),
        'excess_annual': annual_ret - spy_annual_ret,
        'balance': balance,
        'drawdown': drawdown,
    }


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
results = []

for pct in BUDGET_PCTS:
    frac = pct / 100.0

    # Unfiltered
    print(f"  {pct}% unfiltered...", end=' ', flush=True)
    budget_fn = lambda date, tc, f=frac: tc * f
    r = run_config(f'{pct}% no filter', budget_fn)
    results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%")

    # IV-filtered: only buy when IV < rolling median (cheap puts)
    print(f"  {pct}% IV-filtered...", end=' ', flush=True)

    def make_iv_budget(f=frac):
        def iv_budget(date, tc):
            threshold = iv_rolling_median.asof(date)
            if pd.isna(threshold):
                return tc * f  # no history yet, buy normally
            current_iv = put_iv.asof(date)
            if pd.isna(current_iv):
                return 0
            if current_iv < threshold:
                return tc * f
            return 0
        return iv_budget

    r = run_config(f'{pct}% IV filter', make_iv_budget())
    results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, trades {r['trades']}")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"{'Config':<20} {'Annual%':>9} {'Total%':>10} {'MaxDD%':>8} {'Trades':>7} {'Excess/yr':>10}")
print("-" * 100)
print(f"{'SPY Buy & Hold':<20} {spy_annual_ret:>8.2f}% {spy_total_ret:>9.1f}% {spy_dd:>7.1f}%")
print("-" * 100)
for r in results:
    marker = " ***" if r['excess_annual'] > 0 else ""
    print(f"{r['name']:<20} {r['annual_ret']:>8.2f}% {r['total_ret']:>9.1f}% "
          f"{r['max_dd']:>7.1f}% {r['trades']:>7} {r['excess_annual']:>+9.2f}%{marker}")
print("=" * 100)

# Highlight best
beats = [r for r in results if r['excess_annual'] > 0]
if beats:
    best = max(beats, key=lambda r: r['excess_annual'])
    print(f"\nBest: {best['name']} at {best['annual_ret']:.2f}%/yr "
          f"({best['excess_annual']:+.2f}% vs SPY, {best['max_dd']:.1f}% max DD)")
else:
    best = max(results, key=lambda r: r['excess_annual'])
    print(f"\nNo config beat SPY. Closest: {best['name']} "
          f"({best['excess_annual']:+.2f}%/yr vs SPY)")

# Check IV filter is working
for pct in BUDGET_PCTS:
    nf = next(r for r in results if r['name'] == f'{pct}% no filter')
    iv = next(r for r in results if r['name'] == f'{pct}% IV filter')
    if iv['trades'] >= nf['trades']:
        print(f"WARNING: IV filter not reducing trades for {pct}% "
              f"({iv['trades']} >= {nf['trades']})")

# ---------------------------------------------------------------------------
# Plot: 4-panel chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    f'IV-Signal Budget Sweep: SPY + Puts '
    f'(delta [{DELTA_MIN},{DELTA_MAX}], DTE {DTE_MIN}-{DTE_MAX}, '
    f'monthly rebal, IV < rolling 252d median)',
    fontsize=12,
)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL

# Top-left: capital curves
ax = axes[0, 0]
ax.plot(spy_norm.index, spy_norm.values, color='black', linewidth=2,
        linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(results, colors):
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Total Capital')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=7, loc='upper left')

# Top-right: drawdowns
ax = axes[0, 1]
spy_dd_series = (spy_prices - spy_cummax) / spy_cummax * 100
ax.plot(spy_dd_series.index, spy_dd_series.values, color='black', linewidth=2,
        linestyle='--', label='SPY B&H', alpha=0.7)
for r, c in zip(results, colors):
    (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=7, loc='lower left')

# Bottom-left: return vs drawdown scatter
ax = axes[1, 0]
for r, c in zip(results, colors):
    marker = 's' if 'IV filter' in r['name'] else 'o'
    ax.scatter(abs(r['max_dd']), r['annual_ret'], color=c, s=80,
               marker=marker, zorder=5, label=r['name'])
ax.scatter(abs(spy_dd), spy_annual_ret, color='black', s=120,
           marker='*', zorder=5, label='SPY B&H')
ax.set_xlabel('Max Drawdown (%, abs)')
ax.set_ylabel('Annual Return (%)')
ax.set_title('Risk vs Return (squares = IV-filtered)')
ax.legend(fontsize=6, loc='best')

# Bottom-right: bar chart of annual excess vs SPY
ax = axes[1, 1]
names = [r['name'] for r in results]
excess = [r['excess_annual'] for r in results]
bar_colors = ['steelblue' if 'no filter' in n else 'darkorange' for n in names]
bars = ax.bar(range(len(results)), excess, color=bar_colors, alpha=0.8)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Annual Excess vs SPY (%)')
ax.set_title('Annual Excess Return vs SPY (blue=unfiltered, orange=IV-filtered)')

plt.tight_layout()
plt.savefig('sweep_iv_signal.png', dpi=150)
print(f"\nSaved chart to sweep_iv_signal.png")
