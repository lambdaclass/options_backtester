#!/usr/bin/env python3
"""Allocation sweep: vary stock/options split with NO leverage.

Tests different stock/options allocation splits where stocks + options = 100%.
No budget callable — uses the backtester's built-in allocation system so there's
no possibility of exceeding 100% total exposure.

Also tests macro signal filters (VIX, Buffett Indicator, Tobin's Q) on the
best-performing allocation.

Outputs:
  - Comparison table
  - Chart saved to sweep_allocation.png
"""

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000
REBAL_MONTHS = 1

# Options parameters
DELTA_MIN = -0.25
DELTA_MAX = -0.10
DTE_MIN = 60
DTE_MAX = 120
EXIT_DTE = 30

# Call parameters
CALL_DELTA_MIN = 0.10
CALL_DELTA_MAX = 0.25

# Allocation splits to test (stocks%, options%)
SPLITS = [
    (1.00, 0.00),  # pure stocks baseline
    (0.999, 0.001),  # 0.1% options
    (0.998, 0.002),  # 0.2% options
    (0.995, 0.005),  # 0.5% options
    (0.99,  0.01),   # 1% options
    (0.98,  0.02),   # 2% options
    (0.95,  0.05),   # 5% options
    (0.90,  0.10),   # 10% options
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

spy_prices = stocks_data._data[stocks_data._data['symbol'] == 'SPY'].set_index('date')['adjClose'].sort_index()
years = (spy_prices.index[-1] - spy_prices.index[0]).days / 365.25

spy_total_ret = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100
spy_annual_ret = ((1 + spy_total_ret / 100) ** (1 / years) - 1) * 100
spy_cummax = spy_prices.cummax()
spy_dd = ((spy_prices - spy_cummax) / spy_cummax).min() * 100

print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date} ({years:.1f} years)")
print(f"SPY B&H: {spy_total_ret:.1f}% total, {spy_annual_ret:.2f}% annual, {spy_dd:.1f}% max DD\n")

# ---------------------------------------------------------------------------
# Load macro signals
# ---------------------------------------------------------------------------
signals_path = 'data/processed/signals.csv'
has_signals = os.path.exists(signals_path)

if has_signals:
    signals_df = pd.read_csv(signals_path, parse_dates=['date'], index_col='date')
    vix = signals_df['vix'].dropna()
    vix_median = vix.rolling(252, min_periods=60).median()
    buffett = signals_df.get('buffett_indicator')
    buffett_median = buffett.rolling(252, min_periods=60).median() if buffett is not None else None
    tobin = signals_df.get('tobin_q')
    tobin_median = tobin.rolling(252, min_periods=60).median() if tobin is not None else None
    print(f"Loaded macro signals: {list(signals_df.columns)}")
else:
    print("No signals.csv — skipping macro filters. Run data/fetch_signals.py first.")
    vix = vix_median = buffett = buffett_median = tobin = tobin_median = None


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------
def make_puts_strategy():
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX) &
        (schema.delta >= DELTA_MIN) & (schema.delta <= DELTA_MAX)
    )
    leg.entry_sort = ('delta', False)
    leg.exit_filter = (schema.dte <= EXIT_DTE)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def make_calls_strategy():
    leg = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= DTE_MIN) & (schema.dte <= DTE_MAX) &
        (schema.delta >= CALL_DELTA_MIN) & (schema.delta <= CALL_DELTA_MAX)
    )
    leg.entry_sort = ('delta', True)
    leg.exit_filter = (schema.dte <= EXIT_DTE)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


# ---------------------------------------------------------------------------
# Run helper — uses allocation system, NOT budget callable
# ---------------------------------------------------------------------------
def run_config(name, stock_pct, opt_pct, strategy_fn, signal_budget_fn=None):
    bt = Backtest(
        {'stocks': stock_pct, 'options': opt_pct, 'cash': 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    if signal_budget_fn is not None:
        bt.options_budget = signal_budget_fn
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy_fn()
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
        'stock_pct': stock_pct,
        'opt_pct': opt_pct,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'trades': len(bt.trade_log),
        'excess_annual': annual_ret - spy_annual_ret,
        'balance': balance,
        'drawdown': drawdown,
    }


# ---------------------------------------------------------------------------
# Part 1: Allocation sweep — puts
# ---------------------------------------------------------------------------
print("\n=== Part 1: Puts Allocation Sweep (stocks + options = 100%) ===")
puts_results = []
for s_pct, o_pct in SPLITS:
    label = f'{o_pct*100:.1f}% puts'
    if o_pct == 0:
        label = 'Pure stocks'
    print(f"  {s_pct*100:.1f}/{o_pct*100:.1f} ...", end=' ', flush=True)
    r = run_config(label, s_pct, o_pct, make_puts_strategy)
    puts_results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, DD {r['max_dd']:.1f}%")

# ---------------------------------------------------------------------------
# Part 2: Allocation sweep — calls
# ---------------------------------------------------------------------------
print("\n=== Part 2: Calls Allocation Sweep ===")
calls_results = []
for s_pct, o_pct in SPLITS[1:]:  # skip pure stocks (already have it)
    label = f'{o_pct*100:.1f}% calls'
    print(f"  {s_pct*100:.1f}/{o_pct*100:.1f} ...", end=' ', flush=True)
    r = run_config(label, s_pct, o_pct, make_calls_strategy)
    calls_results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, DD {r['max_dd']:.1f}%")

# ---------------------------------------------------------------------------
# Part 3: Signal-filtered puts at best allocation
# ---------------------------------------------------------------------------
signal_results = []
if has_signals:
    # Use 0.2% allocation for signal tests
    sig_s, sig_o = 0.998, 0.002
    print(f"\n=== Part 3: Signal-Filtered Puts ({sig_o*100:.1f}% alloc) ===")

    def make_signal_budget(signal_series, median_series, buy_when_above):
        """Create budget fn that returns allocation% when signal triggers, 0 otherwise."""
        alloc_amount = sig_o  # fraction of total capital
        def fn(date, tc):
            if signal_series is None or median_series is None:
                return tc * alloc_amount
            thresh = median_series.asof(date)
            if pd.isna(thresh):
                return tc * alloc_amount
            cur = signal_series.asof(date)
            if pd.isna(cur):
                return 0
            triggered = cur > thresh if buy_when_above else cur < thresh
            return tc * alloc_amount if triggered else 0
        return fn

    signal_configs = [
        ('VIX<med (cheap)', vix, vix_median, False),
        ('Buffett>med', buffett, buffett_median, True),
        ('TobinQ>med', tobin, tobin_median, True),
    ]
    for sig_name, sig_series, sig_median, above in signal_configs:
        label = f'0.2% puts {sig_name}'
        print(f"  {label}...", end=' ', flush=True)
        budget_fn = make_signal_budget(sig_series, sig_median, above)
        r = run_config(label, sig_s, sig_o, make_puts_strategy, signal_budget_fn=budget_fn)
        signal_results.append(r)
        print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, trades {r['trades']}")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
all_results = puts_results + calls_results + signal_results

print("\n" + "=" * 110)
print(f"{'Config':<25} {'Stk%':>5} {'Opt%':>5} {'Annual%':>9} {'Total%':>10} "
      f"{'MaxDD%':>8} {'Trades':>7} {'Excess/yr':>10}")
print("-" * 110)
print(f"{'SPY Buy & Hold':<25} {'100':>5} {'0':>5} {spy_annual_ret:>8.2f}% "
      f"{spy_total_ret:>9.1f}% {spy_dd:>7.1f}%")
print("-" * 110)

for section_name, section_results in [('PUTS', puts_results), ('CALLS', calls_results), ('SIGNAL-FILTERED', signal_results)]:
    if not section_results:
        continue
    print(f"--- {section_name} ---")
    for r in section_results:
        marker = " ***" if r['excess_annual'] > 0 else ""
        print(f"{r['name']:<25} {r['stock_pct']*100:>4.1f}% {r['opt_pct']*100:>4.1f}% "
              f"{r['annual_ret']:>8.2f}% {r['total_ret']:>9.1f}% "
              f"{r['max_dd']:>7.1f}% {r['trades']:>7} {r['excess_annual']:>+9.2f}%{marker}")

print("=" * 110)

beats = [r for r in all_results if r['excess_annual'] > 0]
if beats:
    best = max(beats, key=lambda r: r['excess_annual'])
    print(f"\nBest: {best['name']} at {best['annual_ret']:.2f}%/yr "
          f"({best['excess_annual']:+.2f}% vs SPY, {best['max_dd']:.1f}% max DD)")
else:
    print("\nNo config beat SPY with proper allocation constraints.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    'Allocation Sweep: Stocks + Options = 100% (No Leverage)\n'
    f'Puts: delta [{DELTA_MIN},{DELTA_MAX}], Calls: delta [{CALL_DELTA_MIN},{CALL_DELTA_MAX}], '
    f'DTE {DTE_MIN}-{DTE_MAX}, monthly rebal',
    fontsize=11,
)

spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL

# Pick top configs for readability
top_puts = sorted(puts_results, key=lambda r: r['annual_ret'], reverse=True)[:4]
top_calls = sorted(calls_results, key=lambda r: r['annual_ret'], reverse=True)[:3] if calls_results else []
top_all = top_puts + top_calls + signal_results
colors = plt.cm.tab10(np.linspace(0, 1, max(len(top_all), 10)))

# Top-left: capital curves
ax = axes[0, 0]
ax.plot(spy_norm.index, spy_norm.values, 'k--', linewidth=2, label='SPY B&H', alpha=0.7)
for r, c in zip(top_all, colors):
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Total Capital')
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=6, loc='upper left')

# Top-right: drawdowns
ax = axes[0, 1]
spy_dd_s = (spy_prices - spy_cummax) / spy_cummax * 100
ax.plot(spy_dd_s.index, spy_dd_s.values, 'k--', linewidth=2, label='SPY B&H', alpha=0.7)
for r, c in zip(top_all, colors):
    (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=6, loc='lower left')

# Bottom-left: annual return vs options allocation %
ax = axes[1, 0]
opt_pcts_p = [r['opt_pct'] * 100 for r in puts_results]
ann_rets_p = [r['annual_ret'] for r in puts_results]
ax.plot(opt_pcts_p, ann_rets_p, 'bo-', label='Puts', markersize=8)
if calls_results:
    opt_pcts_c = [r['opt_pct'] * 100 for r in calls_results]
    ann_rets_c = [r['annual_ret'] for r in calls_results]
    ax.plot(opt_pcts_c, ann_rets_c, 'gs-', label='Calls', markersize=8)
ax.axhline(y=spy_annual_ret, color='black', linestyle='--', alpha=0.5, label='SPY B&H')
ax.set_xlabel('Options Allocation (%)')
ax.set_ylabel('Annual Return (%)')
ax.set_title('Annual Return vs Options Allocation')
ax.legend(fontsize=8)

# Bottom-right: max drawdown vs options allocation %
ax = axes[1, 1]
dd_p = [abs(r['max_dd']) for r in puts_results]
ax.plot(opt_pcts_p, dd_p, 'bo-', label='Puts', markersize=8)
if calls_results:
    dd_c = [abs(r['max_dd']) for r in calls_results]
    ax.plot(opt_pcts_c, dd_c, 'gs-', label='Calls', markersize=8)
ax.axhline(y=abs(spy_dd), color='black', linestyle='--', alpha=0.5, label='SPY B&H')
ax.set_xlabel('Options Allocation (%)')
ax.set_ylabel('Max Drawdown (%, abs)')
ax.set_title('Max Drawdown vs Options Allocation')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('sweep_allocation.png', dpi=150)
print(f"\nSaved chart to sweep_allocation.png")
