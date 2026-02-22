#!/usr/bin/env python3
"""Comprehensive options strategy sweep: puts, calls, combined, with macro signal filters.

Sweeps:
  - OTM puts only (tail-risk hedge)
  - OTM calls only (momentum capture)
  - Both puts + calls (long strangle)
  - With and without macro signal filters (VIX, Buffett Indicator, Tobin's Q)

Budget range: 0.1% to 0.5% of capital per month (conservative, no leverage artifacts).

Signal filters (each tested independently):
  - VIX filter: buy puts only when VIX < rolling 1yr median (cheap protection)
  - Buffett filter: buy puts only when Buffett Indicator > rolling 1yr median (overvalued)
  - Tobin Q filter: buy puts only when Tobin's Q > rolling 1yr median (overvalued)
  - No filter (baseline)

Outputs:
  - Comparison table
  - 4-panel chart saved to sweep_comprehensive.png
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
BUDGET_PCTS = [0.1, 0.2, 0.5]
REBAL_MONTHS = 1

# Put parameters (same as analyze_entries_exits.py)
PUT_DELTA_MIN = -0.25
PUT_DELTA_MAX = -0.10
PUT_DTE_MIN = 60
PUT_DTE_MAX = 120
EXIT_DTE = 30

# Call parameters (symmetric OTM calls)
CALL_DELTA_MIN = 0.10
CALL_DELTA_MAX = 0.25
CALL_DTE_MIN = 60
CALL_DTE_MAX = 120

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

print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")
print(f"SPY B&H: {spy_total_ret:.1f}% total, {spy_annual_ret:.2f}% annual, {spy_dd:.1f}% max DD")

# ---------------------------------------------------------------------------
# Load macro signals
# ---------------------------------------------------------------------------
signals_path = 'data/processed/signals.csv'
has_signals = os.path.exists(signals_path)

if has_signals:
    signals_df = pd.read_csv(signals_path, parse_dates=['date'], index_col='date')
    # Compute rolling 252-day medians for each signal
    vix = signals_df['vix'].dropna()
    vix_median = vix.rolling(252, min_periods=60).median()

    buffett = signals_df.get('buffett_indicator')
    buffett_median = buffett.rolling(252, min_periods=60).median() if buffett is not None else None

    tobin = signals_df.get('tobin_q')
    tobin_median = tobin.rolling(252, min_periods=60).median() if tobin is not None else None

    print(f"Loaded macro signals: {list(signals_df.columns)}")
else:
    print("No signals.csv found — run data/fetch_signals.py first. Skipping macro filters.")
    vix = vix_median = buffett = buffett_median = tobin = tobin_median = None


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------
def make_puts_strategy():
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= PUT_DTE_MIN) & (schema.dte <= PUT_DTE_MAX) &
        (schema.delta >= PUT_DELTA_MIN) & (schema.delta <= PUT_DELTA_MAX)
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
        (schema.dte >= CALL_DTE_MIN) & (schema.dte <= CALL_DTE_MAX) &
        (schema.delta >= CALL_DELTA_MIN) & (schema.delta <= CALL_DELTA_MAX)
    )
    leg.entry_sort = ('delta', True)  # closest to ATM first for calls
    leg.exit_filter = (schema.dte <= EXIT_DTE)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------
def run_config(name, strategy_fn, budget_fn):
    bt = Backtest(
        {'stocks': 1.0, 'options': 0.0, 'cash': 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget = budget_fn
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
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'trades': len(bt.trade_log),
        'excess_annual': annual_ret - spy_annual_ret,
        'balance': balance,
        'drawdown': drawdown,
    }


# ---------------------------------------------------------------------------
# Signal filter budget functions
# ---------------------------------------------------------------------------
def plain_budget(frac):
    return lambda date, tc, f=frac: tc * f


def vix_low_budget(frac):
    """Buy when VIX < rolling median (cheap protection)."""
    def fn(date, tc, f=frac):
        if vix is None:
            return tc * f
        thresh = vix_median.asof(date) if vix_median is not None else None
        if pd.isna(thresh):
            return tc * f
        cur = vix.asof(date)
        return tc * f if (not pd.isna(cur) and cur < thresh) else 0
    return fn


def buffett_high_budget(frac):
    """Buy puts when Buffett Indicator > rolling median (overvalued market)."""
    def fn(date, tc, f=frac):
        if buffett is None:
            return tc * f
        thresh = buffett_median.asof(date) if buffett_median is not None else None
        if pd.isna(thresh):
            return tc * f
        cur = buffett.asof(date)
        return tc * f if (not pd.isna(cur) and cur > thresh) else 0
    return fn


def tobin_high_budget(frac):
    """Buy puts when Tobin's Q > rolling median (overvalued market)."""
    def fn(date, tc, f=frac):
        if tobin is None:
            return tc * f
        thresh = tobin_median.asof(date) if tobin_median is not None else None
        if pd.isna(thresh):
            return tc * f
        cur = tobin.asof(date)
        return tc * f if (not pd.isna(cur) and cur > thresh) else 0
    return fn


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
results = []

# --- Part 1: Strategy type sweep (no signal filter) ---
print("\n=== Part 1: Strategy Type Sweep (no signal filter) ===")
for pct in BUDGET_PCTS:
    frac = pct / 100.0
    budget = plain_budget(frac)

    # Puts only
    print(f"  {pct}% puts...", end=' ', flush=True)
    r = run_config(f'{pct}% puts', make_puts_strategy, budget)
    results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%")

    # Calls only
    print(f"  {pct}% calls...", end=' ', flush=True)
    r = run_config(f'{pct}% calls', make_calls_strategy, budget)
    results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%")

# --- Part 2: Signal-filtered puts (the hedge timing question) ---
if has_signals:
    print("\n=== Part 2: Signal-Filtered Puts ===")
    pct = 0.2  # Fixed budget, vary signal
    frac = pct / 100.0

    signal_configs = [
        ('0.2% puts+VIX<med', vix_low_budget(frac)),
        ('0.2% puts+Buff>med', buffett_high_budget(frac)),
        ('0.2% puts+TobQ>med', tobin_high_budget(frac)),
    ]
    for name, budget in signal_configs:
        print(f"  {name}...", end=' ', flush=True)
        r = run_config(name, make_puts_strategy, budget)
        results.append(r)
        print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, trades {r['trades']}")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 105)
print(f"{'Config':<22} {'Annual%':>9} {'Total%':>10} {'MaxDD%':>8} {'Trades':>7} {'Excess/yr':>10}")
print("-" * 105)
print(f"{'SPY Buy & Hold':<22} {spy_annual_ret:>8.2f}% {spy_total_ret:>9.1f}% {spy_dd:>7.1f}%")
print("-" * 105)
for r in results:
    marker = " ***" if r['excess_annual'] > 0 else ""
    print(f"{r['name']:<22} {r['annual_ret']:>8.2f}% {r['total_ret']:>9.1f}% "
          f"{r['max_dd']:>7.1f}% {r['trades']:>7} {r['excess_annual']:>+9.2f}%{marker}")
print("=" * 105)

beats = [r for r in results if r['excess_annual'] > 0]
if beats:
    best = max(beats, key=lambda r: r['excess_annual'])
    print(f"\nBest: {best['name']} at {best['annual_ret']:.2f}%/yr "
          f"({best['excess_annual']:+.2f}% vs SPY, {best['max_dd']:.1f}% max DD)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    'Comprehensive Sweep: Puts vs Calls vs Signal Filters\n'
    f'(delta puts [{PUT_DELTA_MIN},{PUT_DELTA_MAX}], calls [{CALL_DELTA_MIN},{CALL_DELTA_MAX}], '
    f'DTE {PUT_DTE_MIN}-{PUT_DTE_MAX}, monthly rebal)',
    fontsize=11,
)

colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 10)))
spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL

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
spy_dd_series = (spy_prices - spy_cummax) / spy_cummax * 100
ax.plot(spy_dd_series.index, spy_dd_series.values, 'k--', linewidth=2, label='SPY B&H', alpha=0.7)
for r, c in zip(results, colors):
    (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8)
ax.set_title('Drawdown')
ax.set_ylabel('% from peak')
ax.legend(fontsize=6, loc='lower left')

# Bottom-left: return vs drawdown scatter
ax = axes[1, 0]
for r, c in zip(results, colors):
    marker = 'o' if 'puts' in r['name'] and 'call' not in r['name'] else ('s' if 'call' in r['name'] else 'D')
    ax.scatter(abs(r['max_dd']), r['annual_ret'], color=c, s=80, marker=marker, zorder=5, label=r['name'])
ax.scatter(abs(spy_dd), spy_annual_ret, color='black', s=120, marker='*', zorder=5, label='SPY B&H')
ax.set_xlabel('Max Drawdown (%, abs)')
ax.set_ylabel('Annual Return (%)')
ax.set_title('Risk vs Return (o=puts, s=calls, D=signal)')
ax.legend(fontsize=5, loc='best')

# Bottom-right: excess return bar chart
ax = axes[1, 1]
names = [r['name'] for r in results]
excess = [r['excess_annual'] for r in results]
bar_colors = []
for n in names:
    if 'call' in n:
        bar_colors.append('green')
    elif '+' in n:
        bar_colors.append('darkorange')
    else:
        bar_colors.append('steelblue')
ax.bar(range(len(results)), excess, color=bar_colors, alpha=0.8)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Annual Excess vs SPY (%)')
ax.set_title('Excess Return (blue=puts, green=calls, orange=signal-filtered)')

plt.tight_layout()
plt.savefig('sweep_comprehensive.png', dpi=150)
print(f"\nSaved chart to sweep_comprehensive.png")
