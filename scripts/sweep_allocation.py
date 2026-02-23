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

import pandas as pd

import sys, os; sys.path.insert(0, os.path.dirname(__file__))  # noqa: E702
from backtest_runner import (
    load_data, make_puts_strategy, make_calls_strategy,
    run_backtest, print_results_table, plot_results,
    DELTA_MIN, DELTA_MAX, CALL_DELTA_MIN, CALL_DELTA_MAX, DTE_MIN, DTE_MAX,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPLITS = [
    (1.00, 0.00),   # pure stocks baseline
    (0.999, 0.001), # 0.1% options
    (0.998, 0.002), # 0.2% options
    (0.995, 0.005), # 0.5% options
    (0.99,  0.01),  # 1% options
    (0.98,  0.02),  # 2% options
    (0.95,  0.05),  # 5% options
    (0.90,  0.10),  # 10% options
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = load_data()
schema = data['schema']

# ---------------------------------------------------------------------------
# Part 1: Allocation sweep — puts
# ---------------------------------------------------------------------------
print("\n=== Part 1: Puts Allocation Sweep (stocks + options = 100%) ===")
puts_results = []
for s_pct, o_pct in SPLITS:
    label = f'{o_pct*100:.1f}% puts' if o_pct > 0 else 'Pure stocks'
    print(f"  {s_pct*100:.1f}/{o_pct*100:.1f} ...", end=' ', flush=True)
    r = run_backtest(label, s_pct, o_pct, lambda: make_puts_strategy(schema), data)
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
    r = run_backtest(label, s_pct, o_pct, lambda: make_calls_strategy(schema), data)
    calls_results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, DD {r['max_dd']:.1f}%")

# ---------------------------------------------------------------------------
# Part 3: Signal-filtered puts at best allocation
# ---------------------------------------------------------------------------
signal_results = []
if data['signals_df'] is not None:
    sig_s, sig_o = 0.998, 0.002
    print(f"\n=== Part 3: Signal-Filtered Puts ({sig_o*100:.1f}% alloc) ===")

    def make_signal_budget(signal_series, median_series, buy_when_above):
        alloc_amount = sig_o
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
        ('VIX<med (cheap)', data['vix'], data['vix_median'], False),
        ('Buffett>med', data['buffett'], data['buffett_median'], True),
        ('TobinQ>med', data['tobin'], data['tobin_median'], True),
    ]
    for sig_name, sig_series, sig_median, above in signal_configs:
        label = f'0.2% puts {sig_name}'
        print(f"  {label}...", end=' ', flush=True)
        budget_fn = make_signal_budget(sig_series, sig_median, above)
        r = run_backtest(label, sig_s, sig_o, lambda: make_puts_strategy(schema), data, budget_fn=budget_fn)
        signal_results.append(r)
        print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, trades {r['trades']}")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
all_results = puts_results + calls_results + signal_results

for section_name, section_results in [('PUTS', puts_results), ('CALLS', calls_results), ('SIGNAL-FILTERED', signal_results)]:
    if section_results:
        print_results_table(
            section_results,
            spy_annual=data['spy_annual_ret'],
            spy_total=data['spy_total_ret'],
            spy_dd=data['spy_dd'],
            title=f'Allocation Sweep: {section_name}',
        )

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
# Pick top configs for readability
top_puts = sorted(puts_results, key=lambda r: r['annual_ret'], reverse=True)[:4]
top_calls = sorted(calls_results, key=lambda r: r['annual_ret'], reverse=True)[:3] if calls_results else []
top_all = top_puts + top_calls + signal_results

plot_results(
    top_all,
    data['spy_prices'],
    title='Allocation Sweep: Stocks + Options = 100% (No Leverage)\n'
          f'Puts: delta [{DELTA_MIN},{DELTA_MAX}], Calls: delta [{CALL_DELTA_MIN},{CALL_DELTA_MAX}], '
          f'DTE {DTE_MIN}-{DTE_MAX}, monthly rebal',
    filename='sweep_allocation.png',
)
