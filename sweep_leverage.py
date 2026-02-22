#!/usr/bin/env python3
"""Leverage sweep: what happens when you add options ON TOP of 100% stocks?

Uses the budget callable (implicit leverage) to compare:
  - 100% SPY (baseline)
  - 100% SPY + 0.1% puts budget (mild leverage)
  - 100% SPY + 0.5% puts budget
  - 100% SPY + 1% calls budget
  - 100% SPY + 2% calls budget

Key insight: leverage + calls is the winning combo because you capture
upside twice. Leverage + puts still loses because the put drag remains.

Outputs:
  - Comparison table
  - Chart saved to sweep_leverage.png
"""

from backtest_runner import (
    load_data, make_puts_strategy, make_calls_strategy,
    run_backtest, print_results_table, plot_results,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = load_data()
schema = data['schema']

# ---------------------------------------------------------------------------
# Configs: (name, stock_pct, opt_pct, strategy_fn, budget_pct)
# budget_pct creates a callable that allocates budget_pct% of total capital
# ON TOP of 100% stocks (= leverage).
# ---------------------------------------------------------------------------
CONFIGS = [
    ('100% SPY (baseline)',     1.0, 0.0, lambda: make_puts_strategy(schema), None),
    ('SPY + 0.1% puts budget',  1.0, 0.0, lambda: make_puts_strategy(schema), 0.001),
    ('SPY + 0.5% puts budget',  1.0, 0.0, lambda: make_puts_strategy(schema), 0.005),
    ('SPY + 1% calls budget',   1.0, 0.0, lambda: make_calls_strategy(schema), 0.01),
    ('SPY + 2% calls budget',   1.0, 0.0, lambda: make_calls_strategy(schema), 0.02),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
results = []
for name, s_pct, o_pct, strat_fn, budget_pct in CONFIGS:
    print(f"  {name}...", end=' ', flush=True)
    budget_fn = None
    if budget_pct is not None:
        _bp = budget_pct  # capture for closure
        budget_fn = lambda date, tc, bp=_bp: tc * bp
    r = run_backtest(name, s_pct, o_pct, strat_fn, data, budget_fn=budget_fn)
    results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, DD {r['max_dd']:.1f}%")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print_results_table(
    results,
    spy_annual=data['spy_annual_ret'],
    spy_total=data['spy_total_ret'],
    spy_dd=data['spy_dd'],
    title='Leverage Sweep: 100% SPY + Options Budget (Implicit Leverage)',
)

beats = [r for r in results if r['excess_annual'] > 0]
if beats:
    best = max(beats, key=lambda r: r['excess_annual'])
    print(f"\nBest: {best['name']} at {best['annual_ret']:.2f}%/yr "
          f"({best['excess_annual']:+.2f}% vs SPY, {best['max_dd']:.1f}% max DD)")
else:
    print("\nNo leveraged config beat SPY.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plot_results(
    results,
    data['spy_prices'],
    title='Leverage Sweep: 100% SPY + Options Budget\n'
          'Budget callable creates implicit leverage (stocks + options > 100%)',
    filename='sweep_leverage.png',
)
