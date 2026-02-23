#!/usr/bin/env python3
"""Long vol vs short vol sweep.

Compares:
  - Long vol: buy straddle (BUY call + BUY put, ATM)
  - Short vol: sell strangle (SELL call + SELL put, OTM)
  - Tested across 0.5%, 1%, 2% allocations

Key question: who wins in bull markets vs crash periods?

Outputs:
  - Comparison table
  - Chart saved to sweep_volatility.png
"""

import sys, os; sys.path.insert(0, os.path.dirname(__file__))  # noqa: E702
from backtest_runner import (
    load_data, make_straddle_strategy, make_strangle_strategy,
    run_backtest, print_results_table, plot_results,
    Direction,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = load_data()
schema = data['schema']

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
ALLOC_PCTS = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%

results = []

# Baseline
print("Running baseline...")
r = run_backtest('100% SPY', 1.0, 0.0, lambda: make_straddle_strategy(schema), data)
results.append(r)
print(f"  SPY: annual {r['annual_ret']:+.2f}%\n")

# Long vol: buy straddle
print("=== Long Vol: Buy Straddle (ATM call + put) ===")
for pct in ALLOC_PCTS:
    name = f'Long straddle {pct*100:.1f}%'
    s_pct = 1.0 - pct
    print(f"  {name}...", end=' ', flush=True)
    r = run_backtest(
        name, s_pct, pct,
        lambda: make_straddle_strategy(schema, direction=Direction.BUY),
        data,
    )
    results.append(r)
    print(f"annual {r['annual_ret']:+.2f}%, excess {r['excess_annual']:+.2f}%, DD {r['max_dd']:.1f}%")

# Short vol: sell strangle
print("\n=== Short Vol: Sell Strangle (OTM call + put) ===")
for pct in ALLOC_PCTS:
    name = f'Short strangle {pct*100:.1f}%'
    s_pct = 1.0 - pct
    print(f"  {name}...", end=' ', flush=True)
    r = run_backtest(
        name, s_pct, pct,
        lambda: make_strangle_strategy(schema, direction=Direction.SELL),
        data,
    )
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
    title='Volatility Sweep: Long Vol (Buy Straddle) vs Short Vol (Sell Strangle)',
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plot_results(
    results,
    data['spy_prices'],
    title='Volatility Sweep: Long Vol vs Short Vol\n'
          'Buy ATM straddle vs Sell OTM strangle across allocation sizes',
    filename='sweep_volatility.png',
)
