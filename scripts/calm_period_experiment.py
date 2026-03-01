#!/usr/bin/env python3
"""Calm-period experiment: 2012-2018 subperiod analysis.

Compares SPY-only, Spitznagel (leveraged), and No-leverage framings
at 0.5%, 1.0%, and 3.3% put budgets during the calmest stretch
(no correction > ~20%).
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import (
    load_data, run_backtest, INITIAL_CAPITAL,
    make_deep_otm_put_strategy,
)

# ── Subperiod bounds ──────────────────────────────────────────────
START = pd.Timestamp('2012-01-01')
END   = pd.Timestamp('2019-01-01')   # up to end of 2018

# ── Load data & run all configs on full period ────────────────────
data = load_data()
schema = data['schema']
spy_prices = data['spy_prices']

configs = []

# 1) SPY only (baseline)
configs.append(('SPY only', 'baseline', 1.0, 0.0, None))

# 2) Spitznagel (leveraged): 100% SPY + puts on top
for label, bp in [('0.5%', 0.005), ('1.0%', 0.01), ('3.3%', 0.033)]:
    configs.append((
        f'Spitznagel {label}',
        'spitznagel',
        1.0, 0.0,
        bp,
    ))

# 3) No-leverage: reduce equity to fund puts
for label, spct, opct in [('0.5%', 0.995, 0.005), ('1.0%', 0.99, 0.01), ('3.3%', 0.967, 0.033)]:
    configs.append((
        f'No-leverage {label}',
        'no-leverage',
        spct, opct,
        None,
    ))

results = []
for name, framing, spct, opct, budget_pct in configs:
    print(f'  Running {name}...', end=' ', flush=True)
    bfn = None
    if budget_pct is not None:
        _bp = budget_pct
        bfn = lambda date, tc, bp=_bp: tc * bp
    r = run_backtest(name, spct, opct, lambda: make_deep_otm_put_strategy(schema), data, budget_fn=bfn)
    r['framing'] = framing
    results.append(r)
    print(f'full-period annual {r["annual_ret"]:+.2f}%')


# ── Extract 2012-2018 subperiod stats ────────────────────────────
def subperiod_stats(r, spy_prices, start, end):
    total_cap = r['balance']['total capital']
    mask = (total_cap.index >= start) & (total_cap.index < end)
    cap = total_cap[mask]
    if len(cap) < 20:
        return None

    years = (cap.index[-1] - cap.index[0]).days / 365.25
    annual = ((cap.iloc[-1] / cap.iloc[0]) ** (1 / years) - 1) * 100
    dd = ((cap - cap.cummax()) / cap.cummax()).min() * 100

    # SPY subperiod
    spy_mask = (spy_prices.index >= start) & (spy_prices.index < end)
    spy = spy_prices[spy_mask]
    spy_annual = ((spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1) * 100
    spy_dd = ((spy - spy.cummax()) / spy.cummax()).min() * 100

    return {
        'Strategy': r['name'],
        'Annual %': annual,
        'vs SPY %': annual - spy_annual,
        'Max DD %': dd,
        'SPY Annual %': spy_annual,
        'SPY DD %': spy_dd,
        'Years': years,
    }


print(f'\n{"=" * 90}')
print(f'  Calm-Period Experiment: {START.strftime("%Y")} – {(END - pd.Timedelta(days=1)).strftime("%Y")}')
print(f'{"=" * 90}\n')

rows = []
for r in results:
    s = subperiod_stats(r, spy_prices, START, END)
    if s:
        rows.append(s)

df = pd.DataFrame(rows)

# Print formatted table
print(f'{"Strategy":<22} {"Annual %":>10} {"vs SPY":>10} {"Max DD %":>10}')
print('-' * 55)
for _, row in df.iterrows():
    marker = ' ***' if row['vs SPY %'] > 0 else ''
    print(f'{row["Strategy"]:<22} {row["Annual %"]:>+9.2f}% {row["vs SPY %"]:>+9.2f}% {row["Max DD %"]:>9.1f}%{marker}')

print(f'\n  SPY B&H in period: {df.iloc[0]["SPY Annual %"]:.2f}%/yr, max DD {df.iloc[0]["SPY DD %"]:.1f}%')
print(f'  Period: {df.iloc[0]["Years"]:.1f} years')
print()

# Also show year-by-year for the 0.5% Spitznagel config
print(f'\n{"=" * 90}')
print(f'  Year-by-Year: Spitznagel 0.5% vs SPY (2012-2018)')
print(f'{"=" * 90}\n')

r_spitz = results[1]  # Spitznagel 0.5%
cap = r_spitz['balance']['total capital']
cap_period = cap[(cap.index >= START) & (cap.index < END)]
yearly_strat = cap_period.resample('YE').last().pct_change().dropna() * 100

spy_period = spy_prices[(spy_prices.index >= START) & (spy_prices.index < END)]
yearly_spy = spy_period.resample('YE').last().pct_change().dropna() * 100

print(f'{"Year":<8} {"SPY %":>10} {"Strategy %":>12} {"Excess %":>10}')
print('-' * 42)
for date in yearly_strat.index:
    yr = date.year
    sr = yearly_spy.get(date, 0)
    st = yearly_strat[date]
    print(f'{yr:<8} {sr:>+9.2f}% {st:>+11.2f}% {st - sr:>+9.2f}%')
