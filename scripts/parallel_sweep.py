#!/usr/bin/env python3
"""Parallel grid sweep using multiprocessing.

Runs multiple backtest configs across all CPU cores.
Each worker loads data independently to avoid pickle issues.
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

def run_single(args):
    """Worker: load data, run one config, return results."""
    name, budget_pct, dte_min, dte_max, delta_min, delta_max, exit_dte, rebal_freq, rebal_unit = args

    import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E702
    from backtest_runner import load_data, run_backtest, make_deep_otm_put_strategy

    data = load_data()
    schema = data['schema']

    r = run_backtest(
        name, 1.0, 0.0,
        lambda: make_deep_otm_put_strategy(
            schema, delta_min=delta_min, delta_max=delta_max,
            dte_min=dte_min, dte_max=dte_max, exit_dte=exit_dte),
        data,
        budget_fn=lambda date, tc, bp=budget_pct: tc * bp,
        rebal_months=rebal_freq,
        rebal_unit=rebal_unit,
    )

    daily_rets = r['balance']['% change'].dropna()
    vol = daily_rets.std() * (252 ** 0.5) * 100
    sharpe = (r['annual_ret'] - 4.0) / vol if vol > 0 else 0

    return {
        'name': name,
        'annual': r['annual_ret'],
        'excess': r['excess_annual'],
        'max_dd': r['max_dd'],
        'vol': vol,
        'sharpe': sharpe,
        'trades': r['trades'],
    }


def main():
    from itertools import product

    # Grid
    budgets = [0.003, 0.005, 0.01]
    dtes = [(90, 180), (120, 240)]
    deltas = [(-0.10, -0.02), (-0.15, -0.05)]
    exits = [14, 30, 60]

    configs = []
    for budget, (dte_min, dte_max), (d_min, d_max), exit_dte in product(budgets, dtes, deltas, exits):
        name = f'b{budget*100:.1f} DTE{dte_min}-{dte_max} d({d_min},{d_max}) ex{exit_dte}'
        configs.append((name, budget, dte_min, dte_max, d_min, d_max, exit_dte, 1, 'BMS'))

    n_cores = os.cpu_count()
    print(f'Running {len(configs)} configs on {n_cores} cores...\n')

    # Sequential timing
    start = time.perf_counter()
    seq_result = run_single(configs[0])
    one_time = time.perf_counter() - start
    print(f'One config: {one_time:.1f}s')
    print(f'Sequential estimate: {one_time * len(configs):.0f}s')
    print(f'Parallel estimate:   {one_time * len(configs) / n_cores:.0f}s\n')

    # Parallel
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        results = list(ex.map(run_single, configs))
    par_time = time.perf_counter() - start

    print(f'\nParallel: {par_time:.1f}s ({par_time/60:.1f}min)')
    print(f'Speedup vs sequential estimate: {(one_time * len(configs)) / par_time:.1f}x\n')

    # Sort by Sharpe
    results.sort(key=lambda r: r['sharpe'], reverse=True)

    print(f'{"Config":<50} {"Annual":>8} {"Excess":>8} {"MaxDD":>8} {"Sharpe":>8}')
    print('-' * 90)
    for r in results[:10]:
        print(f'{r["name"]:<50} {r["annual"]:>7.2f}% {r["excess"]:>+7.2f}% {r["max_dd"]:>7.1f}% {r["sharpe"]:>8.3f}')

    print(f'\n... {len(results)} total configs. All beat SPY: {all(r["excess"] > 0 for r in results)}')


if __name__ == '__main__':
    main()
