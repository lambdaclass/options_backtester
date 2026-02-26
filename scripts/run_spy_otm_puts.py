#!/usr/bin/env python3
"""Multi-dimensional strategy sweep for tail-risk hedge configurations.

Tests 6 strategy variants across delta range, DTE, exit DTE, profit target,
and budget to find configurations where the hedge actually reduces drawdowns.

All variants: 99% SPY, single leg, entry_sort=('delta', False) to pick
deepest OTM within range, monthly rebalance.

Usage:
    python data/fetch_data.py all --symbols SPY --start 2008-01-01 --end 2025-12-31
    python run_spy_otm_puts.py
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from options_portfolio_backtester import BacktestEngine as Backtest, Stock, OptionType as Type, Direction
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.analytics.summary import summary

VARIANTS = {
    'A: Baseline (current)': {
        'delta_min': -0.03, 'delta_max': -0.005,
        'dte_min': 60, 'dte_max': 90, 'exit_dte': 30,
        'profit_x': 5, 'budget': 5000,
        'hypothesis': 'Far OTM, short-dated',
    },
    'B: Closer to money': {
        'delta_min': -0.20, 'delta_max': -0.05,
        'dte_min': 90, 'dte_max': 180, 'exit_dte': 30,
        'profit_x': 10, 'budget': 5000,
        'hypothesis': 'Higher delta = more crash payoff',
    },
    'C: Long vega': {
        'delta_min': -0.10, 'delta_max': -0.03,
        'dte_min': 180, 'dte_max': 365, 'exit_dte': 60,
        'profit_x': 20, 'budget': 5000,
        'hypothesis': 'Long-dated for vol spike exposure',
    },
    'D: Crash catcher': {
        'delta_min': -0.15, 'delta_max': -0.05,
        'dte_min': 120, 'dte_max': 240, 'exit_dte': 14,
        'profit_x': 50, 'budget': 5000,
        'hypothesis': 'Hold through crash, let run',
    },
    'E: Moderate + 2x budget': {
        'delta_min': -0.20, 'delta_max': -0.05,
        'dte_min': 90, 'dte_max': 180, 'exit_dte': 30,
        'profit_x': 10, 'budget': 10000,
        'hypothesis': 'Same as B but 2x spend',
    },
    'F: Lottery ticket': {
        'delta_min': -0.10, 'delta_max': -0.02,
        'dte_min': 90, 'dte_max': 180, 'exit_dte': 7,
        'profit_x': 100, 'budget': 3000,
        'hypothesis': 'Tiny cost, huge upside if crash',
    },
}


def run_variant(name, cfg, options_data, stocks_data, schema):
    """Run a single strategy variant and return results dict."""
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY') &
        (schema.dte >= cfg['dte_min']) & (schema.dte <= cfg['dte_max']) &
        (schema.delta >= cfg['delta_min']) & (schema.delta <= cfg['delta_max'])
    )
    leg.entry_sort = ('delta', False)  # pick deepest OTM (closest to 0-delta)
    leg.exit_filter = (schema.dte <= cfg['exit_dte'])

    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=cfg['profit_x'], loss_pct=math.inf)

    bt = Backtest({'stocks': 0.99, 'options': 0.01, 'cash': 0.0}, initial_capital=1_000_000)
    bt.options_budget = cfg['budget']
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data

    bt.run(rebalance_freq=1)

    balance = bt.balance
    cummax = balance['total capital'].cummax()
    drawdown = (balance['total capital'] - cummax) / cummax

    return {
        'name': name,
        'cfg': cfg,
        'final': balance['total capital'].iloc[-1],
        'return': (balance['accumulated return'].iloc[-1] - 1) * 100,
        'max_dd': drawdown.min() * 100,
        'trades': len(bt.trade_log),
        'balance': balance,
        'drawdown': drawdown,
    }


def run_spy_buyhold(stocks_data):
    """Run SPY buy-and-hold as benchmark."""
    sd = stocks_data._data
    spy = sd[sd['symbol'] == 'SPY'].set_index('date')['adjClose']
    ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
    cummax = spy.cummax()
    dd = ((spy - cummax) / cummax).min() * 100
    return {'return': ret, 'max_dd': dd, 'series': spy}


def print_comparison_table(results, benchmark):
    """Print formatted comparison table."""
    hdr = f"{'Variant':<28} {'Delta':>12} {'DTE':>9} {'ExDTE':>6} {'ProfX':>6} {'$Bud':>6} {'Return':>9} {'MaxDD':>8} {'Trades':>7}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    # Benchmark row
    print(f"{'SPY Buy & Hold':<28} {'---':>12} {'---':>9} {'---':>6} {'---':>6} {'---':>6} {benchmark['return']:>8.1f}% {benchmark['max_dd']:>7.1f}% {'---':>7}")
    print("-" * len(hdr))

    for r in results:
        c = r['cfg']
        delta_str = f"{c['delta_min']:.2f}/{c['delta_max']:.3f}"
        dte_str = f"{c['dte_min']}-{c['dte_max']}"
        print(f"{r['name']:<28} {delta_str:>12} {dte_str:>9} {c['exit_dte']:>6} {c['profit_x']:>5}x ${c['budget']//1000:>4}K {r['return']:>8.1f}% {r['max_dd']:>7.1f}% {r['trades']:>7}")

    print("=" * len(hdr))

    # Highlight best drawdown protection
    best_dd = min(results, key=lambda r: abs(r['max_dd']))
    best_ret = max(results, key=lambda r: r['return'])
    print(f"\nLowest max drawdown: {best_dd['name']} ({best_dd['max_dd']:.1f}%)")
    print(f"Highest return:      {best_ret['name']} ({best_ret['return']:.1f}%)")

    # Any variant that beats SPY on drawdown?
    spy_dd = benchmark['max_dd']
    better_dd = [r for r in results if r['max_dd'] > spy_dd]
    if better_dd:
        print(f"\nVariants with better drawdown than SPY ({spy_dd:.1f}%):")
        for r in better_dd:
            print(f"  {r['name']}: {r['max_dd']:.1f}% drawdown, {r['return']:.1f}% return")
    else:
        print(f"\nNo variant beat SPY's drawdown ({spy_dd:.1f}%)")


def plot_results(results, benchmark, output_path='strategy_sweep_results.png'):
    """Generate comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Tail-Risk Hedge Strategy Sweep: 6 Variants vs SPY Buy & Hold', fontsize=13)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Normalize benchmark series to $1M start
    spy = benchmark['series']
    spy_norm = spy / spy.iloc[0] * 1_000_000

    # Top-left: Total capital
    ax = axes[0, 0]
    spy_norm.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
    for r, c in zip(results, colors):
        r['balance']['total capital'].plot(ax=ax, label=r['name'][:20], color=c, alpha=0.8)
    ax.set_title('Total Capital ($1M start)')
    ax.set_ylabel('$')
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(fontsize=7, loc='upper left')

    # Top-right: Accumulated return
    ax = axes[0, 1]
    spy_ret = (spy / spy.iloc[0] - 1) * 100
    spy_ret.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
    for r, c in zip(results, colors):
        ret = r['balance']['accumulated return'].dropna() * 100 - 100
        ret.plot(ax=ax, label=r['name'][:20], color=c, alpha=0.8)
    ax.set_title('Accumulated Return')
    ax.set_ylabel('% return')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')

    # Bottom-left: Drawdown
    ax = axes[1, 0]
    spy_cummax = spy.cummax()
    spy_dd = (spy - spy_cummax) / spy_cummax * 100
    spy_dd.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='SPY B&H', alpha=0.7)
    for r, c in zip(results, colors):
        (r['drawdown'] * 100).plot(ax=ax, label=r['name'][:20], color=c, alpha=0.8)
    ax.set_title('Drawdown')
    ax.set_ylabel('% from peak')
    ax.legend(fontsize=7, loc='lower left')

    # Bottom-right: Return vs Max Drawdown scatter
    ax = axes[1, 1]
    ax.scatter(abs(benchmark['max_dd']), benchmark['return'], color='black', s=120, marker='*', zorder=5, label='SPY B&H')
    for r, c in zip(results, colors):
        ax.scatter(abs(r['max_dd']), r['return'], color=c, s=80, zorder=5)
        ax.annotate(r['name'][:16], (abs(r['max_dd']), r['return']),
                    fontsize=7, textcoords='offset points', xytext=(5, 5))
    ax.set_title('Return vs Max Drawdown')
    ax.set_xlabel('Max Drawdown (%, absolute)')
    ax.set_ylabel('Total Return (%)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved chart to {output_path}")


if __name__ == '__main__':
    print("Loading data...")
    options_data = HistoricalOptionsData("data/processed/options.csv")
    stocks_data = TiingoData("data/processed/stocks.csv")
    schema = options_data.schema

    print(f"Options: {len(options_data._data)} rows")
    print(f"Stocks: {len(stocks_data._data)} rows")
    print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")

    # SPY buy-and-hold benchmark
    benchmark = run_spy_buyhold(stocks_data)
    print(f"\nSPY Buy & Hold: {benchmark['return']:.1f}% return, {benchmark['max_dd']:.1f}% max drawdown")

    # Run all variants
    results = []
    for name, cfg in VARIANTS.items():
        print(f"\nRunning {name}...")
        print(f"  Delta: [{cfg['delta_min']}, {cfg['delta_max']}], DTE: {cfg['dte_min']}-{cfg['dte_max']}, "
              f"Exit DTE: {cfg['exit_dte']}, Profit: {cfg['profit_x']}x, Budget: ${cfg['budget']:,}")
        r = run_variant(name, cfg, options_data, stocks_data, schema)
        results.append(r)
        print(f"  Return: {r['return']:.1f}%  Max DD: {r['max_dd']:.1f}%  Trades: {r['trades']}")

    print_comparison_table(results, benchmark)
    plot_results(results, benchmark)
    print("\nDone.")
