"""Large-scale performance benchmark: Rust vs Python on production data.

Runs the same strategy through Rust full-loop and Python BacktestEngine on
the full SPY options dataset (24.7M rows, 4500+ trading days) with frequent
rebalancing to produce thousands of trades.

Usage:
    python scripts/benchmark_large_pipeline.py
    python scripts/benchmark_large_pipeline.py --rebalance-freq 2 --runs 3
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from options_portfolio_backtester import BacktestEngine as LegacyBacktest
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.core.types import Direction, Stock, OptionType as Type
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.engine._dispatch import use_rust
from options_portfolio_backtester.engine import _dispatch as _rust_dispatch
from options_portfolio_backtester.execution.cost_model import NoCosts


STOCKS_FILE = os.path.join(REPO_ROOT, "data", "processed", "stocks.csv")
OPTIONS_FILE = os.path.join(REPO_ROOT, "data", "processed", "options.csv")


@dataclass
class BenchResult:
    name: str
    runtime_s: float
    final_capital: float
    total_return_pct: float
    n_trades: int
    n_balance_rows: int
    dispatch_mode: str
    peak_mem_mb: float = 0.0
    per_run_times: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Large-scale Rust vs Python benchmark.")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of timing runs (default: 3).")
    p.add_argument("--rebalance-freq", type=int, default=1,
                   help="Rebalance frequency in business months (1=monthly).")
    p.add_argument("--dte-min", type=int, default=20,
                   help="Min DTE for entry filter.")
    p.add_argument("--dte-max", type=int, default=60,
                   help="Max DTE for entry filter.")
    p.add_argument("--dte-exit", type=int, default=10,
                   help="DTE threshold for exit.")
    p.add_argument("--initial-capital", type=int, default=1_000_000,
                   help="Initial capital.")
    p.add_argument("--options-pct", type=float, default=0.10,
                   help="Options allocation pct (0.10 = 10%%).")
    return p.parse_args()


def _load_data():
    print("Loading data...")
    t0 = time.perf_counter()
    stocks_data = TiingoData(STOCKS_FILE)
    options_data = HistoricalOptionsData(OPTIONS_FILE)
    load_time = time.perf_counter() - t0
    n_opt = len(options_data._data)
    n_stk = len(stocks_data._data)
    n_dates = options_data._data["quotedate"].nunique()
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  Options: {n_opt:,} rows, {n_dates:,} trading days")
    print(f"  Stocks:  {n_stk:,} rows")
    return stocks_data, options_data


def _strategy(schema, dte_min, dte_max, dte_exit):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= dte_min)
        & (schema.dte <= dte_max)
    )
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def _copy_data(stocks_data, options_data):
    """Deep-copy data handlers to avoid cross-run contamination."""
    sd = TiingoData.__new__(TiingoData)
    sd.__dict__.update(stocks_data.__dict__)
    sd._data = stocks_data._data.copy()

    od = HistoricalOptionsData.__new__(HistoricalOptionsData)
    od.__dict__.update(options_data.__dict__)
    od._data = options_data._data.copy()
    return sd, od


def run_engine(
    stocks_data, options_data, args, runs, force_python=False,
) -> BenchResult:
    """Run BacktestEngine. If force_python, temporarily disable Rust dispatch."""
    label = "Python BacktestEngine" if force_python else "Rust BacktestEngine"
    times = []
    engine = None

    for i in range(runs):
        sd, od = _copy_data(stocks_data, options_data)
        engine = BacktestEngine(
            {"stocks": 1.0 - args.options_pct, "options": args.options_pct, "cash": 0.0},
            cost_model=NoCosts(),
            initial_capital=args.initial_capital,
        )
        engine.stocks = [Stock("SPY", 1.0)]
        engine.stocks_data = sd
        engine.options_data = od
        engine.options_strategy = _strategy(od.schema, args.dte_min, args.dte_max, args.dte_exit)

        gc.collect()
        saved_rust = _rust_dispatch.RUST_AVAILABLE
        if force_python:
            _rust_dispatch.RUST_AVAILABLE = False
        try:
            t0 = time.perf_counter()
            engine.run(rebalance_freq=args.rebalance_freq)
            elapsed = time.perf_counter() - t0
        finally:
            _rust_dispatch.RUST_AVAILABLE = saved_rust
        times.append(elapsed)
        print(f"  {label} run {i+1}/{runs}: {elapsed:.3f}s")

    assert engine is not None
    mode = engine.run_metadata.get("dispatch_mode", "unknown")
    final = float(engine.balance["total capital"].iloc[-1])
    n_trades = len(engine.trade_log) if not engine.trade_log.empty else 0
    total_ret = (final / args.initial_capital - 1.0) * 100.0
    return BenchResult(
        name=label,
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=n_trades,
        n_balance_rows=len(engine.balance),
        dispatch_mode=mode,
        per_run_times=times,
    )


def run_legacy(stocks_data, options_data, args, runs) -> BenchResult:
    """Run legacy Backtest class."""
    times = []
    bt = None

    for i in range(runs):
        sd, od = _copy_data(stocks_data, options_data)
        bt = LegacyBacktest(
            {"stocks": 1.0 - args.options_pct, "options": args.options_pct, "cash": 0.0},
            initial_capital=args.initial_capital,
        )
        bt.stocks = [Stock("SPY", 1.0)]
        bt.stocks_data = sd
        bt.options_data = od
        bt.options_strategy = _strategy(od.schema, args.dte_min, args.dte_max, args.dte_exit)

        gc.collect()
        t0 = time.perf_counter()
        bt.run(rebalance_freq=args.rebalance_freq)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Legacy Python run {i+1}/{runs}: {elapsed:.3f}s")

    assert bt is not None
    final = float(bt.balance["total capital"].iloc[-1])
    n_trades = len(bt.trade_log) if not bt.trade_log.empty else 0
    total_ret = (final / bt.initial_capital - 1.0) * 100.0
    return BenchResult(
        name="Legacy Python Backtest",
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=n_trades,
        n_balance_rows=len(bt.balance),
        dispatch_mode="python-legacy",
        per_run_times=times,
    )


def print_result(r: BenchResult, indent: str = "  ") -> None:
    print(f"{indent}{r.name}")
    print(f"{indent}  dispatch:       {r.dispatch_mode}")
    print(f"{indent}  avg runtime:    {r.runtime_s:.3f}s")
    print(f"{indent}  per-run times:  [{', '.join(f'{t:.3f}s' for t in r.per_run_times)}]")
    print(f"{indent}  final capital:  ${r.final_capital:,.2f}")
    print(f"{indent}  total return:   {r.total_return_pct:.4f}%")
    print(f"{indent}  trades:         {r.n_trades:,}")
    print(f"{indent}  balance rows:   {r.n_balance_rows:,}")


def print_comparison(a: BenchResult, b: BenchResult) -> None:
    if a.runtime_s > 0:
        speedup = b.runtime_s / a.runtime_s
    else:
        speedup = float("nan")
    cap_delta = abs(a.final_capital - b.final_capital)
    ret_delta = a.total_return_pct - b.total_return_pct
    cap_pct = (cap_delta / max(a.final_capital, 1)) * 100
    print(f"  {a.name} vs {b.name}:")
    print(f"    speedup:          {speedup:.2f}x ({a.name} is {'faster' if speedup > 1 else 'slower'})")
    print(f"    capital delta:    ${cap_delta:,.2f} ({cap_pct:.4f}%)")
    print(f"    return delta:     {ret_delta:+.4f} pct-pts")
    if a.n_trades > 0 and b.n_trades > 0:
        print(f"    trade count:      {a.n_trades:,} vs {b.n_trades:,} ({'match' if a.n_trades == b.n_trades else 'MISMATCH'})")


def main() -> None:
    args = parse_args()

    for f in (STOCKS_FILE, OPTIONS_FILE):
        if not Path(f).exists():
            print(f"ERROR: Missing data file: {f}")
            print("Run this benchmark from the repo root with production data in data/processed/")
            sys.exit(1)

    print(f"\n{'='*65}")
    print("Large-Scale Performance Benchmark: Rust vs Python")
    print(f"{'='*65}")
    print(f"  Rust available:    {use_rust()}")
    print(f"  runs per backend:  {args.runs}")
    print(f"  rebalance freq:    {args.rebalance_freq} BMS")
    print(f"  strategy:          BUY PUT, DTE {args.dte_min}-{args.dte_max}, exit DTE <= {args.dte_exit}")
    print(f"  allocation:        {(1-args.options_pct)*100:.0f}% stocks / {args.options_pct*100:.0f}% options")
    print(f"  initial capital:   ${args.initial_capital:,}")
    print()

    stocks_data, options_data = _load_data()
    print()

    # -- Run all backends --
    results = []

    if use_rust():
        print("Running Rust BacktestEngine...")
        rust_result = run_engine(stocks_data, options_data, args, args.runs, force_python=False)
        results.append(rust_result)
        print()

    print("Running Python BacktestEngine...")
    python_result = run_engine(stocks_data, options_data, args, args.runs, force_python=True)
    results.append(python_result)
    print()

    print("Running Legacy Python Backtest...")
    legacy_result = run_legacy(stocks_data, options_data, args, args.runs)
    results.append(legacy_result)
    print()

    # -- Report --
    print(f"{'='*65}")
    print("Results")
    print(f"{'='*65}")
    for r in results:
        print_result(r)
        print()

    print(f"{'='*65}")
    print("Comparisons")
    print(f"{'='*65}")
    if use_rust():
        print_comparison(rust_result, python_result)
        print()
        print_comparison(rust_result, legacy_result)
        print()
    print_comparison(python_result, legacy_result)
    print()

    # -- Summary table --
    print(f"{'='*65}")
    print("Summary Table")
    print(f"{'='*65}")
    rows = []
    for r in results:
        rows.append({
            "Backend": r.name,
            "Dispatch": r.dispatch_mode,
            "Avg Time (s)": f"{r.runtime_s:.3f}",
            "Trades": f"{r.n_trades:,}",
            "Final Capital": f"${r.final_capital:,.0f}",
            "Return %": f"{r.total_return_pct:.2f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if use_rust() and rust_result.runtime_s > 0:
        print(f"\n  Rust speedup over Python Engine: {python_result.runtime_s / rust_result.runtime_s:.2f}x")
        print(f"  Rust speedup over Legacy:        {legacy_result.runtime_s / rust_result.runtime_s:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
