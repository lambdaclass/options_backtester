"""Benchmark: Rust parallel_sweep vs Python sequential grid search.

This is the PRIMARY benchmark for justifying the Rust backend.
Single backtests have Pandas<->Polars conversion overhead, but
parallel_sweep amortizes that cost over N grid points and runs
all backtests on Rayon threads (no GIL, no pickle, zero-copy data).

Usage:
    python scripts/benchmark_sweep.py
    python scripts/benchmark_sweep.py --grid-sizes 10 50 100 --runs 3
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.enums import Direction, Stock, Type
from backtester.strategy import Strategy, StrategyLeg

from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.engine._dispatch import use_rust, rust
from options_portfolio_backtester.engine import _dispatch as _rust_dispatch
from options_portfolio_backtester.execution.cost_model import NoCosts

TEST_DIR = os.path.join(REPO_ROOT, "backtester", "test")
STOCKS_FILE = os.path.join(TEST_DIR, "test_data", "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "test_data", "options_data.csv")
PROD_STOCKS_FILE = os.path.join(REPO_ROOT, "data", "processed", "stocks.csv")
PROD_OPTIONS_FILE = os.path.join(REPO_ROOT, "data", "processed", "options.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Rust parallel_sweep vs Python sequential")
    p.add_argument("--grid-sizes", nargs="+", type=int, default=[5, 10, 25, 50],
                   help="Grid sizes to test (number of parameter combos)")
    p.add_argument("--runs", type=int, default=2, help="Timing runs per grid size")
    p.add_argument("--use-prod-data", action="store_true", help="Use production data")
    return p.parse_args()


def _load_data(use_prod: bool):
    if use_prod and Path(PROD_STOCKS_FILE).exists() and Path(PROD_OPTIONS_FILE).exists():
        sf, of = PROD_STOCKS_FILE, PROD_OPTIONS_FILE
    else:
        sf, of = STOCKS_FILE, OPTIONS_FILE

    stocks_data = TiingoData(sf)
    options_data = HistoricalOptionsData(of)

    if sf == STOCKS_FILE:
        stocks_data._data["adjClose"] = 10
        options_data._data.at[2, "ask"] = 1
        options_data._data.at[2, "bid"] = 0.5
        options_data._data.at[51, "ask"] = 1.5
        options_data._data.at[50, "bid"] = 0.5
        options_data._data.at[130, "bid"] = 0.5
        options_data._data.at[131, "bid"] = 1.5
        options_data._data.at[206, "bid"] = 0.5
        options_data._data.at[207, "bid"] = 1.5

    return stocks_data, options_data, sf


def _build_param_grid(n: int, underlying: str = "SPX") -> list[dict]:
    """Generate n parameter override dicts varying DTE thresholds.

    Returns list of dicts with both Rust filter strings AND raw dte values
    so both the Rust and Python paths can use the same grid.
    """
    dte_mins = np.linspace(20, 90, max(int(n**0.5), 2)).astype(int)
    dte_exits = np.linspace(5, 45, max(int(n / len(dte_mins)) + 1, 2)).astype(int)
    grid = []
    for dmin in dte_mins:
        for dex in dte_exits:
            if len(grid) >= n:
                break
            grid.append({
                "label": f"dte_min={dmin}_exit={dex}",
                "leg_entry_filters": [
                    f"(underlying == '{underlying}') & (dte >= {dmin})",
                ],
                "leg_exit_filters": [
                    f"dte <= {dex}",
                ],
                # Raw params for Python path
                "_dte_min": int(dmin),
                "_dte_exit": int(dex),
                "_underlying": underlying,
            })
        if len(grid) >= n:
            break
    return grid[:n]


def _build_rust_config(stocks_data, options_data, stocks, underlying="SPX"):
    """Build the config dict for rust.parallel_sweep."""
    schema = options_data.schema
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == underlying) & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])

    date_fmt = "%Y-%m-%d %H:%M:%S"
    dates_df = (
        pd.DataFrame(options_data._data[["quotedate", "volume"]])
        .drop_duplicates("quotedate")
        .set_index("quotedate")
    )
    rebalancing_days = pd.to_datetime(
        dates_df.groupby(pd.Grouper(freq="1BMS"))
        .apply(lambda x: x.index.min())
        .values
    )
    rb_dates = [d.strftime(date_fmt) for d in rebalancing_days]

    config = {
        "allocation": {"stocks": 0.97, "options": 0.03, "cash": 0.0},
        "initial_capital": 1_000_000.0,
        "shares_per_contract": 100,
        "rebalance_dates": rb_dates,
        "legs": [{
            "name": leg.name,
            "entry_filter": leg.entry_filter.query,
            "exit_filter": leg.exit_filter.query,
            "direction": leg.direction.value,
            "type": leg.type.value,
            "entry_sort_col": None,
            "entry_sort_asc": True,
        }],
        "profit_pct": None,
        "loss_pct": None,
        "stocks": [(s.symbol, s.percentage) for s in stocks],
    }

    stocks_schema = stocks_data.schema
    opts_schema = options_data.schema
    schema_mapping = {
        "contract": opts_schema["contract"],
        "date": opts_schema["date"],
        "stocks_date": stocks_schema["date"],
        "stocks_symbol": stocks_schema["symbol"],
        "stocks_price": stocks_schema["adjClose"],
        "underlying": opts_schema["underlying"],
        "expiration": opts_schema["expiration"],
        "type": opts_schema["type"],
        "strike": opts_schema["strike"],
    }

    # Convert datetime columns
    opts_copy = options_data._data.copy()
    for c in [opts_schema["date"], opts_schema["expiration"]]:
        if c in opts_copy.columns and pd.api.types.is_datetime64_any_dtype(opts_copy[c]):
            opts_copy[c] = opts_copy[c].dt.strftime(date_fmt)

    stocks_copy = stocks_data._data.copy()
    sc = stocks_schema["date"]
    if sc in stocks_copy.columns and pd.api.types.is_datetime64_any_dtype(stocks_copy[sc]):
        stocks_copy[sc] = stocks_copy[sc].dt.strftime(date_fmt)

    opts_pl = pl.from_pandas(opts_copy)
    stocks_pl = pl.from_pandas(stocks_copy)

    return config, schema_mapping, opts_pl, stocks_pl, strat


def run_rust_sweep(opts_pl, stocks_pl, config, schema_mapping, param_grid, runs):
    """Run Rust parallel_sweep."""
    times = []
    last_results = None
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        results = rust.parallel_sweep(
            opts_pl, stocks_pl, config, schema_mapping, param_grid, None,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_results = results
    return times, last_results


def run_python_sequential(stocks_data, options_data, stocks, param_grid, runs, underlying="SPX"):
    """Run sequential Python backtests for same grid."""
    times = []
    last_results = None
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        results = []
        for params in param_grid:
            sd = TiingoData.__new__(TiingoData)
            sd.__dict__.update(stocks_data.__dict__)
            sd._data = stocks_data._data.copy()

            od = HistoricalOptionsData.__new__(HistoricalOptionsData)
            od.__dict__.update(options_data.__dict__)
            od._data = options_data._data.copy()

            schema = od.schema
            strat = Strategy(schema)
            leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)

            # Construct filters from raw DTE params
            dte_min = params.get("_dte_min", 60)
            dte_exit = params.get("_dte_exit", 30)
            und = params.get("_underlying", underlying)
            leg.entry_filter = (schema.underlying == und) & (schema.dte >= dte_min)
            leg.exit_filter = schema.dte <= dte_exit
            strat.add_legs([leg])

            engine = BacktestEngine(
                {"stocks": 0.97, "options": 0.03, "cash": 0},
                cost_model=NoCosts(),
            )
            engine.stocks = stocks
            engine.stocks_data = sd
            engine.options_data = od
            engine.options_strategy = strat

            saved = _rust_dispatch.RUST_AVAILABLE
            _rust_dispatch.RUST_AVAILABLE = False
            try:
                engine.run(rebalance_freq=1)
            finally:
                _rust_dispatch.RUST_AVAILABLE = saved

            final = float(engine.balance["total capital"].iloc[-1])
            n_trades = len(engine.trade_log) if not engine.trade_log.empty else 0
            results.append({
                "label": params.get("label", ""),
                "final_capital": final,
                "total_trades": n_trades,
            })

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_results = results
    return times, last_results


def run_rust_single(opts_pl, stocks_pl, config, schema_mapping, runs):
    """Run a single Rust backtest (for overhead measurement)."""
    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        rust.run_backtest_py(opts_pl, stocks_pl, config, schema_mapping)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return times


def run_python_single(stocks_data, options_data, stocks, runs, underlying="SPX"):
    """Run a single Python backtest."""
    times = []
    for _ in range(runs):
        sd = TiingoData.__new__(TiingoData)
        sd.__dict__.update(stocks_data.__dict__)
        sd._data = stocks_data._data.copy()

        od = HistoricalOptionsData.__new__(HistoricalOptionsData)
        od.__dict__.update(options_data.__dict__)
        od._data = options_data._data.copy()

        engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
        )
        engine.stocks = stocks
        engine.stocks_data = sd
        engine.options_data = od
        schema = od.schema
        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == underlying) & (schema.dte >= 60)
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])
        engine.options_strategy = strat

        saved = _rust_dispatch.RUST_AVAILABLE
        _rust_dispatch.RUST_AVAILABLE = False
        try:
            gc.collect()
            t0 = time.perf_counter()
            engine.run(rebalance_freq=1)
            elapsed = time.perf_counter() - t0
        finally:
            _rust_dispatch.RUST_AVAILABLE = saved
        times.append(elapsed)
    return times


def main():
    args = parse_args()

    if not use_rust():
        print("ERROR: Rust extension not available. Build with: make rust-build")
        sys.exit(1)

    stocks_data, options_data, sf = _load_data(args.use_prod_data)
    if args.use_prod_data and Path(PROD_STOCKS_FILE).exists():
        stocks = [Stock("SPY", 1.0)]
    else:
        stocks = [Stock("VTI", 0.2), Stock("VEU", 0.2), Stock("BND", 0.2),
                  Stock("VNQ", 0.2), Stock("DBC", 0.2)]
    underlying = "SPY" if args.use_prod_data else "SPX"
    n_rows = len(options_data._data)
    n_dates = options_data._data["quotedate"].nunique()

    print(f"\n{'='*65}")
    print("Benchmark: Rust parallel_sweep vs Python sequential")
    print(f"{'='*65}")
    print(f"  Data: {'production' if args.use_prod_data else 'test'} ({n_rows:,} options rows, {n_dates} dates)")
    print(f"  Underlying: {underlying}")
    print(f"  Grid sizes: {args.grid_sizes}")
    print(f"  Runs per test: {args.runs}")
    print(f"  CPU cores: {os.cpu_count()}")
    print()

    # Build Rust config once (amortized over all grid sizes)
    config, schema_mapping, opts_pl, stocks_pl, strat = _build_rust_config(
        stocks_data, options_data, stocks, underlying=underlying
    )

    # -- Single backtest comparison --
    print("--- Single Backtest (1 run) ---")
    rust_single = run_rust_single(opts_pl, stocks_pl, config, schema_mapping, args.runs)
    python_single = run_python_single(stocks_data, options_data, stocks, args.runs, underlying=underlying)
    rust_avg = np.mean(rust_single)
    py_avg = np.mean(python_single)
    print(f"  Rust single:    {rust_avg:.4f}s (per-run: [{', '.join(f'{t:.4f}s' for t in rust_single)}])")
    print(f"  Python single:  {py_avg:.4f}s (per-run: [{', '.join(f'{t:.4f}s' for t in python_single)}])")
    print(f"  Speedup:        {py_avg/rust_avg:.2f}x {'(Rust faster)' if rust_avg < py_avg else '(Python faster)'}")
    print()

    # -- Grid sweep comparison --
    print("--- Grid Sweep (N parallel Rust vs N sequential Python) ---")
    rows = []
    for grid_size in args.grid_sizes:
        param_grid = _build_param_grid(grid_size, underlying=underlying)

        print(f"\n  Grid size: {grid_size}")

        # Rust parallel_sweep
        rust_times, rust_results = run_rust_sweep(
            opts_pl, stocks_pl, config, schema_mapping, param_grid, args.runs
        )
        rust_avg = np.mean(rust_times)
        print(f"    Rust parallel:  {rust_avg:.4f}s (per-run: [{', '.join(f'{t:.4f}s' for t in rust_times)}])")

        # Python sequential
        python_times, python_results = run_python_sequential(
            stocks_data, options_data, stocks, param_grid, args.runs, underlying=underlying
        )
        py_avg = np.mean(python_times)
        print(f"    Python seq:     {py_avg:.4f}s (per-run: [{', '.join(f'{t:.4f}s' for t in python_times)}])")

        speedup = py_avg / rust_avg if rust_avg > 0 else float("nan")
        throughput_rust = grid_size / rust_avg if rust_avg > 0 else 0
        throughput_py = grid_size / py_avg if py_avg > 0 else 0
        print(f"    Speedup:        {speedup:.2f}x {'(Rust faster)' if speedup > 1 else '(Python faster)'}")
        print(f"    Throughput:     Rust={throughput_rust:.1f}/s, Python={throughput_py:.1f}/s")

        rows.append({
            "Grid": grid_size,
            "Rust (s)": f"{rust_avg:.4f}",
            "Python (s)": f"{py_avg:.4f}",
            "Speedup": f"{speedup:.2f}x",
            "Rust runs/s": f"{throughput_rust:.1f}",
            "Python runs/s": f"{throughput_py:.1f}",
        })

    # -- Summary Table --
    print(f"\n{'='*65}")
    print("Summary")
    print(f"{'='*65}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print(f"\n{'='*65}")
    print("Conclusion")
    print(f"{'='*65}")
    if rows:
        final_speedup = float(rows[-1]["Speedup"].replace("x", ""))
        if final_speedup > 1:
            print(f"  Rust parallel_sweep is {final_speedup:.1f}x faster for {args.grid_sizes[-1]} grid points.")
            print(f"  For optimization/grid search, Rust + Rayon provides real value.")
        else:
            print(f"  Rust is {1/final_speedup:.1f}x slower even for parallel sweep.")
            print(f"  The Pandas<->Polars conversion overhead dominates.")

    single_speedup = np.mean(python_single) / np.mean(rust_single)
    if single_speedup < 1:
        print(f"  Single backtest: Rust is {1/single_speedup:.1f}x SLOWER (conversion overhead).")
    else:
        print(f"  Single backtest: Rust is {single_speedup:.1f}x faster.")

    print("\nDone.")


if __name__ == "__main__":
    main()
