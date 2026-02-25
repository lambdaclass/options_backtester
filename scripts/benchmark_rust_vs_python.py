"""Benchmark: Rust full-loop vs Python BacktestEngine vs legacy Backtest vs bt.

Runs options backtest (with options data) through Rust and Python paths, plus
a stock-only comparison against bt if installed.

Usage:
    python scripts/benchmark_rust_vs_python.py
    python scripts/benchmark_rust_vs_python.py --runs 5 --stock-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester import Backtest as LegacyBacktest
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.enums import Direction, Stock, Type
from backtester.strategy import Strategy, StrategyLeg

from options_backtester.engine.engine import BacktestEngine
from options_backtester.engine._dispatch import use_rust
from options_backtester.engine import _dispatch as _rust_dispatch
from options_backtester.execution.cost_model import NoCosts
from options_backtester.execution.fill_model import MarketAtBidAsk
from options_backtester.execution.signal_selector import FirstMatch

TEST_DIR = os.path.join(REPO_ROOT, "backtester", "test")
STOCKS_FILE = os.path.join(TEST_DIR, "test_data", "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "test_data", "options_data.csv")
PROD_STOCKS_FILE = os.path.join(REPO_ROOT, "data", "processed", "stocks.csv")
PROD_OPTIONS_FILE = os.path.join(REPO_ROOT, "data", "processed", "options.csv")


@dataclass
class BenchResult:
    name: str
    runtime_s: float
    final_capital: float
    total_return_pct: float
    n_trades: int
    dispatch_mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Rust vs Python backtest paths.")
    p.add_argument("--runs", type=int, default=3, help="Timing averaging repeats.")
    p.add_argument("--stock-only", action="store_true", help="Also run stock-only comparison vs bt.")
    p.add_argument("--use-prod-data", action="store_true", help="Use production data files if available.")
    p.add_argument("--rebalance-freq", type=int, default=1, help="Rebalance frequency.")
    return p.parse_args()


def _stocks(use_prod: bool = False):
    if use_prod:
        return [Stock("SPY", 1.0)]
    return [Stock("VTI", 0.2), Stock("VEU", 0.2), Stock("BND", 0.2),
            Stock("VNQ", 0.2), Stock("DBC", 0.2)]


def _load_data(use_prod: bool):
    if use_prod and Path(PROD_STOCKS_FILE).exists() and Path(PROD_OPTIONS_FILE).exists():
        stocks_file, options_file = PROD_STOCKS_FILE, PROD_OPTIONS_FILE
    else:
        stocks_file, options_file = STOCKS_FILE, OPTIONS_FILE

    stocks_data = TiingoData(stocks_file)
    options_data = HistoricalOptionsData(options_file)

    if stocks_file == STOCKS_FILE:
        stocks_data._data["adjClose"] = 10
        options_data._data.at[2, "ask"] = 1
        options_data._data.at[2, "bid"] = 0.5
        options_data._data.at[51, "ask"] = 1.5
        options_data._data.at[50, "bid"] = 0.5
        options_data._data.at[130, "bid"] = 0.5
        options_data._data.at[131, "bid"] = 1.5
        options_data._data.at[206, "bid"] = 0.5
        options_data._data.at[207, "bid"] = 1.5

    return stocks_data, options_data, stocks_file


def _buy_strategy(schema):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    return strat


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_engine_python(stocks_data, options_data, stocks, rebalance_freq, runs) -> BenchResult:
    """Force Python path by temporarily disabling Rust dispatch."""
    times = []
    engine = None
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
        engine.options_strategy = _buy_strategy(od.schema)

        saved_rust = _rust_dispatch.RUST_AVAILABLE
        _rust_dispatch.RUST_AVAILABLE = False
        try:
            t0 = time.perf_counter()
            engine.run(rebalance_freq=rebalance_freq)
            times.append(time.perf_counter() - t0)
        finally:
            _rust_dispatch.RUST_AVAILABLE = saved_rust

    assert engine is not None
    final = float(engine.balance["total capital"].iloc[-1])
    n_trades = len(engine.trade_log) if not engine.trade_log.empty else 0
    total_ret = (final / engine.initial_capital - 1) * 100
    return BenchResult(
        name="Python BacktestEngine",
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=n_trades,
        dispatch_mode="python",
    )


def run_engine_rust(stocks_data, options_data, stocks, rebalance_freq, runs) -> BenchResult | None:
    """Let Rust dispatch happen naturally (default path)."""
    if not use_rust():
        return None

    times = []
    engine = None
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
        engine.options_strategy = _buy_strategy(od.schema)

        t0 = time.perf_counter()
        engine.run(rebalance_freq=rebalance_freq)
        times.append(time.perf_counter() - t0)

    assert engine is not None
    mode = engine.run_metadata.get("dispatch_mode", "unknown")
    final = float(engine.balance["total capital"].iloc[-1])
    n_trades = len(engine.trade_log) if not engine.trade_log.empty else 0
    total_ret = (final / engine.initial_capital - 1) * 100
    return BenchResult(
        name="Rust BacktestEngine",
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=n_trades,
        dispatch_mode=mode,
    )


def run_legacy_python(stocks_data, options_data, stocks, rebalance_freq, runs) -> BenchResult:
    """Legacy Backtest class."""
    times = []
    bt = None
    for _ in range(runs):
        sd = TiingoData.__new__(TiingoData)
        sd.__dict__.update(stocks_data.__dict__)
        sd._data = stocks_data._data.copy()

        od = HistoricalOptionsData.__new__(HistoricalOptionsData)
        od.__dict__.update(options_data.__dict__)
        od._data = options_data._data.copy()

        bt = LegacyBacktest({"stocks": 0.97, "options": 0.03, "cash": 0})
        bt.stocks = stocks
        bt.stocks_data = sd
        bt.options_data = od
        bt.options_strategy = _buy_strategy(od.schema)

        t0 = time.perf_counter()
        bt.run(rebalance_freq=rebalance_freq)
        times.append(time.perf_counter() - t0)

    assert bt is not None
    final = float(bt.balance["total capital"].iloc[-1])
    n_trades = len(bt.trade_log) if not bt.trade_log.empty else 0
    total_ret = (final / bt.initial_capital - 1) * 100
    return BenchResult(
        name="Legacy Python Backtest",
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=n_trades,
        dispatch_mode="python-legacy",
    )


def run_bt_stock_only(stocks_file, symbols, weights, initial_capital, runs) -> BenchResult | None:
    """bt library stock-only benchmark."""
    try:
        import bt
    except Exception:
        return None

    prices = pd.read_csv(stocks_file, parse_dates=["date"])
    prices = prices[prices["symbol"].isin(symbols)].copy()
    px = prices.pivot(index="date", columns="symbol", values="adjClose").sort_index().dropna()
    px = px[symbols]

    times = []
    last_res = None
    for _ in range(runs):
        algos = [
            bt.algos.RunMonthly(),
            bt.algos.SelectThese(symbols),
            bt.algos.WeighSpecified(**dict(zip(symbols, weights))),
            bt.algos.Rebalance(),
        ]
        strat = bt.Strategy("bench", algos)
        test = bt.Backtest(strat, px, initial_capital=initial_capital)
        t0 = time.perf_counter()
        last_res = bt.run(test)
        times.append(time.perf_counter() - t0)

    assert last_res is not None
    series = last_res.prices.iloc[:, 0]
    # bt normalizes NAV to start at initial_capital
    final = float(series.iloc[-1])
    start = float(series.iloc[0])
    total_ret = (final / start - 1) * 100
    return BenchResult(
        name="bt library",
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=0,
        dispatch_mode="bt",
    )


def run_ob_stock_only(stocks_file, symbols, weights, initial_capital, runs) -> BenchResult:
    """options_backtester stock-only benchmark."""
    stocks = [Stock(sym, w) for sym, w in zip(symbols, weights)]
    times = []
    bt_obj = None
    for _ in range(runs):
        stocks_data = TiingoData(stocks_file)
        bt_obj = LegacyBacktest({"stocks": 1.0, "options": 0.0, "cash": 0.0},
                                initial_capital=int(initial_capital))
        bt_obj.stocks = stocks
        bt_obj.stocks_data = stocks_data
        t0 = time.perf_counter()
        bt_obj.run(rebalance_freq=1, rebalance_unit="BMS")
        times.append(time.perf_counter() - t0)

    assert bt_obj is not None
    bal = bt_obj.balance["total capital"].dropna()
    final = float(bal.iloc[-1])
    total_ret = (final / initial_capital - 1) * 100
    return BenchResult(
        name="options_backtester (stock-only)",
        runtime_s=float(np.mean(times)),
        final_capital=final,
        total_return_pct=total_ret,
        n_trades=0,
        dispatch_mode="python-legacy-stock-only",
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(r: BenchResult) -> None:
    print(f"  {r.name}")
    print(f"    dispatch:     {r.dispatch_mode}")
    print(f"    runtime:      {r.runtime_s:.4f}s")
    print(f"    final_capital: {r.final_capital:,.2f}")
    print(f"    total_return: {r.total_return_pct:.4f}%")
    print(f"    n_trades:     {r.n_trades}")


def print_comparison(a: BenchResult, b: BenchResult) -> None:
    speedup = b.runtime_s / a.runtime_s if a.runtime_s > 0 else float("nan")
    cap_delta = abs(a.final_capital - b.final_capital)
    ret_delta = a.total_return_pct - b.total_return_pct
    print(f"  {a.name} vs {b.name}:")
    print(f"    speedup:       {speedup:.2f}x ({a.name} is {'faster' if speedup > 1 else 'slower'})")
    print(f"    capital delta: ${cap_delta:,.2f}")
    print(f"    return delta:  {ret_delta:+.4f} pct-pts")
    if a.n_trades > 0 and b.n_trades > 0:
        print(f"    trades match:  {a.n_trades == b.n_trades} ({a.n_trades} vs {b.n_trades})")


def main() -> None:
    args = parse_args()
    stocks_data, options_data, stocks_file = _load_data(args.use_prod_data)
    stocks = _stocks(use_prod=args.use_prod_data)

    print(f"\n{'='*60}")
    print("Benchmark: Rust vs Python vs Legacy")
    print(f"{'='*60}")
    print(f"  Rust available: {use_rust()}")
    print(f"  runs per backend: {args.runs}")
    print(f"  rebalance_freq: {args.rebalance_freq}")
    print(f"  data: {'production' if args.use_prod_data else 'test'}")
    print()

    # -- Options backtest benchmarks --
    print("--- Options Backtest (with options data) ---")
    results = []

    legacy = run_legacy_python(stocks_data, options_data, stocks, args.rebalance_freq, args.runs)
    results.append(legacy)
    print_result(legacy)

    python_engine = run_engine_python(stocks_data, options_data, stocks, args.rebalance_freq, args.runs)
    results.append(python_engine)
    print_result(python_engine)

    rust_engine = run_engine_rust(stocks_data, options_data, stocks, args.rebalance_freq, args.runs)
    if rust_engine:
        results.append(rust_engine)
        print_result(rust_engine)
    else:
        print("  Rust BacktestEngine: SKIPPED (Rust not available)")

    print()
    print("--- Comparisons ---")
    if rust_engine:
        print_comparison(rust_engine, python_engine)
        print_comparison(rust_engine, legacy)
    print_comparison(python_engine, legacy)

    # -- Stock-only benchmarks --
    if args.stock_only and Path(PROD_STOCKS_FILE).exists():
        print()
        print("--- Stock-Only Monthly Rebalance (vs bt) ---")
        symbols = ["SPY"]
        weights = [1.0]
        capital = 1_000_000.0

        ob_stock = run_ob_stock_only(PROD_STOCKS_FILE, symbols, weights, capital, args.runs)
        print_result(ob_stock)

        bt_res = run_bt_stock_only(PROD_STOCKS_FILE, symbols, weights, capital, args.runs)
        if bt_res:
            print_result(bt_res)
            print()
            print_comparison(ob_stock, bt_res)
        else:
            print("  bt: SKIPPED (not installed)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
