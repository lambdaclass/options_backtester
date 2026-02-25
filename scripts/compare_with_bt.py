"""Head-to-head comparison: options_portfolio_backtester stock-only mode vs bt.

This harness runs the same monthly stock-rebalance policy in both frameworks:
- options_portfolio_backtester (legacy Backtest with options allocation = 0)
- bt (if installed)

Outputs a small scorecard with performance and runtime metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester import Backtest
from backtester.datahandler import TiingoData
from backtester.enums import Stock


@dataclass
class RunResult:
    name: str
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    vol_annual_pct: float
    sharpe: float
    runtime_s: float
    start_date: str
    end_date: str
    n_days: int
    equity: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare options_portfolio_backtester vs bt on stock-only allocation.")
    parser.add_argument("--stocks-file", default="data/processed/stocks.csv")
    parser.add_argument("--options-file", default="data/processed/options.csv")
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols. Example: SPY or SPY,TLT,GLD")
    parser.add_argument("--weights", default=None, help="Comma-separated weights matching symbols. Defaults equal.")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-months", type=int, default=1, help="Business-month-start rebalance frequency.")
    parser.add_argument("--runs", type=int, default=3, help="Runtime averaging repeats.")
    return parser.parse_args()


def normalize_weights(symbols: list[str], raw_weights: str | None) -> list[float]:
    if raw_weights is None:
        return [1.0 / len(symbols)] * len(symbols)
    vals = [float(x) for x in raw_weights.split(",")]
    if len(vals) != len(symbols):
        raise ValueError("--weights length must match --symbols length")
    total = float(sum(vals))
    if total <= 0:
        raise ValueError("--weights must sum to > 0")
    return [v / total for v in vals]


def compute_metrics(total_capital: pd.Series) -> tuple[float, float, float, float, float]:
    total_capital = total_capital.dropna()
    if total_capital.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    rets = total_capital.pct_change().dropna()
    total_return = total_capital.iloc[-1] / total_capital.iloc[0] - 1.0
    n_years = len(total_capital) / 252.0
    cagr = (total_capital.iloc[-1] / total_capital.iloc[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    peak = total_capital.cummax()
    dd = total_capital / peak - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0

    vol = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = float((rets.mean() / rets.std(ddof=1)) * np.sqrt(252)) if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0
    return total_return, cagr, max_dd, vol, sharpe


def run_options_portfolio_backtester(
    stocks_file: str,
    symbols: list[str],
    weights: list[float],
    initial_capital: float,
    rebalance_months: int,
    runs: int,
) -> RunResult:
    stocks_data = TiingoData(stocks_file)

    stocks = [Stock(sym, w) for sym, w in zip(symbols, weights)]
    times: list[float] = []
    bt_obj = None
    for _ in range(runs):
        bt = Backtest({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=int(initial_capital))
        bt.stocks = stocks
        bt.stocks_data = stocks_data

        t0 = time.perf_counter()
        bt.run(rebalance_freq=rebalance_months, rebalance_unit="BMS")
        times.append(time.perf_counter() - t0)
        bt_obj = bt

    assert bt_obj is not None
    bal = bt_obj.balance["total capital"].dropna()
    tr, cagr, mdd, vol, sharpe = compute_metrics(bal)
    return RunResult(
        name="options_portfolio_backtester",
        total_return_pct=tr * 100.0,
        cagr_pct=cagr * 100.0,
        max_drawdown_pct=mdd * 100.0,
        vol_annual_pct=vol * 100.0,
        sharpe=sharpe,
        runtime_s=float(np.mean(times)),
        start_date=str(bal.index.min().date()),
        end_date=str(bal.index.max().date()),
        n_days=int(len(bal)),
        equity=bal,
    )


def run_bt(
    stocks_file: str,
    symbols: list[str],
    weights: list[float],
    initial_capital: float,
    runs: int,
) -> RunResult | None:
    try:
        import bt  # type: ignore
    except Exception:
        return None

    prices = pd.read_csv(stocks_file, parse_dates=["date"])
    prices = prices[prices["symbol"].isin(symbols)].copy()
    px = prices.pivot(index="date", columns="symbol", values="adjClose").sort_index().dropna()
    px = px[symbols]

    times: list[float] = []
    last_res = None
    for _ in range(runs):
        algos = [
            bt.algos.RunMonthly(),
            bt.algos.SelectThese(symbols),
            bt.algos.WeighSpecified(**{s: w for s, w in zip(symbols, weights)}),
            bt.algos.Rebalance(),
        ]
        strat = bt.Strategy("bt_monthly_rebal", algos)
        test = bt.Backtest(strat, px, initial_capital=initial_capital)

        t0 = time.perf_counter()
        last_res = bt.run(test)
        times.append(time.perf_counter() - t0)

    assert last_res is not None
    series = last_res.prices.iloc[:, 0]
    tr, cagr, mdd, vol, sharpe = compute_metrics(series)
    return RunResult(
        name="bt",
        total_return_pct=tr * 100.0,
        cagr_pct=cagr * 100.0,
        max_drawdown_pct=mdd * 100.0,
        vol_annual_pct=vol * 100.0,
        sharpe=sharpe,
        runtime_s=float(np.mean(times)),
        start_date=str(series.index.min().date()),
        end_date=str(series.index.max().date()),
        n_days=int(len(series)),
        equity=series,
    )


def print_result(r: RunResult) -> None:
    print(f"{r.name}")
    print(f"  period: {r.start_date} -> {r.end_date} ({r.n_days} rows)")
    print(f"  total_return: {r.total_return_pct:8.2f}%")
    print(f"  cagr:         {r.cagr_pct:8.2f}%")
    print(f"  max_drawdown: {r.max_drawdown_pct:8.2f}%")
    print(f"  vol_annual:   {r.vol_annual_pct:8.2f}%")
    print(f"  sharpe:       {r.sharpe:8.3f}")
    print(f"  runtime:      {r.runtime_s:8.4f}s")


def print_overlap_parity(a: RunResult, b: RunResult) -> None:
    common = a.equity.index.intersection(b.equity.index)
    if len(common) == 0:
        print("  overlap: none")
        return

    a_n = a.equity.loc[common] / a.equity.loc[common].iloc[0]
    b_n = b.equity.loc[common] / b.equity.loc[common].iloc[0]
    diff = a_n - b_n
    print(f"  overlap rows: {len(common)}")
    print(f"  overlap end delta: {float(diff.iloc[-1]):.6e}")
    print(f"  overlap max abs delta: {float(diff.abs().max()):.6e}")


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")
    weights = normalize_weights(symbols, args.weights)

    for file_path in (args.stocks_file,):
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

    ob = run_options_portfolio_backtester(
        stocks_file=args.stocks_file,
        symbols=symbols,
        weights=weights,
        initial_capital=args.initial_capital,
        rebalance_months=args.rebalance_months,
        runs=args.runs,
    )
    bt_res = run_bt(
        stocks_file=args.stocks_file,
        symbols=symbols,
        weights=weights,
        initial_capital=args.initial_capital,
        runs=args.runs,
    )

    print("\n=== Comparison Scorecard ===")
    print_result(ob)
    if bt_res is None:
        print("\nbt")
        print("  not available (module 'bt' is not installed in this environment).")
        print("  install in nix shell and rerun:")
        print("    pip install bt")
    else:
        print()
        print_result(bt_res)
        speedup = bt_res.runtime_s / ob.runtime_s if ob.runtime_s > 0 else float("nan")
        print("\nsummary")
        print(f"  speed ratio (bt / options_portfolio_backtester): {speedup:0.2f}x")
        print(
            f"  return delta (options_portfolio_backtester - bt): "
            f"{(ob.total_return_pct - bt_res.total_return_pct):0.2f} pct-pts"
        )
        print(
            f"  maxDD delta (options_portfolio_backtester - bt): "
            f"{(ob.max_drawdown_pct - bt_res.max_drawdown_pct):0.2f} pct-pts"
        )
        print("  overlap parity:")
        print_overlap_parity(ob, bt_res)


if __name__ == "__main__":
    main()
