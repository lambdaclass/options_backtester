"""Standardized benchmark matrix for options_backtester vs bt.

Runs multiple scenarios over date ranges/rebalance frequencies and writes
a CSV scorecard with runtime and parity metrics.
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


@dataclass(frozen=True)
class Scenario:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    rebalance_months: int
    initial_capital: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark matrix vs bt.")
    p.add_argument("--stocks-file", default="data/processed/stocks.csv")
    p.add_argument("--symbols", default="SPY")
    p.add_argument("--weights", default=None)
    p.add_argument("--date-ranges", default="2008-01-01:2025-12-12,2016-01-01:2025-12-12")
    p.add_argument("--rebalance-months", default="1,3")
    p.add_argument("--initial-capitals", default="1000000")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output", default="data/processed/benchmark_matrix.csv")
    return p.parse_args()


def parse_csv_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


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


def slice_stocks_data(stocks_file: str, start: pd.Timestamp, end: pd.Timestamp) -> TiingoData:
    d = TiingoData(stocks_file)
    m = (d._data["date"] >= start) & (d._data["date"] <= end)
    d._data = d._data.loc[m].copy()
    d.start_date = d._data["date"].min()
    d.end_date = d._data["date"].max()
    return d


def run_options_backtester(
    stocks_file: str,
    symbols: list[str],
    weights: list[float],
    scenario: Scenario,
    runs: int,
) -> tuple[dict[str, float], pd.Series]:
    stocks = [Stock(sym, w) for sym, w in zip(symbols, weights)]
    runtimes = []
    last_eq = pd.Series(dtype=float)
    for _ in range(runs):
        stocks_data = slice_stocks_data(stocks_file, scenario.start, scenario.end)
        bt = Backtest({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=int(scenario.initial_capital))
        bt.stocks = stocks
        bt.stocks_data = stocks_data
        t0 = time.perf_counter()
        bt.run(rebalance_freq=scenario.rebalance_months, rebalance_unit="BMS")
        runtimes.append(time.perf_counter() - t0)
        last_eq = bt.balance["total capital"].dropna()
    tr, cagr, mdd, vol, sharpe = compute_metrics(last_eq)
    return ({
        "ob_runtime_s": float(np.mean(runtimes)),
        "ob_total_return_pct": tr * 100.0,
        "ob_cagr_pct": cagr * 100.0,
        "ob_max_drawdown_pct": mdd * 100.0,
        "ob_vol_annual_pct": vol * 100.0,
        "ob_sharpe": sharpe,
        "ob_rows": float(len(last_eq)),
    }, last_eq)


def run_bt(
    stocks_file: str,
    symbols: list[str],
    weights: list[float],
    scenario: Scenario,
    runs: int,
) -> tuple[dict[str, float], pd.Series | None]:
    try:
        import bt  # type: ignore
    except Exception:
        return ({"bt_available": 0.0}, None)

    prices = pd.read_csv(stocks_file, parse_dates=["date"])
    m = (prices["date"] >= scenario.start) & (prices["date"] <= scenario.end) & (prices["symbol"].isin(symbols))
    prices = prices.loc[m].copy()
    px = prices.pivot(index="date", columns="symbol", values="adjClose").sort_index().dropna()
    px = px[symbols]

    runtimes = []
    last_eq = None
    for _ in range(runs):
        algos = [
            bt.algos.RunMonthly(),
            bt.algos.SelectThese(symbols),
            bt.algos.WeighSpecified(**{s: w for s, w in zip(symbols, weights)}),
            bt.algos.Rebalance(),
        ]
        test = bt.Backtest(bt.Strategy("bench_matrix", algos), px, initial_capital=scenario.initial_capital)
        t0 = time.perf_counter()
        res = bt.run(test)
        runtimes.append(time.perf_counter() - t0)
        last_eq = res.prices.iloc[:, 0]

    assert last_eq is not None
    tr, cagr, mdd, vol, sharpe = compute_metrics(last_eq)
    return ({
        "bt_available": 1.0,
        "bt_runtime_s": float(np.mean(runtimes)),
        "bt_total_return_pct": tr * 100.0,
        "bt_cagr_pct": cagr * 100.0,
        "bt_max_drawdown_pct": mdd * 100.0,
        "bt_vol_annual_pct": vol * 100.0,
        "bt_sharpe": sharpe,
        "bt_rows": float(len(last_eq)),
    }, last_eq)


def overlap_parity(ob_eq: pd.Series, bt_eq: pd.Series | None) -> dict[str, float]:
    if bt_eq is None:
        return {"overlap_rows": 0.0, "overlap_end_delta": np.nan, "overlap_max_abs_delta": np.nan}
    common = ob_eq.index.intersection(bt_eq.index)
    if len(common) == 0:
        return {"overlap_rows": 0.0, "overlap_end_delta": np.nan, "overlap_max_abs_delta": np.nan}
    ob_n = ob_eq.loc[common] / ob_eq.loc[common].iloc[0]
    bt_n = bt_eq.loc[common] / bt_eq.loc[common].iloc[0]
    d = ob_n - bt_n
    return {
        "overlap_rows": float(len(common)),
        "overlap_end_delta": float(d.iloc[-1]),
        "overlap_max_abs_delta": float(d.abs().max()),
    }


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    date_ranges = []
    for chunk in args.date_ranges.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        s, e = chunk.split(":")
        date_ranges.append((pd.Timestamp(s), pd.Timestamp(e)))
    rebal = parse_csv_list(args.rebalance_months, int)
    capitals = parse_csv_list(args.initial_capitals, float)

    scenarios = []
    idx = 1
    for s, e in date_ranges:
        for r in rebal:
            for c in capitals:
                scenarios.append(Scenario(
                    label=f"S{idx}",
                    start=s,
                    end=e,
                    rebalance_months=r,
                    initial_capital=c,
                ))
                idx += 1
    return scenarios


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")
    weights = normalize_weights(symbols, args.weights)
    scenarios = build_scenarios(args)

    rows = []
    for sc in scenarios:
        ob_stats, ob_eq = run_options_backtester(
            stocks_file=args.stocks_file,
            symbols=symbols,
            weights=weights,
            scenario=sc,
            runs=args.runs,
        )
        bt_stats, bt_eq = run_bt(
            stocks_file=args.stocks_file,
            symbols=symbols,
            weights=weights,
            scenario=sc,
            runs=args.runs,
        )
        parity = overlap_parity(ob_eq, bt_eq)
        row = {
            "scenario": sc.label,
            "start": sc.start.date().isoformat(),
            "end": sc.end.date().isoformat(),
            "rebalance_months": sc.rebalance_months,
            "initial_capital": sc.initial_capital,
            "symbols": ",".join(symbols),
            "weights": ",".join(f"{w:.6f}" for w in weights),
            **ob_stats,
            **bt_stats,
            **parity,
        }
        if bt_stats.get("bt_available", 0.0) == 1.0:
            row["speed_ratio_bt_over_ob"] = row["bt_runtime_s"] / row["ob_runtime_s"] if row["ob_runtime_s"] > 0 else np.nan
            row["return_delta_pct_pts"] = row["ob_total_return_pct"] - row["bt_total_return_pct"]
            row["maxdd_delta_pct_pts"] = row["ob_max_drawdown_pct"] - row["bt_max_drawdown_pct"]
        else:
            row["speed_ratio_bt_over_ob"] = np.nan
            row["return_delta_pct_pts"] = np.nan
            row["maxdd_delta_pct_pts"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["start", "rebalance_months", "initial_capital"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\n=== Benchmark Matrix Summary ===")
    print(f"scenarios: {len(out)}")
    print(f"output: {out_path}")
    cols = [
        "scenario", "start", "end", "rebalance_months",
        "ob_runtime_s", "bt_runtime_s", "speed_ratio_bt_over_ob",
        "return_delta_pct_pts", "maxdd_delta_pct_pts", "overlap_max_abs_delta",
    ]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
