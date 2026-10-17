#!/usr/bin/env python3
"""Config x start-year robustness matrix for the put overlay.

A single full-period CAGR hides how much an "edge" depends on when you happened
to start (start in 2008 and you bank the GFC immediately; start in 2010 and you
bleed through the calm 2010s). This runs a panel of configs across many start
years (all ending 2025) and reports, per config: excess at each start, the
WORST start, the mean, and the fraction of starts that were positive. A robust
config is positive across all starts -- not just the ones that begin before a
crash.

Config tuple: (label, otm_lo, otm_hi, dte_lo, dte_hi, exit_dte, budget).
All bi-monthly (validated less churn than monthly). Fill-rate is checked; a
config whose fill drops below 90% in any window is flagged.

Runs in parallel: the SPX put chain for each start year is written once, and
the (config, start) pairs are spread across a process pool; each worker caches
the data providers per start year.

Usage:  python research/spitznagel_spy/experiments/start_year_robustness.py [n_workers]
Needs the local licensed SPX parquet under data/processed/ (skips if absent).
"""

import logging
import math
import os
import sys
import tempfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OPTS = ROOT / "data" / "processed" / "spx_full_options.parquet"
STK = ROOT / "data" / "processed" / "spx_full_stocks.csv"

END = "2025-12-31"
START_YEARS = ["1996", "2000", "2004", "2008", "2010", "2012", "2015", "2018"]

# (label, otm_lo, otm_hi, dte_lo, dte_hi, exit_dte, budget)
CONFIGS = [
    ("25-30/90-180/e30/3.3", 0.25, 0.30, 90, 180, 30, 0.033),
    ("25-30/90-180/e30/1.0", 0.25, 0.30, 90, 180, 30, 0.010),
    # Exit-DTE neighbours: the full sweep is positive only at exit 30 -- does
    # that hold across start years, or is 30 a lucky alignment?
    ("25-30/90-180/e21/2.0", 0.25, 0.30, 90, 180, 21, 0.020),
    ("25-30/90-180/e30/2.0", 0.25, 0.30, 90, 180, 30, 0.020),
    ("25-30/90-180/e45/2.0", 0.25, 0.30, 90, 180, 45, 0.020),
    ("20-25/90-180/e30/1.0", 0.20, 0.25, 90, 180, 30, 0.010),
    ("15-20/90-180/e30/1.0", 0.15, 0.20, 90, 180, 30, 0.010),
    ("10-15/90-180/e30/3.3", 0.10, 0.15, 90, 180, 30, 0.033),
    ("10-15/60-90/e30/1.0",  0.10, 0.15, 60, 90, 30, 0.010),
    ("15-20/60-90/e30/3.3",  0.15, 0.20, 60, 90, 30, 0.033),
    ("25-30/60-90/e30/3.3",  0.25, 0.30, 60, 90, 30, 0.033),
]

_W = {}  # per-worker cache: start year -> (options data, stocks data)


def _ann(s):
    y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) * 100 - 100 if y > 0 else float("nan")


def _data(start, td):
    """Load (and cache) the data providers for one start year in this worker."""
    # Fill shortfalls are reported in the table; silence the engine's per-run log.
    logging.getLogger("options_portfolio_backtester.engine.engine").setLevel(logging.ERROR)
    if start not in _W:
        from options_portfolio_backtester import HistoricalOptionsData, TiingoData
        _W[start] = (HistoricalOptionsData(os.path.join(td, f"o{start}.parquet")),
                     TiingoData(os.path.join(td, f"s{start}.csv")))
    return _W[start]


def _run(job):
    from options_portfolio_backtester import BacktestEngine, Stock, Strategy, StrategyLeg
    from options_portfolio_backtester.core.types import OptionType, Direction
    cfg, start, td, bh = job
    o, s = _data(start, td)
    sch = o.schema
    _, olo, ohi, dlo, dhi, edte, bud = cfg
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
    bt.options_budget_annual_pct = bud
    bt.options_budget_fresh_spend = True  # spend the full budget each rebalance
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.assert_invariants = True
    bt.stocks = [Stock("SPX", 1.0)]
    bt.stocks_data = s; bt.options_data = o
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (sch.underlying == "SPX") & (sch.dte >= dlo) & (sch.dte <= dhi)
        & (sch.strike <= sch.underlying_last * (1 - olo))
        & (sch.strike >= sch.underlying_last * (1 - ohi))
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = sch.dte <= edte
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.run(rebalance_freq=2, rebalance_unit="BMS")
    return cfg[0], start, _ann(bt.balance["total capital"]) - bh, bt.option_fill_rate


def main():
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, max(1, (os.cpu_count() or 4) - 1))
    opt_all = pd.read_parquet(OPTS)
    stk_all = pd.read_csv(STK, parse_dates=["date"])
    opt_all["quotedate"] = pd.to_datetime(opt_all["quotedate"])
    opt_all["expiration"] = pd.to_datetime(opt_all["expiration"])
    # SPX puts only: the strategy never touches calls. Never filter by strike/DTE
    # -- held puts need every row until exit (see parallel_sweep.py).
    opt_all = opt_all[opt_all["type"].astype(str).str.lower().str[0] == "p"]
    hi = pd.Timestamp(END)

    with tempfile.TemporaryDirectory() as td:
        bh = {}
        for start in START_YEARS:
            lo = pd.Timestamp(start + "-01-01")
            opt = opt_all[(opt_all["quotedate"] >= lo) & (opt_all["quotedate"] <= hi)]
            stk = stk_all[(stk_all["date"] >= lo) & (stk_all["date"] <= hi)]
            shared = set(opt["quotedate"]) & set(stk["date"])
            opt = opt[opt["quotedate"].isin(shared)].sort_values("quotedate", kind="stable")
            stk = stk[stk["date"].isin(shared)].sort_values("date", kind="stable")
            opt.to_parquet(os.path.join(td, f"o{start}.parquet"), index=False)
            stk.to_csv(os.path.join(td, f"s{start}.csv"), index=False)
            bh[start] = _ann(stk.set_index("date")["adjClose"])
        del opt_all, stk_all, opt, stk

        # Start-major order so each worker's chunk mostly reuses one cached year.
        jobs = [(cfg, y, td, bh[y]) for y in START_YEARS for cfg in CONFIGS]
        with Pool(n_workers) as pool:
            out = pool.map(_run, jobs, chunksize=max(1, len(CONFIGS) // 2))

    res = {(label, y): (ex, fr) for label, y, ex, fr in out}
    print(f"=== Excess (pp) by config x start-year (all -> {END}, bi-monthly) ===")
    print("LOW = fill <90% (or no entries) in some window\n")
    hdr = "  " + f"{'config':22s}" + "".join(f"{y:>7s}" for y in START_YEARS) + f"{'worst':>7s}{'mean':>7s}{'>0':>5s}"
    print(hdr)
    for cfg in CONFIGS:
        exraw = [res[(cfg[0], y)][0] for y in START_YEARS]
        # not (fr >= 0.90) also catches NaN fill (no entries attempted)
        anylow = any(not (res[(cfg[0], y)][1] >= 0.90) for y in START_YEARS)
        worst = min(exraw); mean = sum(exraw) / len(exraw)
        pos = sum(1 for e in exraw if e > 0)
        cells = "".join(f"{e:>+7.2f}" for e in exraw)
        flag = " LOW" if anylow else ""
        print(f"  {cfg[0]:22s}{cells}{worst:>+7.2f}{mean:>+7.2f}{pos:>3d}/{len(START_YEARS)}{flag}")


if __name__ == "__main__":
    main()
