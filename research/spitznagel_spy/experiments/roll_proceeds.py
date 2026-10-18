#!/usr/bin/env python3
"""Roll the proceeds of a put sale into new puts (Universa-style).

rehedge.py re-hedged after a monetization with one regular entry's budget,
far too little to replace the convexity sold. Universa instead rolls: it
cashes in puts that have paid and spends part of the proceeds on new
deep-OTM puts struck off the lower spot, keeping the hedge's size. The engine
does that with `roll_exit_proceeds_pct = f`: on every put sale (DTE exit or
monetization) f of the proceeds buys new 25-30% OTM, 90-180 DTE puts the same
day; the rest goes to stock. In calm markets sales are worth little and the
roll is ~free; it only matters after a crash.

Pre-registered grid, 1996-2025 SPX (cleaned chain), full budget spent each
rebalance, 1.0x equity, scored over the six entry calendars of
spitznagel_variants.py. f = 0 is the no-roll reference.

Usage:  python research/spitznagel_spy/experiments/roll_proceeds.py [n_workers]
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

from spx_clean import clean_spx_chain
from spitznagel_variants import CALENDARS, GFC, COVID, _ann, _maxdd, _schedule, _sharpe, _wdd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OPTS = ROOT / "data" / "processed" / "spx_full_options.parquet"
STK = ROOT / "data" / "processed" / "spx_full_stocks.csv"
OUT = Path(__file__).resolve().parents[1] / "findings" / "roll_proceeds_results.csv"

START, END = "1996-01-01", "2025-12-31"

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = 30
MONETIZE = [None, 0.90, 0.95]       # also sell once strike >= m * spot; None = DTE exit only
ROLL = [0.0, 0.25, 0.50, 1.00]      # fraction of each sale's proceeds rolled into new puts
BUDGETS = [0.010, 0.020]

_W = {}  # per-worker state (data providers), set by the initializer


def _init(op, sp, bh_cagr):
    """Load data once per worker process."""
    warnings.simplefilter("ignore")
    # Fill shortfalls are reported in the table; silence the engine's per-run log.
    logging.getLogger("options_portfolio_backtester.engine.engine").setLevel(logging.ERROR)
    from options_portfolio_backtester import HistoricalOptionsData, TiingoData
    _W["o"] = HistoricalOptionsData(op)
    _W["s"] = TiingoData(sp)
    _W["bh"] = bh_cagr


def _run(job):
    from options_portfolio_backtester import BacktestEngine, Stock, Strategy, StrategyLeg
    from options_portfolio_backtester.core.types import OptionType, Direction
    sched, cal, dates, m, roll, bud = job
    (olo, ohi), (dlo, dhi) = OTM, DTE
    o, s = _W["o"], _W["s"]
    sch = o.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
    bt.options_budget_annual_pct = bud
    bt.options_budget_fresh_spend = True  # spend the full budget each rebalance
    bt.roll_exit_proceeds_pct = roll or None
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.assert_invariants = True
    bt.stocks = [Stock("SPX", 1.0)]
    bt.stocks_data = s
    bt.options_data = o
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (sch.underlying == "SPX") & (sch.dte >= dlo) & (sch.dte <= dhi)
        & (sch.strike <= sch.underlying_last * (1 - olo))
        & (sch.strike >= sch.underlying_last * (1 - ohi))
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = sch.dte <= EXIT_DTE
    if m is not None:
        leg.exit_filter = leg.exit_filter | (sch.strike >= sch.underlying_last * m)
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.run(rebalance_dates=dates)
    b = bt.balance["total capital"]
    return {
        "schedule": sched, "calendar": cal, "m": m or 0.0, "roll": roll, "bud": bud,
        "cagr": _ann(b), "excess": _ann(b) - _W["bh"], "dd": _maxdd(b), "sharpe": _sharpe(b),
        "fill": bt.option_fill_rate, "gfc": _wdd(b, *GFC), "covid": _wdd(b, *COVID),
    }


def main():
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, max(1, (os.cpu_count() or 4) - 1))

    opt = clean_spx_chain(pd.read_parquet(OPTS))
    stk = pd.read_csv(STK, parse_dates=["date"])
    opt["quotedate"] = pd.to_datetime(opt["quotedate"])
    opt["expiration"] = pd.to_datetime(opt["expiration"])
    lo, hi = pd.Timestamp(START), pd.Timestamp(END)
    opt = opt[(opt["quotedate"] >= lo) & (opt["quotedate"] <= hi)]
    stk = stk[(stk["date"] >= lo) & (stk["date"] <= hi)]
    shared = set(opt["quotedate"]) & set(stk["date"])
    opt = opt[opt["quotedate"].isin(shared)]
    stk = stk[stk["date"].isin(shared)].sort_values("date", kind="stable")
    # SPX puts only; never filter by strike/DTE (held puts need every row until
    # exit -- see parallel_sweep.py).
    puts = opt[(opt["type"].astype(str).str.lower().str[0] == "p")
               & (opt["underlying"] == "SPX")].sort_values("quotedate", kind="stable")

    bh_cagr = _ann(stk.set_index("date")["adjClose"])
    jobs = [(sched, cal, _schedule(stk["date"], step, par, off), m, roll, bud)
            for sched, cal, step, par, off in CALENDARS
            for m in MONETIZE for roll in ROLL for bud in BUDGETS]

    print(f"=== Roll sale proceeds into new puts: {len(jobs)} runs on {n_workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%\n")

    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False); stk.to_csv(sp, index=False)
        del opt, puts, stk  # free the parent's copies; workers read the files
        with Pool(n_workers, initializer=_init, initargs=(op, sp, bh_cagr)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=2), 1):
                results.append(r)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done", flush=True)
        print()

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"All {len(df)} rows written to {OUT.relative_to(ROOT)}\n")

    g = df.groupby(["schedule", "bud", "m", "roll"]).agg(
        excess=("excess", "mean"), worst=("excess", "min"), best=("excess", "max"),
        dd=("dd", "mean"), gfc=("gfc", "mean"), covid=("covid", "mean"),
        sharpe=("sharpe", "mean"), fill=("fill", "min"),
    ).reset_index()
    print("=== Calendar-averaged excess CAGR (pp vs B&H); m=0 is DTE exit only ===")
    print(f"  {'schedule':9s} {'bud':>4s} {'m':>5s} {'roll':>5s} {'excess':>7s} {'worst':>6s} "
          f"{'best':>6s} {'DD':>6s} {'GFCdd':>6s} {'CVDdd':>6s} {'Sharpe':>6s} {'fill':>5s}")
    for r in g.itertuples():
        print(f"  {r.schedule:9s} {r.bud*100:>4.1f} {r.m:>5.2f} {r.roll:>5.2f} "
              f"{r.excess:>+7.2f} {r.worst:>+6.2f} {r.best:>+6.2f} {r.dd:>6.1f} {r.gfc:>6.1f} "
              f"{r.covid:>6.1f} {r.sharpe:>6.3f} {r.fill:>4.0%}")


if __name__ == "__main__":
    main()
