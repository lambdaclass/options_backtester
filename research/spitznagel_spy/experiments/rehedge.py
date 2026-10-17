#!/usr/bin/env python3
"""Monetize in the crash AND re-hedge the same day (Universa-style).

spitznagel_variants.py found that selling puts once SPX nears the strike
hurts: the portfolio then sits unhedged until the next scheduled entry while
the crash keeps going (2008). Universa re-strikes immediately. The engine only
opens positions on rebalance dates, so this emulates it without engine
changes, by iterating to a fixed point:

  1. run with the regular entry schedule
  2. find the days a put was sold early (exit DTE above the normal exit, i.e.
     the monetization rule fired)
  3. add those days as rebalance dates and rerun -- on each one the engine
     sells the monetized put, buys fresh 25-30% OTM puts struck off the new,
     lower spot, and puts the rest of the proceeds into stock
  4. repeat until the set of dates stops changing (new puts bought mid-crash
     can monetize again)

The budget is set per entry (options_budget_pct), so a re-hedge spends one
regular entry's premium on top of the schedule. Pass 1 is the same strategy
without re-hedging, so both are reported from one job.

Usage:  python research/spitznagel_spy/experiments/rehedge.py [n_workers]
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
OUT = Path(__file__).resolve().parents[1] / "findings" / "rehedge_results.csv"

START, END = "1996-01-01", "2025-12-31"
MAX_PASSES = 6

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = 30
MONETIZE = [None, 0.85, 0.90, 0.95]   # sell once strike >= m * spot; None = core
BUDGETS = [0.010, 0.020]              # annual, converted to per-entry below

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


def _backtest(dates, per_entry, m):
    from options_portfolio_backtester import BacktestEngine, Stock, Strategy, StrategyLeg
    from options_portfolio_backtester.core.types import OptionType, Direction
    (olo, ohi), (dlo, dhi) = OTM, DTE
    o, s = _W["o"], _W["s"]
    sch = o.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
    bt.options_budget_fresh_spend = True  # spend the full budget each entry
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
        # Per-entry budget; the alias setter warns about exactly that semantics.
        bt.options_budget_pct = per_entry
        bt.run(rebalance_dates=dates)
    return bt


def _monetized_dates(bt):
    """Days a put was sold before the normal exit DTE (the monetization fired)."""
    tl = bt.trade_log
    if tl is None or tl.empty:
        return set()
    date = pd.to_datetime(tl["totals"]["date"])
    dte = (pd.to_datetime(tl["leg_1"]["expiration"]) - date).dt.days
    sold = tl["leg_1"]["order"].astype(str).str.contains("STC")
    return set(date[sold & (dte > EXIT_DTE)])


def _stats(bt, bh):
    b = bt.balance["total capital"]
    return {"cagr": _ann(b), "excess": _ann(b) - bh, "dd": _maxdd(b), "sharpe": _sharpe(b),
            "fill": bt.option_fill_rate, "gfc": _wdd(b, *GFC), "covid": _wdd(b, *COVID)}


def _run(job):
    sched, cal, base, step, m, bud = job
    per_entry = bud * step / 12  # annual budget spread over the scheduled entries
    dates = sorted(base)
    bt = _backtest(dates, per_entry, m)
    rows = [{"schedule": sched, "calendar": cal, "m": m, "bud": bud, "rehedge": False,
             "passes": 1, "extra_entries": 0, **_stats(bt, _W["bh"])}]
    if m is None:
        return rows
    for n in range(2, MAX_PASSES + 1):
        new = sorted(set(base) | _monetized_dates(bt))
        if new == dates:
            break
        dates = new
        bt = _backtest(dates, per_entry, m)
    rows.append({"schedule": sched, "calendar": cal, "m": m, "bud": bud, "rehedge": True,
                 "passes": n, "extra_entries": len(dates) - len(base), **_stats(bt, _W["bh"])})
    return rows


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
    jobs = [(sched, cal, _schedule(stk["date"], step, par, off), step, m, bud)
            for sched, cal, step, par, off in CALENDARS for m in MONETIZE for bud in BUDGETS]

    print(f"=== Monetize + same-day re-hedge: {len(jobs)} jobs on {n_workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%\n")

    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False); stk.to_csv(sp, index=False)
        del opt, puts, stk  # free the parent's copies; workers read the files
        with Pool(n_workers, initializer=_init, initargs=(op, sp, bh_cagr)) as pool:
            results = []
            for i, rows in enumerate(pool.imap_unordered(_run, jobs, chunksize=1), 1):
                results += rows
                if i % 8 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done", flush=True)
        print()

    df = pd.DataFrame(results)
    df["m"] = df["m"].fillna(0.0)  # 0 = core (no monetization)
    df.to_csv(OUT, index=False)
    print(f"All {len(df)} rows written to {OUT.relative_to(ROOT)}\n")

    g = df.groupby(["schedule", "m", "rehedge", "bud"]).agg(
        excess=("excess", "mean"), worst=("excess", "min"), best=("excess", "max"),
        dd=("dd", "mean"), gfc=("gfc", "mean"), covid=("covid", "mean"),
        sharpe=("sharpe", "mean"), extra=("extra_entries", "mean"), passes=("passes", "max"),
    ).reset_index().sort_values(["schedule", "bud", "m", "rehedge"])
    print("=== Calendar-averaged excess CAGR (pp vs B&H); m=0 is core ===")
    print(f"  {'schedule':9s} {'bud':>4s} {'m':>5s} {'rehedge':>7s} {'excess':>7s} {'worst':>6s} "
          f"{'best':>6s} {'DD':>6s} {'GFCdd':>6s} {'CVDdd':>6s} {'Sharpe':>6s} {'extra':>5s} {'pass':>4s}")
    for r in g.itertuples():
        print(f"  {r.schedule:9s} {r.bud*100:>4.1f} {r.m:>5.2f} {str(r.rehedge):>7s} "
              f"{r.excess:>+7.2f} {r.worst:>+6.2f} {r.best:>+6.2f} {r.dd:>6.1f} {r.gfc:>6.1f} "
              f"{r.covid:>6.1f} {r.sharpe:>6.3f} {r.extra:>5.1f} {r.passes:>4d}")


if __name__ == "__main__":
    main()
