#!/usr/bin/env python3
"""Strategies built from what Spitznagel / Universa describe, calendar-robust.

Earlier experiments varied knobs on one recipe (25-30% OTM, 90-180 DTE,
hold to DTE 30). This tests the mechanisms Spitznagel actually argues for:

  core       hold cheap deep-OTM convexity; sell at DTE 30 (reference)
  monetize   "monetize in the crash, redeploy into stocks at the bottom":
             also sell once SPX has fallen to within m of the strike
             (strike >= m * spot, m = 0.85 / 0.90 / 0.95); proceeds are
             reinvested in stock the same day
  short      Universa-style short-dated deep-OTM puts: 50-100 DTE, sell at
             DTE 14; plain or with the m = 0.90 monetization
  schedule   bi-monthly, or a monthly ladder at half size per entry (the
             hedge is always on and no single calendar decides the result)

Pre-registered grid, 1996-2025 SPX (cleaned chain), full budget spent each
rebalance, 1.0x equity. Each config is scored over entry calendars: four
for bi-monthly (odd/even months x 1st/10th trading day), two for monthly
(1st/10th trading day); all start invested on the first trading day.
After a monetization the next hedge is bought at the next scheduled entry.

Usage:  python research/spitznagel_spy/experiments/spitznagel_variants.py [n_workers]
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

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OPTS = ROOT / "data" / "processed" / "spx_full_options.parquet"
STK = ROOT / "data" / "processed" / "spx_full_stocks.csv"
OUT = Path(__file__).resolve().parents[1] / "findings" / "spitznagel_variants_results.csv"

START, END = "1996-01-01", "2025-12-31"
GFC = (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-06-30"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
FILL_FLOOR = 0.90

# Pre-registered grid -- do not tune after seeing results.
# (family, otm_lo, otm_hi, dte_lo, dte_hi, exit_dte, monetize_m or None)
LEGS = [
    ("core",          0.25, 0.30,  90, 180, 30, None),
    ("monetize.85",   0.25, 0.30,  90, 180, 30, 0.85),
    ("monetize.90",   0.25, 0.30,  90, 180, 30, 0.90),
    ("monetize.95",   0.25, 0.30,  90, 180, 30, 0.95),
    ("short20-25",    0.20, 0.25,  50, 100, 14, None),
    ("short25-30",    0.25, 0.30,  50, 100, 14, None),
    ("short20-25.90", 0.20, 0.25,  50, 100, 14, 0.90),
    ("short25-30.90", 0.25, 0.30,  50, 100, 14, 0.90),
]
BUDGETS = [0.010, 0.020]
# (schedule, label, months between entries, month parity, trading-day offset)
CALENDARS = [
    ("bimonthly", "odd/1st", 2, 0, 0), ("bimonthly", "even/1st", 2, 1, 0),
    ("bimonthly", "odd/10th", 2, 0, 9), ("bimonthly", "even/10th", 2, 1, 9),
    ("monthly", "1st", 1, 0, 0), ("monthly", "10th", 1, 0, 9),
]

_W = {}  # per-worker state (data providers), set by the initializer


def _ann(s):
    y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) * 100 - 100 if y > 0 else float("nan")


def _maxdd(s):
    cm = s.cummax()
    return ((s - cm) / cm).min() * 100


def _sharpe(s):
    d = s.pct_change().dropna()
    return _ann(s) / (d.std() * math.sqrt(252) * 100) if d.std() > 0 else 0.0


def _wdd(b, lo, hi):
    s = b.loc[(b.index >= lo) & (b.index <= hi)]
    return _maxdd(s) if len(s) > 2 else float("nan")


def _schedule(days, step, parity, offset):
    """Entry dates: the offset-th trading day of every step-th month, plus the
    first trading day so every calendar starts invested."""
    days = pd.Series(pd.DatetimeIndex(sorted(days)))
    months = days.dt.year * 12 + days.dt.month
    keep = (months - months.iloc[0]) % step == parity
    dates = [g.iloc[min(offset, len(g) - 1)] for _, g in days[keep].groupby(months[keep])]
    return dates if dates[0] == days.iloc[0] else [days.iloc[0]] + dates


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
    sched, cal, dates, leg_cfg, bud = job
    family, olo, ohi, dlo, dhi, edte, m = leg_cfg
    o, s, bh = _W["o"], _W["s"], _W["bh"]
    sch = o.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
    bt.options_budget_annual_pct = bud
    bt.options_budget_fresh_spend = True  # spend the full budget each rebalance
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
    leg.exit_filter = sch.dte <= edte
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
        "schedule": sched, "calendar": cal, "family": family, "bud": bud,
        "cagr": _ann(b), "excess": _ann(b) - bh, "dd": _maxdd(b), "sharpe": _sharpe(b),
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

    bh = stk.set_index("date")["adjClose"]
    bh_cagr = _ann(bh)
    jobs = []
    for sched, cal, step, par, off in CALENDARS:
        dates = _schedule(stk["date"], step, par, off)
        jobs += [(sched, cal, dates, leg, bud) for leg in LEGS for bud in BUDGETS]

    print(f"=== Spitznagel variants: {len(jobs)} runs on {n_workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%  maxDD {_maxdd(bh):.1f}%  "
          f"(GFC DD {_wdd(bh, *GFC):.1f}%, COVID DD {_wdd(bh, *COVID):.1f}%)\n")

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

    # Score each config by its mean over calendars, and its worst calendar.
    g = df.groupby(["schedule", "family", "bud"]).agg(
        excess=("excess", "mean"), worst=("excess", "min"), best=("excess", "max"),
        dd=("dd", "mean"), gfc=("gfc", "mean"), covid=("covid", "mean"),
        sharpe=("sharpe", "mean"), fill=("fill", "min"),
    ).reset_index().sort_values("excess", ascending=False)
    print("=== Calendar-averaged, sorted by mean excess CAGR (pp vs B&H) ===")
    print(f"  {'schedule':9s} {'family':14s} {'bud':>4s} {'excess':>7s} {'worst':>6s} {'best':>6s} "
          f"{'DD':>6s} {'GFCdd':>6s} {'CVDdd':>6s} {'Sharpe':>6s} {'fill':>5s}")
    for r in g.itertuples():
        flag = "" if r.fill >= FILL_FLOOR else " LOW"
        print(f"  {r.schedule:9s} {r.family:14s} {r.bud*100:>4.1f} {r.excess:>+7.2f} "
              f"{r.worst:>+6.2f} {r.best:>+6.2f} {r.dd:>6.1f} {r.gfc:>6.1f} {r.covid:>6.1f} "
              f"{r.sharpe:>6.3f} {r.fill:>4.0%}{flag}")


if __name__ == "__main__":
    main()
