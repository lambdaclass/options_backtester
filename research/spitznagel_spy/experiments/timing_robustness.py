#!/usr/bin/env python3
"""Is the exit-30 edge real, or calendar luck? Exit-DTE x rebalance-phase grid.

The 400-config sweep found the 25-30% OTM / 90-180 DTE overlay is positive only
at exit-DTE 30; exits 21 and 45 are both negative. A real effect should decay
smoothly as the exit moves, and should not depend on *which* days the bi-monthly
entries happen to land on. This checks both, over full history 1996-2025:

  - exit-DTE on a fine grid around 30
  - four entry calendars: first business day of odd months (the default
    2BMS schedule), of even months, and the 10th trading day of each
  - three budgets

If the edge is real, exit-30 should sit on a smooth hump and stay positive in
every phase. If it's a spike at 30 in one phase only, the sweep's winner is a
coincidence of entry/exit dates lining up with a few crashes.

Usage:  python research/spitznagel_spy/experiments/timing_robustness.py [n_workers]
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
OUT = Path(__file__).resolve().parents[1] / "findings" / "timing_robustness_results.csv"

START, END = "1996-01-01", "2025-12-31"
GFC = (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-06-30"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))

OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = [21, 25, 28, 30, 32, 35, 40, 45]
BUDGETS = [0.010, 0.020, 0.033]
# (label, month parity relative to the first month, trading-day offset in month)
PHASES = [("odd/1st", 0, 0), ("even/1st", 1, 0), ("odd/10th", 0, 9), ("even/10th", 1, 9)]

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


def _schedule(days, parity, offset):
    """Bi-monthly entry dates: the offset-th trading day of every other month.

    Every schedule also rebalances on the first trading day, so all calendars
    are invested from the same start as buy & hold instead of sitting in cash
    until their first phase date.
    """
    days = pd.Series(pd.DatetimeIndex(sorted(days)))
    months = days.dt.year * 12 + days.dt.month
    keep = (months - months.iloc[0]) % 2 == parity
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
    phase, dates, edte, bud = job
    (olo, ohi), (dlo, dhi) = OTM, DTE
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
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.run(rebalance_dates=dates)
    b = bt.balance["total capital"]
    return {
        "phase": phase, "exit": edte, "bud": bud,
        "cagr": _ann(b), "excess": _ann(b) - bh, "dd": _maxdd(b), "sharpe": _sharpe(b),
        "fill": bt.option_fill_rate, "gfc": _wdd(b, *GFC), "covid": _wdd(b, *COVID),
    }


def main():
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, max(1, (os.cpu_count() or 4) - 1))

    opt = pd.read_parquet(OPTS)
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
    schedules = {lbl: _schedule(stk["date"], par, off) for lbl, par, off in PHASES}
    jobs = [(lbl, schedules[lbl], edte, bud)
            for lbl, _, _ in PHASES for edte in EXIT_DTE for bud in BUDGETS]

    print(f"=== Timing robustness: {round(OTM[0]*100)}-{round(OTM[1]*100)}% OTM, "
          f"{DTE[0]}-{DTE[1]} DTE, {len(jobs)} runs on {n_workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%  (GFC DD {_wdd(bh, *GFC):.1f}%, "
          f"COVID DD {_wdd(bh, *COVID):.1f}%)\n")

    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False); stk.to_csv(sp, index=False)
        del opt, puts, stk  # free the parent's copies; workers read the files
        with Pool(n_workers, initializer=_init, initargs=(op, sp, bh_cagr)) as pool:
            results = pool.map(_run, jobs, chunksize=2)

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"All {len(df)} rows written to {OUT.relative_to(ROOT)}\n")

    order = [lbl for lbl, _, _ in PHASES]
    for bud in BUDGETS:
        sub = df[df["bud"] == bud]
        print(f"=== Excess CAGR (pp) by exit-DTE x entry phase, budget {bud*100:.1f}% ===")
        print(sub.pivot(index="exit", columns="phase", values="excess")[order]
              .round(2).to_string(), "\n")
        print(f"--- GFC drawdown (%), budget {bud*100:.1f}% ---")
        print(sub.pivot(index="exit", columns="phase", values="gfc")[order]
              .round(1).to_string(), "\n")
    low = df[~(df["fill"] >= 0.90)]
    print(f"{len(low)}/{len(df)} runs had fill <90% (or no entries)")


if __name__ == "__main__":
    main()
