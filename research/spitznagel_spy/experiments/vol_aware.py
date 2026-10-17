#!/usr/bin/env python3
"""Don't buy insurance when it's expensive: IV-aware monthly ladder.

Spitznagel's case rests on cost-effectiveness, but the earlier signal battery
switched the hedge fully OFF on signals, leaving the portfolio naked. Here the
hedge stays on: a monthly ladder of 90-180 DTE puts (each tranche is held
~2-5 months) simply skips the month's purchase when implied volatility is
expensive, so crisis-priced premium is not bought.

IV is the median implied vol of ~30-day (20-45 DTE), near-the-money (0.97-1.03)
SPX puts each day, built from the chain itself (corr 0.986 with VIX over
2007-2025; the VIX file only starts in 2007). Rules, pre-registered, all
point-in-time (only data up to the entry day):

  always     buy every month (the monthly-ladder baseline)
  rank80     skip when IV is above its trailing 252-day 80th percentile
  rank50     skip when IV is above its trailing 252-day median
  abs25      skip when IV > 25%

Skipping spends less, so each rule's per-entry budget is scaled by
(scheduled entries / entries kept): every rule spends the same total premium
as `always`, and the comparison is about *when* it is spent.

Usage:  python research/spitznagel_spy/experiments/vol_aware.py [n_workers]
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
from spitznagel_variants import GFC, COVID, _ann, _maxdd, _schedule, _sharpe, _wdd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OPTS = ROOT / "data" / "processed" / "spx_full_options.parquet"
STK = ROOT / "data" / "processed" / "spx_full_stocks.csv"
OUT = Path(__file__).resolve().parents[1] / "findings" / "vol_aware_results.csv"

START, END = "1996-01-01", "2025-12-31"

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = 30
RULES = ["always", "rank80", "rank50", "abs25"]
BUDGETS = [0.010, 0.020]
CALENDARS = [("1st", 0), ("10th", 9)]  # monthly ladder, trading-day offset

_W = {}  # per-worker state (data providers), set by the initializer


def atm_iv(puts):
    """Daily median IV of ~30-day near-the-money puts."""
    dte = (puts["expiration"] - puts["quotedate"]).dt.days
    mny = puts["strike"] / puts["underlying_last"]
    near = (dte.between(20, 45) & mny.between(0.97, 1.03)
            & puts["impliedvol"].between(0.01, 3.0) & (puts["bid"] > 0))
    return puts[near].groupby("quotedate")["impliedvol"].median()


def keep_entry(rule, iv, date):
    """Whether `rule` buys on `date`, using IV up to and including that day."""
    if rule == "always":
        return True
    hist = iv.loc[:date]
    if hist.empty:
        return True
    now = hist.iloc[-1]
    if rule == "abs25":
        return now <= 0.25
    window = hist.iloc[-252:]
    q = 0.80 if rule == "rank80" else 0.50
    return now <= window.quantile(q)


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
    cal, rule, dates, scheduled, bud = job
    (olo, ohi), (dlo, dhi) = OTM, DTE
    o, s = _W["o"], _W["s"]
    sch = o.schema
    per_entry = bud / 12 * scheduled / len(dates)  # same total spend as `always`
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
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Per-entry budget; the alias setter warns about exactly that semantics.
        bt.options_budget_pct = per_entry
        bt.run(rebalance_dates=dates)
    b = bt.balance["total capital"]
    return {
        "calendar": cal, "rule": rule, "bud": bud, "kept": len(dates) / scheduled,
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
    iv = atm_iv(puts)

    bh_cagr = _ann(stk.set_index("date")["adjClose"])
    jobs = []
    for cal, off in CALENDARS:
        base = _schedule(stk["date"], 1, 0, off)
        for rule in RULES:
            # Always keep the first day so every rule starts invested.
            dates = [d for i, d in enumerate(base) if i == 0 or keep_entry(rule, iv, d)]
            jobs += [(cal, rule, dates, len(base), bud) for bud in BUDGETS]

    print(f"=== IV-aware monthly ladder: {len(jobs)} runs on {n_workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%\n")

    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False); stk.to_csv(sp, index=False)
        del opt, puts, stk  # free the parent's copies; workers read the files
        with Pool(n_workers, initializer=_init, initargs=(op, sp, bh_cagr)) as pool:
            results = pool.map(_run, jobs, chunksize=1)

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"All {len(df)} rows written to {OUT.relative_to(ROOT)}\n")

    g = df.groupby(["bud", "rule"]).agg(
        kept=("kept", "mean"), excess=("excess", "mean"), worst=("excess", "min"),
        best=("excess", "max"), dd=("dd", "mean"), gfc=("gfc", "mean"),
        covid=("covid", "mean"), sharpe=("sharpe", "mean"), fill=("fill", "min"),
    ).reset_index()
    print("=== Calendar-averaged (1st/10th), equal total premium per budget ===")
    print(f"  {'bud':>4s} {'rule':7s} {'kept':>5s} {'excess':>7s} {'worst':>6s} {'best':>6s} "
          f"{'DD':>6s} {'GFCdd':>6s} {'CVDdd':>6s} {'Sharpe':>6s} {'fill':>5s}")
    for r in g.itertuples():
        print(f"  {r.bud*100:>4.1f} {r.rule:7s} {r.kept:>5.0%} {r.excess:>+7.2f} {r.worst:>+6.2f} "
              f"{r.best:>+6.2f} {r.dd:>6.1f} {r.gfc:>6.1f} {r.covid:>6.1f} {r.sharpe:>6.3f} "
              f"{r.fill:>4.0%}")


if __name__ == "__main__":
    main()
