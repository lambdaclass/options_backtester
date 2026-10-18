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

The budget is set per entry (options_budget_per_rebalance_pct), so a re-hedge spends one
regular entry's premium on top of the schedule. Pass 1 is the same strategy
without re-hedging, so both are reported from one job.

Usage:  python research/spitznagel_spy/experiments/rehedge.py [n_workers]
Needs the local licensed SPX parquet under data/processed/ (skips if absent).
"""

import pandas as pd
import spx_common as C

OUT = "rehedge_results.csv"
MAX_PASSES = 6

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = 30
MONETIZE = [None, 0.85, 0.90, 0.95]   # sell once strike >= m * spot; None = core
BUDGETS = [0.010, 0.020]              # annual, converted to per-entry below


def _backtest(dates, per_entry, m):
    st = C.put_strategy(C.W["o"].schema, OTM, DTE, EXIT_DTE, monetize=m)
    return C.backtest(st, dates, per_entry=per_entry)


def _monetized_dates(bt):
    """Days a put was sold before the normal exit DTE (the monetization fired)."""
    tl = bt.trade_log
    if tl is None or tl.empty:
        return set()
    date = pd.to_datetime(tl["totals"]["date"])
    dte = (pd.to_datetime(tl["leg_1"]["expiration"]) - date).dt.days
    sold = tl["leg_1"]["order"].astype(str).str.contains("STC")
    return set(date[sold & (dte > EXIT_DTE)])


def _stats(bt):
    return {"fill": bt.option_fill_rate, **C.metrics(bt.balance["total capital"], C.W["bh"])}


def _run(job):
    sched, cal, base, step, m, bud = job
    per_entry = bud * step / 12  # annual budget spread over the scheduled entries
    dates = sorted(base)
    bt = _backtest(dates, per_entry, m)
    rows = [{"schedule": sched, "calendar": cal, "m": m, "bud": bud, "rehedge": False,
             "passes": 1, "extra_entries": 0, **_stats(bt)}]
    if m is None:
        return rows
    for n in range(2, MAX_PASSES + 1):
        new = sorted(set(base) | _monetized_dates(bt))
        if new == dates:
            break
        dates = new
        bt = _backtest(dates, per_entry, m)
    rows.append({"schedule": sched, "calendar": cal, "m": m, "bud": bud, "rehedge": True,
                 "passes": n, "extra_entries": len(dates) - len(base), **_stats(bt)})
    return rows


def main():
    workers = C.n_workers()
    puts, stk = C.load_spx()
    bh_cagr = C.ann(stk.set_index("date")["adjClose"])
    jobs = [(sched, cal, C.schedule(stk["date"], step, par, off), step, m, bud)
            for sched, cal, step, par, off in C.CALENDARS for m in MONETIZE for bud in BUDGETS]

    print(f"=== Monetize + same-day re-hedge: {len(jobs)} jobs on {workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"bh": bh_cagr}, every=8))
    df["m"] = df["m"].fillna(0.0)  # 0 = core (no monetization)
    C.write(df, OUT)

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
