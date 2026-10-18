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

import pandas as pd
import spx_common as C

OUT = "roll_proceeds_results.csv"

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = 30
MONETIZE = [None, 0.90, 0.95]       # also sell once strike >= m * spot; None = DTE exit only
ROLL = [0.0, 0.25, 0.50, 1.00]      # fraction of each sale's proceeds rolled into new puts
BUDGETS = [0.010, 0.020]


def _run(job):
    sched, cal, dates, m, roll, bud = job
    st = C.put_strategy(C.W["o"].schema, OTM, DTE, EXIT_DTE, monetize=m)
    bt = C.backtest(st, dates, annual=bud, roll=roll or None)
    return {"schedule": sched, "calendar": cal, "m": m or 0.0, "roll": roll, "bud": bud,
            "fill": bt.option_fill_rate, **C.metrics(bt.balance["total capital"], C.W["bh"])}


def main():
    workers = C.n_workers()
    puts, stk = C.load_spx()
    bh_cagr = C.ann(stk.set_index("date")["adjClose"])
    jobs = [(sched, cal, C.schedule(stk["date"], step, par, off), m, roll, bud)
            for sched, cal, step, par, off in C.CALENDARS
            for m in MONETIZE for roll in ROLL for bud in BUDGETS]

    print(f"=== Roll sale proceeds into new puts: {len(jobs)} runs on {workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"bh": bh_cagr}))
    C.write(df, OUT)

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
