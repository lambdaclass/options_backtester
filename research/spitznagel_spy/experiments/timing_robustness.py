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

import pandas as pd
import spx_common as C

OUT = "timing_robustness_results.csv"

OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = [21, 25, 28, 30, 32, 35, 40, 45]
BUDGETS = [0.010, 0.020, 0.033]


def _run(job):
    phase, dates, edte, bud = job
    bt = C.backtest(C.put_strategy(C.W["o"].schema, OTM, DTE, edte), dates, annual=bud)
    return {"phase": phase, "exit": edte, "bud": bud, "fill": bt.option_fill_rate,
            **C.metrics(bt.balance["total capital"], C.W["bh"])}


def main():
    workers = C.n_workers()
    puts, stk = C.load_spx()
    bh = stk.set_index("date")["adjClose"]
    bh_cagr = C.ann(bh)
    schedules = {lbl: C.schedule(stk["date"], step, par, off)
                 for _, lbl, step, par, off in C.BIMONTHLY}
    jobs = [(lbl, schedules[lbl], edte, bud)
            for lbl in schedules for edte in EXIT_DTE for bud in BUDGETS]

    print(f"=== Timing robustness: {round(OTM[0]*100)}-{round(OTM[1]*100)}% OTM, "
          f"{DTE[0]}-{DTE[1]} DTE, {len(jobs)} runs on {workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%  (GFC DD {C.wdd(bh, *C.GFC):.1f}%, "
          f"COVID DD {C.wdd(bh, *C.COVID):.1f}%)\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"bh": bh_cagr}))
    C.write(df, OUT)

    order = list(schedules)
    for bud in BUDGETS:
        sub = df[df["bud"] == bud]
        print(f"=== Excess CAGR (pp) by exit-DTE x entry phase, budget {bud*100:.1f}% ===")
        print(sub.pivot(index="exit", columns="phase", values="excess")[order]
              .round(2).to_string(), "\n")
        print(f"--- GFC drawdown (%), budget {bud*100:.1f}% ---")
        print(sub.pivot(index="exit", columns="phase", values="gfc")[order]
              .round(1).to_string(), "\n")
    low = df[~(df["fill"] >= C.FILL_FLOOR)]
    print(f"{len(low)}/{len(df)} runs had fill <90% (or no entries)")


if __name__ == "__main__":
    main()
