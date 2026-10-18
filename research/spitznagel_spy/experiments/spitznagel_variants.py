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

import pandas as pd
import spx_common as C

OUT = "spitznagel_variants_results.csv"

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


def _run(job):
    sched, cal, dates, leg_cfg, bud = job
    family, olo, ohi, dlo, dhi, edte, m = leg_cfg
    st = C.put_strategy(C.W["o"].schema, (olo, ohi), (dlo, dhi), edte, monetize=m)
    bt = C.backtest(st, dates, annual=bud)
    return {"schedule": sched, "calendar": cal, "family": family, "bud": bud,
            "fill": bt.option_fill_rate, **C.metrics(bt.balance["total capital"], C.W["bh"])}


def main():
    workers = C.n_workers()
    puts, stk = C.load_spx()
    bh = stk.set_index("date")["adjClose"]
    bh_cagr = C.ann(bh)
    jobs = []
    for sched, cal, step, par, off in C.CALENDARS:
        dates = C.schedule(stk["date"], step, par, off)
        jobs += [(sched, cal, dates, leg, bud) for leg in LEGS for bud in BUDGETS]

    print(f"=== Spitznagel variants: {len(jobs)} runs on {workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%  maxDD {C.maxdd(bh):.1f}%  "
          f"(GFC DD {C.wdd(bh, *C.GFC):.1f}%, COVID DD {C.wdd(bh, *C.COVID):.1f}%)\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"bh": bh_cagr}))
    C.write(df, OUT)

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
        flag = "" if r.fill >= C.FILL_FLOOR else " LOW"
        print(f"  {r.schedule:9s} {r.family:14s} {r.bud*100:>4.1f} {r.excess:>+7.2f} "
              f"{r.worst:>+6.2f} {r.best:>+6.2f} {r.dd:>6.1f} {r.gfc:>6.1f} {r.covid:>6.1f} "
              f"{r.sharpe:>6.3f} {r.fill:>4.0%}{flag}")


if __name__ == "__main__":
    main()
