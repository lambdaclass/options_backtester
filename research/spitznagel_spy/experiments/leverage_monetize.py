#!/usr/bin/env python3
"""Spitznagel's actual claim: hedge + leverage + monetization, phase-averaged.

Every earlier experiment tested hedged 1x vs unhedged 1x with puts held until
an exit-DTE, never sold on a spike. That is not the thesis. Spitznagel argues
the hedge (a) pays off when monetized near the bottom and reinvested, and
(b) lets you safely hold MORE equity than unhedged, so the hedged *levered*
portfolio beats unhedged 1x. This tests both levers over 1996-2025:

  - monetization: sell a put once it is worth k x its premium (profit_pct)
  - leverage: stocks L x capital, financed by negative cash
  - both scored as the AVERAGE over four entry calendars (odd/even months x
    1st/10th trading day) -- timing_robustness.py showed a single calendar
    can swing excess by ~1.6pp on luck alone

Financing: the engine charges no interest on negative cash and pays none on
positive cash, so each run is adjusted post hoc. Daily, cash earns the 3-month
T-bill rate (FRED DTB3) and borrowed cash pays T-bill + BORROW_SPREAD; the
interest is applied as a proportional drag on total capital. This ignores the
second-order effect of interest on position sizing between rebalances.

Unhedged runs at the same leverage and calendars are included so the hedge's
contribution is separable from leverage's. Margin calls are NOT modeled: a
levered unhedged run's drawdown is reported, not enforced.

Usage:  python research/spitznagel_spy/experiments/leverage_monetize.py [n_workers]
Needs the local licensed SPX parquet under data/processed/ (skips if absent).
"""

import math

import pandas as pd
import spx_common as C

from options_portfolio_backtester import apply_financing

TBILL = C.ROOT / "data" / "processed" / "dtb3.csv"
TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
OUT = "leverage_monetize_results.csv"
BORROW_SPREAD = 0.005  # broker / box-spread financing over T-bills

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = [30, 45, 60]   # sell well before expiry, not only near it
PROFIT_X = [math.inf, 2.0, 4.0, 9.0]  # sell when value >= (1 + x) * premium
LEVERAGE = [1.0, 1.03, 1.05]   # small: 3-5% borrowed on top of 100% equity
BUDGETS = [0.010, 0.020]


def _tbill():
    """Daily 3-month T-bill rate (decimal), cached from FRED."""
    if not TBILL.exists():
        pd.read_csv(TBILL_URL).to_csv(TBILL, index=False)
    r = pd.read_csv(TBILL, parse_dates=["observation_date"], na_values=".")
    return r.set_index("observation_date")["DTB3"].ffill() / 100


def _run(job):
    phase, dates, lev, bud, edte, px = job
    st = C.put_strategy(C.W["o"].schema, OTM, DTE, edte, profit_x=px)
    bt = C.backtest(st, dates, annual=bud, leverage=lev)
    b = apply_financing(bt.balance, C.W["r"], BORROW_SPREAD)
    m = C.metrics(b, 0.0)
    del m["excess"]  # scored against B&H and unhedged runs in the report
    return {"phase": phase, "lev": lev, "bud": bud, "exit": edte, "profit_x": px, **m,
            "fill": bt.option_fill_rate if bud > 0 else float("nan")}


def main():
    workers = C.n_workers()
    rate = _tbill()
    puts, stk = C.load_spx()
    bh = stk.set_index("date")["adjClose"]
    bh_cagr, bh_dd = C.ann(bh), C.maxdd(bh)
    schedules = {lbl: C.schedule(stk["date"], step, par, off)
                 for _, lbl, step, par, off in C.BIMONTHLY}
    # bud 0 = unhedged at that leverage (no entries are attempted).
    jobs = [(lbl, schedules[lbl], lev, 0.0, EXIT_DTE[0], math.inf)
            for lbl in schedules for lev in LEVERAGE]
    jobs += [(lbl, schedules[lbl], lev, bud, edte, px)
             for lbl in schedules for lev in LEVERAGE for bud in BUDGETS
             for edte in EXIT_DTE for px in PROFIT_X]

    print(f"=== Leverage x monetization: {round(OTM[0]*100)}-{round(OTM[1]*100)}% OTM, "
          f"{DTE[0]}-{DTE[1]} DTE, {len(jobs)} runs on {workers} workers ===")
    print(f"Buy & hold 1x (unfinanced index): CAGR {bh_cagr:+.2f}%  maxDD {bh_dd:.1f}%  "
          f"(GFC DD {C.wdd(bh, *C.GFC):.1f}%)")
    print(f"Financing: cash earns T-bill, borrowing pays T-bill + {BORROW_SPREAD*100:.1f}%\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"r": rate}))
    C.write(df, OUT)

    # Score each config by its mean over the four calendars, and its worst one.
    keys = ["lev", "bud", "exit", "profit_x"]
    g = df.groupby(keys).agg(
        cagr=("cagr", "mean"), worst=("cagr", "min"), dd=("dd", "mean"),
        gfc=("gfc", "mean"), covid=("covid", "mean"), sharpe=("sharpe", "mean"),
        fill=("fill", "min"),
    ).reset_index()
    g["vs_bh"] = g["cagr"] - bh_cagr
    unh = g[g["bud"] == 0].set_index("lev")
    g["vs_unh_same_lev"] = g["cagr"] - g["lev"].map(unh["cagr"])

    def show(title, rows):
        print(f"=== {title} ===")
        print(f"  {'lev':>5s} {'bud':>4s} {'exit':>4s} {'sell@':>6s} {'CAGR':>6s} {'vsB&H':>6s} "
              f"{'vsUnh':>6s} {'worst':>6s} {'DD':>6s} {'GFCdd':>6s} {'CVDdd':>6s} {'Sharpe':>6s}")
        for r in rows.itertuples():
            sell = "hold" if math.isinf(r.profit_x) else f"{r.profit_x + 1:.0f}x"
            print(f"  {r.lev:>5.2f} {r.bud*100:>4.1f} {r.exit:>4d} {sell:>6s} {r.cagr:>6.2f} "
                  f"{r.vs_bh:>+6.2f} {r.vs_unh_same_lev:>+6.2f} {r.worst:>6.2f} {r.dd:>6.1f} "
                  f"{r.gfc:>6.1f} {r.covid:>6.1f} {r.sharpe:>6.3f}")
        print()

    show("Unhedged, by leverage (calendar-averaged)", g[g["bud"] == 0])
    hedged = g[g["bud"] > 0]
    show("Hedged, all configs, sorted by calendar-averaged CAGR", hedged.sort_values("cagr", ascending=False))
    # Spitznagel's test: hedged levered portfolio that beats 1x buy & hold on
    # return while drawing down no more than unhedged 1x.
    dd_1x = unh.loc[1.0, "dd"]
    ok = hedged[(hedged["vs_bh"] > 0) & (hedged["dd"] >= dd_1x)]
    show(f"Beats 1x B&H CAGR with DD no worse than unhedged 1x ({dd_1x:.1f}%)", ok)
    low = df[(df["bud"] > 0) & ~(df["fill"] >= 0.90)]
    print(f"{len(low)}/{(df['bud'] > 0).sum()} hedged runs had fill <90% (or no entries)")


if __name__ == "__main__":
    main()
