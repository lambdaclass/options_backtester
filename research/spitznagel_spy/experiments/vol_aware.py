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

import pandas as pd
import spx_common as C

OUT = "vol_aware_results.csv"

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = 30
RULES = ["always", "rank80", "rank50", "abs25"]
BUDGETS = [0.010, 0.020]
CALENDARS = [(lbl, step, par, off) for sched, lbl, step, par, off in C.CALENDARS
             if sched == "monthly"]  # monthly ladder


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


def _run(job):
    cal, rule, dates, scheduled, bud = job
    per_entry = bud / 12 * scheduled / len(dates)  # same total spend as `always`
    st = C.put_strategy(C.W["o"].schema, OTM, DTE, EXIT_DTE)
    bt = C.backtest(st, dates, per_entry=per_entry)
    return {"calendar": cal, "rule": rule, "bud": bud, "kept": len(dates) / scheduled,
            "fill": bt.option_fill_rate, **C.metrics(bt.balance["total capital"], C.W["bh"])}


def main():
    workers = C.n_workers()
    puts, stk = C.load_spx()
    iv = atm_iv(puts)
    bh_cagr = C.ann(stk.set_index("date")["adjClose"])
    jobs = []
    for cal, step, par, off in CALENDARS:
        base = C.schedule(stk["date"], step, par, off)
        for rule in RULES:
            # Always keep the first day so every rule starts invested.
            dates = [d for i, d in enumerate(base) if i == 0 or keep_entry(rule, iv, d)]
            jobs += [(cal, rule, dates, len(base), bud) for bud in BUDGETS]

    print(f"=== IV-aware monthly ladder: {len(jobs)} runs on {workers} workers ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"bh": bh_cagr}))
    C.write(df, OUT)

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
