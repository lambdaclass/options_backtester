#!/usr/bin/env python3
"""Spitznagel's own recipe, his valuation switch, dividends and execution.

Earlier experiments tested a reconstruction (25-30% OTM, 90-180 DTE puts held
to DTE 30, always on, price index, bid/ask fills). Spitznagel's published
example and backtest differ in four ways, all tested here, pre-registered:

  recipe     "spitz": ~30% OTM (27.5-32.5%) 2-month puts (45-75 DTE),
             sold after about a month (DTE <= 30), bought monthly;
             budgets 1 / 2 / 3.3 / 6%/yr (6% = his 0.5% a month).
             "core": the reconstruction above, bi-monthly, 1 / 2 / 3.3%/yr.
  gate       his edge is conditional on an expensive market: buy puts only
             when Tobin's Q (Fed Z.1, known with a 75-day publication lag) is
             at or above its expanding 75th / 50th percentile since 1945.
             Gated runs spend the same total premium as always-on (per-entry
             budget scaled by scheduled / kept entries); the first trading
             day is always an entry (all runs start invested).
  fills      bid/ask (worst case), half the half-spread (SpreadFraction 0.5),
             mid -- on entries, exits and marks alike.
  dividends  SPX total return (Shiller dividends reinvested daily) on the
             stock side, for every run and the frontier.

Every run is financed (cash earns T-bills; apply_financing) and compared,
like risk_matched.py, with the unhedged stock/T-bill mix of equal mean max
drawdown and of equal volatility, averaged over entry calendars.

Usage:  python research/spitznagel_spy/experiments/spitznagel_recipe.py [n_workers] [recipe,...]
        (naming recipes reruns only those, keeping the other recipes' rows)
Needs the local licensed SPX parquet under data/processed/ (skips if absent),
FRED access for Tobin's Q and T-bills, and Shiller's ie_data (first run
needs xlrd; see spx_common.shiller_dividends).
"""

import sys

import numpy as np
import pandas as pd
import spx_common as C

from options_portfolio_backtester import MarketAtBidAsk, MidPrice, SpreadFraction, apply_financing

OUT = "spitznagel_recipe_results.csv"

# Pre-registered grid -- do not tune after seeing results.
# (name, otm band, dte window, exit dte, months between entries, budgets)
RECIPES = [
    # 45-75 DTE, not 50-70: the chain lists mostly monthly expirations ~4-5
    # weeks apart, and a 20-day window missed them on ~4 of 5 dates (smoke
    # test fill rate 21%, 2019-2020) -- fixed before any results were seen.
    ("spitz", (0.275, 0.325), (45, 75), 30, 1, [0.010, 0.020, 0.033, 0.060]),
    ("core", (0.25, 0.30), (90, 180), 30, 2, [0.010, 0.020, 0.033]),
    # Exit 30 sold the Dec-2008 puts on the 2008-11-20 low (ROBUSTNESS_REVIEW
    # section 1); 28 and 32 are its honest neighbours.
    ("core28", (0.25, 0.30), (90, 180), 28, 2, [0.010, 0.020, 0.033]),
    ("core32", (0.25, 0.30), (90, 180), 32, 2, [0.010, 0.020, 0.033]),
]
GATES = [("always", None), ("q75", 0.75), ("q50", 0.50)]
FILLS = {"bidask": MarketAtBidAsk(), "half": SpreadFraction(0.5), "mid": MidPrice()}
FRONTIER = [round(w, 2) for w in np.arange(0.40, 1.501, 0.05)]


def _run(job):
    kind, recipe, cal, dates, scheduled, bud, gate, fill, w = job
    if kind == "frontier":
        st = C.put_strategy(C.W["o"].schema, (0.25, 0.30), (90, 180), 30)
        bt = C.backtest(st, dates, annual=0.0, leverage=w)
    else:
        _, otm, dte, edte, step, _ = next(r for r in RECIPES if r[0] == recipe)
        st = C.put_strategy(C.W["o"].schema, otm, dte, edte)
        per_entry = bud * step / 12 * scheduled / len(dates)  # equal total spend
        bt = C.backtest(st, dates, per_entry=per_entry, fill_model=FILLS[fill])
    b = apply_financing(bt.balance, C.W["r"])
    d = b.pct_change().dropna()
    m = C.metrics(b, 0.0)
    del m["excess"]
    return {"kind": kind, "recipe": recipe, "calendar": cal, "bud": bud, "gate": gate, "fill": fill,
            "w": w, "kept": len(dates) / scheduled, **m, "vol": d.std() * np.sqrt(252) * 100,
            "fill_rate": bt.option_fill_rate if kind == "hedged" else float("nan")}


def _match(frontier, col, x):
    f = frontier.sort_values(col)
    if not f[col].min() <= x <= f[col].max():
        return float("nan")
    return float(np.interp(x, f[col], f["cagr"]))


def main():
    workers = C.n_workers()
    rate = C.tbill()
    puts, stk = C.load_spx()
    stk = C.with_total_return(stk)
    days = stk["date"]

    jobs = []
    for _, cal, step, par, off in C.BIMONTHLY:
        dates = C.schedule(days, step, par, off)
        jobs += [("frontier", "-", cal, dates, len(dates), 0.0, "-", "-", w) for w in FRONTIER]
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    for recipe, _, _, _, step, budgets in RECIPES:
        if only and recipe not in only:
            continue
        for sched, cal, cstep, par, off in C.CALENDARS:
            if cstep != step:
                continue
            base = C.schedule(days, step, par, off)
            for gate, pct in GATES:
                on = (pd.Series(True, index=pd.DatetimeIndex(base)) if pct is None
                      else C.q_gate_on(base, pct))
                dates = [d for i, d in enumerate(base) if i == 0 or on.iloc[i]]
                jobs += [("hedged", recipe, cal, dates, len(base), bud, gate, fill, 1.0)
                         for bud in budgets for fill in FILLS]

    print(f"=== Spitznagel recipe x Q gate x fills (total return): {len(jobs)} runs on {workers} workers ===\n")
    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"r": rate}))
    if only and (C.FINDINGS / OUT).exists():
        kept = pd.read_csv(C.FINDINGS / OUT)
        kept = kept[(kept["kind"] == "hedged") & ~kept["recipe"].isin(only)]
        df = pd.concat([df, kept], ignore_index=True)
    C.write(df, OUT)

    frontier = df[df["kind"] == "frontier"].groupby("w")[["cagr", "dd", "vol"]].mean().reset_index()
    bh = frontier[frontier["w"] == 1.0].iloc[0]
    print(f"Unhedged 100% SPX total return (financed): CAGR {bh.cagr:.2f}  maxDD {bh.dd:.1f}  vol {bh.vol:.1f}\n")

    h = df[df["kind"] == "hedged"]
    g = h.groupby(["recipe", "gate", "fill", "bud"]).agg(
        cagr=("cagr", "mean"), worst=("cagr", "min"), best=("cagr", "max"), dd=("dd", "mean"),
        vol=("vol", "mean"), gfc=("gfc", "mean"), covid=("covid", "mean"),
        kept=("kept", "mean"), fill_rate=("fill_rate", "min"),
    ).reset_index()
    g["vs_bh"] = g["cagr"] - bh.cagr
    g["edge_dd"] = g["cagr"] - [_match(frontier, "dd", x) for x in g["dd"]]
    g["edge_vol"] = g["cagr"] - [_match(frontier, "vol", x) for x in g["vol"]]

    print("edge = CAGR minus the unhedged stock/T-bill mix with equal max DD / vol (pp/yr)")
    print(f"  {'recipe':6s} {'gate':6s} {'fill':6s} {'bud':>4s} {'kept':>5s} {'CAGR':>6s} {'worst':>6s} "
          f"{'vsB&H':>6s} {'maxDD':>6s} {'GFCdd':>6s} {'CVDdd':>6s} {'edgeDD':>6s} {'edgeV':>6s} {'fill%':>5s}")
    for r in g.sort_values(["recipe", "fill", "gate", "bud"]).itertuples():
        print(f"  {r.recipe:6s} {r.gate:6s} {r.fill:6s} {r.bud*100:>4.1f} {r.kept:>5.0%} {r.cagr:>6.2f} "
              f"{r.worst:>6.2f} {r.vs_bh:>+6.2f} {r.dd:>6.1f} {r.gfc:>6.1f} {r.covid:>6.1f} "
              f"{r.edge_dd:>+6.2f} {r.edge_vol:>+6.2f} {r.fill_rate:>4.0%}")

    # Exit 30 rides the 2008-11-20 sale date; its neighbours 28 and 32 are
    # the honest number (averaging all three still inflates the edge).
    core = g[g["recipe"].isin(["core28", "core32"])]
    if core["recipe"].nunique() > 1:
        s = core.groupby(["gate", "fill", "bud"])[["cagr", "vs_bh", "dd", "gfc", "edge_dd", "edge_vol"]].mean()
        print("\n=== core, exits 28 and 32 (the honest number) ===")
        print(s.round(2).to_string())


if __name__ == "__main__":
    main()
