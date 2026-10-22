#!/usr/bin/env python3
"""Does the hedge beat the other way of cutting risk? Equal-drawdown test.

Spitznagel's claim is not that a put overlay beats 100% stocks on return.
It is that it beats the *alternatives for reducing risk*: holding less
stock and more cash/bonds lowers drawdowns too, but costs return every
year, while cheap convex puts cost little and pay in the crash, so the
hedged portfolio can carry more equity (even leverage) for the same risk.

The test, over SPX 1996-2025 with financing (cash earns the 3-month
T-bill; borrowing pays T-bill + 0.5%):

  frontier   unhedged stock weight w = 0.40 ... 1.50, rest T-bills (or
             borrowed); rebalanced on the same calendars
  hedged     25-30% OTM, 90-180 DTE puts at 1 / 2 / 3.3%/yr, leverage
             L = 1.0 / 1.1 / 1.2 / 1.3 / 1.5, exit at DTE 28 / 30 / 32
             (30 alone carries the one-day 2008 luck; 28 and 32 are its
             honest neighbours)

Each config is averaged over the four bi-monthly calendars. Then each hedged
config is compared with the frontier portfolio that has the SAME mean max
drawdown (and, separately, the same volatility), interpolated along the
frontier: a positive "edge" means the hedge buys more return per unit of
risk than holding cash does. Margin calls are not modeled.

Usage:  python research/spitznagel_spy/experiments/risk_matched.py [n_workers]
Needs the local licensed SPX parquet under data/processed/ (skips if absent).
"""

import numpy as np
import pandas as pd
import spx_common as C

from options_portfolio_backtester import apply_financing

OUT = "risk_matched_results.csv"
BORROW_SPREAD = 0.005

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = [28, 30, 32]
BUDGETS = [0.010, 0.020, 0.033]
LEVERAGE = [1.0, 1.1, 1.2, 1.3, 1.5]
FRONTIER = [round(w, 2) for w in np.arange(0.40, 1.501, 0.05)]


def _run(job):
    kind, cal, dates, lev, bud, edte = job
    st = C.put_strategy(C.W["o"].schema, OTM, DTE, edte)
    bt = C.backtest(st, dates, annual=bud, leverage=lev)
    b = apply_financing(bt.balance, C.W["r"], BORROW_SPREAD)
    d = b.pct_change().dropna()
    m = C.metrics(b, 0.0)
    del m["excess"]
    return {"kind": kind, "calendar": cal, "lev": lev, "bud": bud, "exit": edte, **m,
            "vol": d.std() * np.sqrt(252) * 100,
            "fill": bt.option_fill_rate if bud > 0 else float("nan")}


def _match(frontier, x_col, x):
    """Frontier CAGR and stock weight at frontier[x_col] == x (interpolated)."""
    f = frontier.sort_values(x_col)
    if not f[x_col].min() <= x <= f[x_col].max():
        return float("nan"), float("nan")
    return (float(np.interp(x, f[x_col], f["cagr"])), float(np.interp(x, f[x_col], f["lev"])))


def main():
    workers = C.n_workers()
    rate = C.tbill()
    puts, stk = C.load_spx()
    cals = {lbl: C.schedule(stk["date"], step, par, off) for _, lbl, step, par, off in C.BIMONTHLY}
    jobs = [("frontier", cal, d, w, 0.0, 30) for cal, d in cals.items() for w in FRONTIER]
    jobs += [("hedged", cal, d, lev, bud, edte) for cal, d in cals.items()
             for lev in LEVERAGE for bud in BUDGETS for edte in EXIT_DTE]

    print(f"=== Equal-risk test: {len(jobs)} runs on {workers} workers ===")
    print(f"Financing: cash earns T-bill, borrowing pays T-bill + {BORROW_SPREAD*100:.1f}%\n")

    df = pd.DataFrame(C.run_pool(_run, jobs, puts, stk, workers, {"r": rate}))
    C.write(df, OUT)

    g = df.groupby(["kind", "lev", "bud", "exit"]).agg(
        cagr=("cagr", "mean"), worst=("cagr", "min"), dd=("dd", "mean"), vol=("vol", "mean"),
        gfc=("gfc", "mean"), covid=("covid", "mean"), sharpe=("sharpe", "mean"),
        fill=("fill", "min"),
    ).reset_index()
    frontier = g[g["kind"] == "frontier"]
    print("=== Unhedged frontier (stock weight w, rest T-bills / borrowed) ===")
    print(f"  {'w':>5s} {'CAGR':>6s} {'maxDD':>6s} {'vol':>5s} {'GFCdd':>6s} {'CVDdd':>6s} {'Sharpe':>6s}")
    for r in frontier.itertuples():
        print(f"  {r.lev:>5.2f} {r.cagr:>6.2f} {r.dd:>6.1f} {r.vol:>5.1f} {r.gfc:>6.1f} "
              f"{r.covid:>6.1f} {r.sharpe:>6.3f}")

    hedged = g[g["kind"] == "hedged"].copy()
    hedged[["cagr_at_dd", "w_at_dd"]] = [_match(frontier, "dd", x) for x in hedged["dd"]]
    hedged[["cagr_at_vol", "w_at_vol"]] = [_match(frontier, "vol", x) for x in hedged["vol"]]
    hedged["edge_dd"] = hedged["cagr"] - hedged["cagr_at_dd"]
    hedged["edge_vol"] = hedged["cagr"] - hedged["cagr_at_vol"]

    print("\n=== Hedged vs the unhedged mix with the same max drawdown / volatility ===")
    print("  edge = hedged CAGR - frontier CAGR at equal risk (pp/yr); "
          "w = matching stock weight")
    print(f"  {'lev':>4s} {'bud':>4s} {'exit':>4s} {'CAGR':>6s} {'worst':>6s} {'maxDD':>6s} "
          f"{'vol':>5s} {'GFCdd':>6s} | {'w@DD':>5s} {'edgeDD':>6s} | {'w@vol':>5s} {'edgeV':>6s} {'fill':>5s}")
    for r in hedged.sort_values(["bud", "lev", "exit"]).itertuples():
        print(f"  {r.lev:>4.1f} {r.bud*100:>4.1f} {r.exit:>4d} {r.cagr:>6.2f} {r.worst:>6.2f} "
              f"{r.dd:>6.1f} {r.vol:>5.1f} {r.gfc:>6.1f} | {r.w_at_dd:>5.2f} {r.edge_dd:>+6.2f} | "
              f"{r.w_at_vol:>5.2f} {r.edge_vol:>+6.2f} {r.fill:>4.0%}")

    # The honest summary: average exits 28/30/32 so the 2008-11-20 sale date
    # does not decide the verdict.
    s = hedged.groupby(["lev", "bud"])[["cagr", "dd", "edge_dd", "edge_vol"]].mean().reset_index()
    print("\n=== Averaged over exits 28/30/32 ===")
    print(f"  {'lev':>4s} {'bud':>4s} {'CAGR':>6s} {'maxDD':>6s} {'edgeDD':>6s} {'edgeV':>6s}")
    for r in s.itertuples():
        print(f"  {r.lev:>4.1f} {r.bud*100:>4.1f} {r.cagr:>6.2f} {r.dd:>6.1f} "
              f"{r.edge_dd:>+6.2f} {r.edge_vol:>+6.2f}")


if __name__ == "__main__":
    main()
