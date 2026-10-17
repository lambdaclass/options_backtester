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
TBILL = ROOT / "data" / "processed" / "dtb3.csv"
TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
OUT = Path(__file__).resolve().parents[1] / "findings" / "leverage_monetize_results.csv"

START, END = "1996-01-01", "2025-12-31"
GFC = (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-06-30"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
BORROW_SPREAD = 0.005  # broker / box-spread financing over T-bills

# Pre-registered grid -- do not tune after seeing results.
OTM = (0.25, 0.30)
DTE = (90, 180)
EXIT_DTE = [30, 45, 60]   # sell well before expiry, not only near it
PROFIT_X = [math.inf, 2.0, 4.0, 9.0]  # sell when value >= (1 + x) * premium
LEVERAGE = [1.0, 1.03, 1.05]   # small: 3-5% borrowed on top of 100% equity
BUDGETS = [0.010, 0.020]
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


def _tbill():
    """Daily 3-month T-bill rate (decimal), cached from FRED."""
    if not TBILL.exists():
        pd.read_csv(TBILL_URL).to_csv(TBILL, index=False)
    r = pd.read_csv(TBILL, parse_dates=["observation_date"], na_values=".")
    return r.set_index("observation_date")["DTB3"].ffill() / 100


def _financed(bal, rate):
    """Total capital after crediting/charging interest on the cash balance."""
    total, cash = bal["total capital"], bal["cash"]
    r = rate.reindex(total.index, method="ffill").bfill()
    r = r.where(cash.shift() >= 0, r + BORROW_SPREAD)
    dt = total.index.to_series().diff().dt.days.fillna(0) / 365.0
    interest = (cash.shift().fillna(0) * r.shift().fillna(0) * dt)
    return total * (1 + interest / total.shift().fillna(total)).cumprod()


def _init(op, sp, rate):
    """Load data once per worker process."""
    warnings.simplefilter("ignore")
    # Fill shortfalls are reported in the table; silence the engine's per-run log.
    logging.getLogger("options_portfolio_backtester.engine.engine").setLevel(logging.ERROR)
    from options_portfolio_backtester import HistoricalOptionsData, TiingoData
    _W["o"] = HistoricalOptionsData(op)
    _W["s"] = TiingoData(sp)
    _W["r"] = rate


def _run(job):
    from options_portfolio_backtester import BacktestEngine, Stock, Strategy, StrategyLeg
    from options_portfolio_backtester.core.types import OptionType, Direction
    phase, dates, lev, bud, edte, px = job
    (olo, ohi), (dlo, dhi) = OTM, DTE
    o, s = _W["o"], _W["s"]
    sch = o.schema
    bt = BacktestEngine({"stocks": lev, "options": 0.0, "cash": 1.0 - lev},
                        initial_capital=1_000_000)
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
    st.add_exit_thresholds(profit_pct=px, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.run(rebalance_dates=dates)
    b = _financed(bt.balance, _W["r"])
    return {
        "phase": phase, "lev": lev, "bud": bud, "exit": edte, "profit_x": px,
        "cagr": _ann(b), "dd": _maxdd(b), "sharpe": _sharpe(b),
        "gfc": _wdd(b, *GFC), "covid": _wdd(b, *COVID),
        "fill": bt.option_fill_rate if bud > 0 else float("nan"),
    }


def main():
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, max(1, (os.cpu_count() or 4) - 1))
    rate = _tbill()

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
    bh_cagr, bh_dd = _ann(bh), _maxdd(bh)
    schedules = {lbl: _schedule(stk["date"], par, off) for lbl, par, off in PHASES}
    jobs = [(lbl, schedules[lbl], lev, 0.0, EXIT_DTE[0], math.inf)
            for lbl, _, _ in PHASES for lev in LEVERAGE]
    jobs += [(lbl, schedules[lbl], lev, bud, edte, px)
             for lbl, _, _ in PHASES for lev in LEVERAGE for bud in BUDGETS
             for edte in EXIT_DTE for px in PROFIT_X]

    print(f"=== Leverage x monetization: {round(OTM[0]*100)}-{round(OTM[1]*100)}% OTM, "
          f"{DTE[0]}-{DTE[1]} DTE, {len(jobs)} runs on {n_workers} workers ===")
    print(f"Buy & hold 1x (unfinanced index): CAGR {bh_cagr:+.2f}%  maxDD {bh_dd:.1f}%  "
          f"(GFC DD {_wdd(bh, *GFC):.1f}%)")
    print(f"Financing: cash earns T-bill, borrowing pays T-bill + {BORROW_SPREAD*100:.1f}%\n")

    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False); stk.to_csv(sp, index=False)
        del opt, puts, stk  # free the parent's copies; workers read the files
        with Pool(n_workers, initializer=_init, initargs=(op, sp, rate)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=2), 1):
                results.append(r)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done", flush=True)
        print()

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"All {len(df)} rows written to {OUT.relative_to(ROOT)}\n")

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
