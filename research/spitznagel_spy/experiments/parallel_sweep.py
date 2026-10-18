#!/usr/bin/env python3
"""Parallel full-history config sweep (multiprocessing across CPU cores).

The per-config backtests are independent, so this fans them out over a process
pool. Each worker loads the SPX put chain (calls dropped, ~half the full file)
once via the pool initializer and reuses it across all its configs, so the big
parquet is read once per worker, not once per config.

Encodes the validated lessons: full-history 1996-2025, bi-monthly, fill-gated,
crash-decomposed (GFC/COVID drawdown). Grid is broad by design — refine OTM
around the 25-30% optimum, DTE around the 90-180 winner, plus exit-DTE and
budget dials.

Caveat: ranking ~400 configs on one history is in-sample selection. Treat the
top rows as candidates and confirm them with start_year_robustness.py.

Every config's metrics are written to findings/parallel_sweep_results.csv so the
full surface (e.g. budget curves) can be inspected without rerunning.

Usage:  python research/spitznagel_spy/experiments/parallel_sweep.py [n_workers]
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

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OPTS = ROOT / "data" / "processed" / "spx_full_options.parquet"
STK = ROOT / "data" / "processed" / "spx_full_stocks.csv"
OUT = Path(__file__).resolve().parents[1] / "findings" / "parallel_sweep_results.csv"

START, END = "1996-01-01", "2025-12-31"
GFC = (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-06-30"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
FILL_FLOOR = 0.90

# --- broad grid (focused around the validated optimum) ---
OTM_BANDS = [(0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35)]
DTE_BANDS = [(60, 90), (90, 120), (90, 150), (90, 180), (120, 180)]
EXIT_DTE = [45, 30, 21, 14]
BUDGETS = [0.005, 0.010, 0.020, 0.033, 0.050]

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


def _coverage(trade_log, index):
    """Fraction of trading days with at least one option position held."""
    if trade_log is None or trade_log.empty:
        return 0.0
    fl = trade_log.columns.levels[0][0]
    sub = trade_log[fl][["contract", "order"]].copy()
    sub["date"] = pd.to_datetime(trade_log["totals"]["date"].values)
    ent = sub[sub["order"].astype(str).str.contains("BTO")]
    ex = sub[~sub["order"].astype(str).str.contains("BTO")]
    held = pd.Series(False, index=index)
    for _, e in ent.iterrows():
        m = ex[ex["contract"] == e["contract"]]
        xd = m["date"].iloc[0] if len(m) else index[-1]
        held |= (index >= e["date"]) & (index <= xd)
    return held.mean()


def _init(op, sp, bh_cagr):
    """Load data once per worker process."""
    # Fill shortfalls are reported in the table; silence the engine's per-run log.
    logging.getLogger("options_portfolio_backtester.engine.engine").setLevel(logging.ERROR)
    warnings.simplefilter("ignore")
    from options_portfolio_backtester import HistoricalOptionsData, TiingoData
    _W["o"] = HistoricalOptionsData(op)
    _W["s"] = TiingoData(sp)
    _W["bh"] = bh_cagr


def _run(cfg):
    from options_portfolio_backtester import BacktestEngine, Stock, Strategy, StrategyLeg
    from options_portfolio_backtester.core.types import Direction, OptionType
    olo, ohi, dlo, dhi, edte, bud = cfg
    o, s, bh = _W["o"], _W["s"], _W["bh"]
    sch = o.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
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
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.run(rebalance_freq=2, rebalance_unit="BMS")
    b = bt.balance["total capital"]
    return {
        "otm": f"{round(olo*100)}-{round(ohi*100)}%", "dte": f"{dlo}-{dhi}", "exit": edte, "bud": bud,
        "cagr": _ann(b), "excess": _ann(b) - bh, "dd": _maxdd(b), "sharpe": _sharpe(b),
        "fill": bt.option_fill_rate, "coverage": _coverage(bt.trade_log, bt.balance.index),
        "gfc": _wdd(b, *GFC), "covid": _wdd(b, *COVID),
    }


def main():
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    # Cap workers to bound memory — each loads the full SPX-put chain.
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

    # Filter to SPX puts ONLY. This is the one safe reduction: the strategy
    # trades and holds only SPX puts, so dropping calls/other underlyings can't
    # affect entry, mark-to-market, or exit. Do NOT filter by strike or DTE —
    # a held put must remain in the data as its DTE decays to the exit threshold
    # and as its moneyness drifts with the market, or exits silently never fire
    # (an earlier strike/DTE pre-filter did exactly that and corrupted results).
    puts = opt[(opt["type"].astype(str).str.lower().str[0] == "p")
               & (opt["underlying"] == "SPX")].sort_values("quotedate", kind="stable")

    bh = stk.set_index("date")["adjClose"]
    bh_cagr, bh_dd = _ann(bh), _maxdd(bh)
    bh_gfc_dd, bh_covid_dd = _wdd(bh, *GFC), _wdd(bh, *COVID)

    # Every exit DTE is below every entry floor, so no combination is invalid.
    grid = [(olo, ohi, dlo, dhi, edte, bud)
            for (olo, ohi) in OTM_BANDS for (dlo, dhi) in DTE_BANDS
            for edte in EXIT_DTE for bud in BUDGETS]

    print(f"=== Parallel sweep: {len(grid)} configs on {n_workers} workers "
          f"(full {START[:4]}-{END[:4]}, bi-monthly) ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%  maxDD {bh_dd:.1f}%  "
          f"(GFC DD {bh_gfc_dd:.1f}%, COVID DD {bh_covid_dd:.1f}%)\n")

    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False); stk.to_csv(sp, index=False)
        del opt, puts, stk, bh  # free the parent's copies; workers read the files
        with Pool(n_workers, initializer=_init, initargs=(op, sp, bh_cagr)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_run, grid, chunksize=4), 1):
                results.append(r)
                if i % 20 == 0 or i == len(grid):
                    print(f"  {i}/{len(grid)} done", flush=True)
        print()

    pd.DataFrame(results).sort_values(["otm", "dte", "exit", "bud"]).to_csv(OUT, index=False)
    print(f"All {len(results)} rows written to {OUT.relative_to(ROOT)}\n")

    trust = [r for r in results if r["fill"] >= FILL_FLOOR]
    print(f"{len(trust)}/{len(results)} configs passed fill>=90%\n")

    def show(title, rows, key, rev=True):
        print(f"=== {title} ===")
        print(f"  {'OTM':7s} {'DTE':8s} {'exit':>4s} {'bud':>5s} {'excess':>7s} "
              f"{'DD':>6s} {'Sharpe':>7s} {'GFCdd':>6s} {'CVDdd':>6s} {'cover':>5s} {'fill':>5s}")
        for r in sorted(rows, key=key, reverse=rev)[:12]:
            print(f"  {r['otm']:7s} {r['dte']:8s} {r['exit']:>4d} {r['bud']*100:>4.1f} "
                  f"{r['excess']:>+7.2f} {r['dd']:>6.1f} {r['sharpe']:>7.3f} "
                  f"{r['gfc']:>6.1f} {r['covid']:>6.1f} {r['coverage']:>4.0%} {r['fill']:>4.0%}")
        print()

    show("TOP 12 by excess CAGR (fill>=90%)", trust, lambda r: r["excess"])
    show("TOP 12 by Sharpe (fill>=90%)", trust, lambda r: r["sharpe"])
    show(f"TOP 12 by crash protection: avg(GFC,COVID) DD (fill>=90%; "
         f"B&H {bh_gfc_dd:.0f}/{bh_covid_dd:.0f})", trust,
         lambda r: (r["gfc"] + r["covid"]) / 2, rev=True)


if __name__ == "__main__":
    main()
