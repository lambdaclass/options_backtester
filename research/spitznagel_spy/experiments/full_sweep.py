#!/usr/bin/env python3
"""Honest full-history config sweep for the short-dated put overlay.

Encodes the lessons from the per-crash investigation (see findings/):

  1. Judge over the FULL 1996-2025 history, not cherry-picked crash windows.
     Per-crash tuning overfits — the exit-timing rule that won COVID lost on
     1998/2015 (findings/EXIT_AND_DEPTH.md). The only honest metric is how a
     fixed config does across everything, since you can't predict the crash.
  2. Entry DTE <= 90 (operational constraint — no long-dated LEAPs).
  3. Gate on hedge fill-rate >= 90%. Deep-OTM / short-DTE bands the chain can't
     supply silently degrade to partial buy-and-hold; those rows are flagged
     and excluded from the rankings.
  4. The exit-DTE threshold is a real lever, so sweep it; report coverage (the
     fraction of trading days actually hedged) because a calendar entry + DTE
     exit leaves naked gaps.
  5. Decompose: report drawdown *during GFC and COVID* (the crashes deep enough
     to matter) alongside the full-period drag, so we can see which configs buy
     real crash protection without bleeding too much the rest of the time.

Each config is run ONCE over full history; the GFC and COVID sub-windows are
sliced from that single continuous run (no separate per-window backtests, so
coverage and carry are accounted for honestly).

Usage:  python research/spitznagel_spy/experiments/full_sweep.py
Needs the local licensed SPX parquet under data/processed/ (skips if absent).
"""

import math
import warnings
from pathlib import Path

import pandas as pd

from options_portfolio_backtester import (
    BacktestEngine, HistoricalOptionsData, TiingoData, Stock, Strategy, StrategyLeg,
)
from options_portfolio_backtester.core.types import OptionType, Direction

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OPTS = ROOT / "data" / "processed" / "spx_full_options.parquet"
STK = ROOT / "data" / "processed" / "spx_full_stocks.csv"

FULL = ("1996-01-01", "2025-12-31")
# Sub-windows for crash decomposition (the two crashes deep enough to pay).
GFC = (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-06-30"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))

# --- the grid ---
# DTE spans both the operational <=90 range AND the longer 90-180 range, because
# the only positive full-history excess lives at 90-180 (the duration lever — a
# 2x2 showed 90-180 bi-monthly = +0.95pp vs 60-90 = -0.91pp). Including both makes
# the cost of the <=90 constraint explicit rather than hidden.
OTM_BANDS = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30)]
ENTRY_DTE = [("30-60", 30, 60), ("60-90", 60, 90), ("90-180", 90, 180), ("120-180", 120, 180)]
EXIT_DTE = [30, 14]              # days-to-expiry at sale (longer hold = lower number)
BUDGETS = [0.01, 0.033]
REBALANCE_FREQ = 2               # bi-monthly: less premium churn than monthly
                                 # (monthly rolling roughly halved the excess —
                                 #  more entries = more bid/ask + decay paid)
FILL_FLOOR = 0.90


def _ann(s):
    y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) * 100 - 100 if y > 0 else float("nan")


def _maxdd(s):
    cm = s.cummax()
    return ((s - cm) / cm).min() * 100


def _sharpe(s):
    d = s.pct_change().dropna()
    return _ann(s) / (d.std() * math.sqrt(252) * 100) if d.std() > 0 else 0.0


def _window_dd(bal, lo, hi):
    sub = bal.loc[(bal.index >= lo) & (bal.index <= hi)]
    return _maxdd(sub) if len(sub) > 2 else float("nan")


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


def _run(o, s, stocks_bh, olo, ohi, dlo, dhi, exit_dte, budget):
    sch = o.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
    bt.options_budget_annual_pct = budget
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
    leg.exit_filter = sch.dte <= exit_dte
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # fill-rate reported explicitly
        bt.run(rebalance_freq=REBALANCE_FREQ, rebalance_unit="BMS")
    bal = bt.balance["total capital"]
    return {
        "cagr": _ann(bal), "dd": _maxdd(bal), "sharpe": _sharpe(bal),
        "fill": bt.option_fill_rate, "coverage": _coverage(bt.trade_log, bt.balance.index),
        "gfc_dd": _window_dd(bt.balance["total capital"], *GFC),
        "covid_dd": _window_dd(bt.balance["total capital"], *COVID),
    }


def main():
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    opt = pd.read_parquet(OPTS)
    stk = pd.read_csv(STK, parse_dates=["date"])
    lo, hi = pd.Timestamp(FULL[0]), pd.Timestamp(FULL[1])
    opt = opt[(opt["quotedate"] >= lo) & (opt["quotedate"] <= hi)]
    stk = stk[(stk["date"] >= lo) & (stk["date"] <= hi)]
    shared = set(opt["quotedate"]) & set(stk["date"])
    opt = opt[opt["quotedate"].isin(shared)].sort_values("quotedate", kind="stable")
    stk = stk[stk["date"].isin(shared)].sort_values("date", kind="stable")

    import os, shutil, tempfile
    td = tempfile.mkdtemp()
    op = os.path.join(td, "o.parquet"); sp = os.path.join(td, "s.csv")
    opt.to_parquet(op, index=False); stk.to_csv(sp, index=False)
    o = HistoricalOptionsData(op); s = TiingoData(sp)
    shutil.rmtree(td)  # providers load eagerly; don't leave a chain copy in /tmp

    bh = stk.set_index("date")["adjClose"]
    bh_cagr, bh_dd = _ann(bh), _maxdd(bh)
    bh_gfc_dd = _maxdd(bh.loc[(bh.index >= GFC[0]) & (bh.index <= GFC[1])])
    bh_covid_dd = _maxdd(bh.loc[(bh.index >= COVID[0]) & (bh.index <= COVID[1])])

    print(f"=== Full-history sweep 1996-2025 (bi-monthly rebalance) ===")
    print(f"Buy & hold: CAGR {bh_cagr:+.2f}%  maxDD {bh_dd:.1f}%  "
          f"(GFC DD {bh_gfc_dd:.1f}%, COVID DD {bh_covid_dd:.1f}%)\n")
    hdr = (f"  {'OTM':7s} {'entry':6s} {'exit':>4s} {'bud':>5s} {'CAGR':>7s} {'excess':>7s} "
           f"{'DD':>6s} {'Sharpe':>7s} {'cover':>5s} {'fill':>5s} {'GFCdd':>6s} {'CVDdd':>6s}")
    print(hdr)
    rows = []
    for (olo, ohi) in OTM_BANDS:
        for (dlbl, dlo, dhi) in ENTRY_DTE:
            for exit_dte in EXIT_DTE:
                if exit_dte >= dhi:      # exit threshold must be below entry DTE
                    continue
                for bud in BUDGETS:
                    r = _run(o, s, bh, olo, ohi, dlo, dhi, exit_dte, bud)
                    ex = r["cagr"] - bh_cagr
                    flag = "" if r["fill"] >= FILL_FLOOR else " LOW"  # NaN = no entries
                    rows.append((ex, r, f"{round(olo*100)}-{round(ohi*100)}%", dlbl, exit_dte, bud))
                    print(f"  {round(olo*100)}-{round(ohi*100)}%  {dlbl:6s} {exit_dte:>4d} {bud*100:>4.1f} "
                          f"{r['cagr']:>7.2f} {ex:>+7.2f} {r['dd']:>6.1f} {r['sharpe']:>7.3f} "
                          f"{r['coverage']:>4.0%} {r['fill']:>4.0%} {r['gfc_dd']:>6.1f} {r['covid_dd']:>6.1f}{flag}",
                          flush=True)

    trust = [t for t in rows if t[1]["fill"] >= FILL_FLOOR]
    print(f"\n=== TOP 8 by full-period Sharpe (fill>=90%) ===")
    for ex, r, otm, dlbl, edte, bud in sorted(trust, key=lambda t: t[1]["sharpe"], reverse=True)[:8]:
        print(f"  {otm:7s} {dlbl} exit{edte} {bud*100:.1f}%  Sharpe {r['sharpe']:.3f}  "
              f"excess {ex:+.2f}pp  DD {r['dd']:.1f}%  GFCdd {r['gfc_dd']:.1f}%  CVDdd {r['covid_dd']:.1f}%")
    print(f"\n=== TOP 8 by full-period excess CAGR (fill>=90%) ===")
    for ex, r, otm, dlbl, edte, bud in sorted(trust, reverse=True)[:8]:
        print(f"  {otm:7s} {dlbl} exit{edte} {bud*100:.1f}%  excess {ex:+.2f}pp  "
              f"Sharpe {r['sharpe']:.3f}  DD {r['dd']:.1f}%  GFCdd {r['gfc_dd']:.1f}%  CVDdd {r['covid_dd']:.1f}%")
    print(f"\n=== BEST crash protection: shallowest avg(GFC,COVID) DD (fill>=90%) ===")
    # DDs are negative; least-negative = best protection, so sort descending.
    for ex, r, otm, dlbl, edte, bud in sorted(trust, key=lambda t: (t[1]["gfc_dd"]+t[1]["covid_dd"])/2, reverse=True)[:8]:
        print(f"  {otm:7s} {dlbl} exit{edte} {bud*100:.1f}%  GFCdd {r['gfc_dd']:.1f}%  "
              f"CVDdd {r['covid_dd']:.1f}%  (B&H {bh_gfc_dd:.0f}/{bh_covid_dd:.0f})  excess {ex:+.2f}pp  Sharpe {r['sharpe']:.3f}")


if __name__ == "__main__":
    main()
