"""SPX tail-hedge configuration sweep, full 30-year history (1996-2025).

Sweeps OTM depth x entry-DTE band x annual budget over the entire SPX series
(which contains the dot-com slow grind, the GFC sharp crash, COVID's flash
crash, and the 2022 bear), vs buy & hold, with runtime invariants armed.
Prints a per-config table and a leaderboard sorted by excess CAGR.

Pass a regime name as argv[1] to sweep a sub-window instead of the full
period (e.g. `python spx_sweep.py gfc`). Regimes: full, dotcom, gfc, covid,
bear2022, calm.

DATA: purchased DeltaNeutral ALLSPX, local-only (data/processed/spx_full_*).
Licensed — never committed or redistributed. Skips cleanly if absent.
SPX is a price index; the synthetic stock series carries no dividends, so
CAGR is ~1.8%/yr low in absolute terms — but every overlay is compared to
buy & hold on the same series, so excess CAGR is like-for-like.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from options_portfolio_backtester import (
    BacktestEngine, Direction, OptionType, Stock, Strategy, StrategyLeg,
)
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData, TiingoData,
)

DATA = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()) / "data" / "processed"
OPTS = DATA / "spx_full_options.parquet"
STK = DATA / "spx_full_stocks.csv"

REGIMES = {
    "full":     ("1996-01-01", "2025-12-31"),
    "dotcom":   ("2000-01-01", "2003-12-31"),
    "gfc":      ("2007-06-01", "2009-12-31"),
    "covid":    ("2019-01-01", "2021-12-31"),
    "bear2022": ("2022-01-01", "2023-12-31"),
    "calm":     ("2012-01-01", "2019-12-31"),
}

OTM_BANDS = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30),
             (0.30, 0.35), (0.35, 0.40), (0.40, 0.45)]
# (label, entry_dte_lo, entry_dte_hi, exit_dte)
DTE_CONFIGS = [
    ("30-60/14",   30,  60, 14),
    ("60-120/30",  60, 120, 30),
    ("90-180/30",  90, 180, 30),
    ("180-365/60", 180, 365, 60),
]
BUDGETS = [0.01, 0.033]


def _ann(s):
    y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) * 100 - 100 if y > 0 else float("nan")


def _maxdd(s):
    cm = s.cummax()
    return ((s - cm) / cm).min() * 100


def _sharpe(s):
    d = s.pct_change().dropna()
    return _ann(s) / (d.std() * math.sqrt(252) * 100) if d.std() > 0 else 0.0


def _run(opts_data, stocks_data, otm_lo, otm_hi, dte_lo, dte_hi, exit_dte, budget):
    sch = opts_data.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0},
                        initial_capital=1_000_000)
    bt.options_budget_annual_pct = budget
    bt.options_budget_fresh_spend = True  # spend the full budget each rebalance
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.assert_invariants = True
    bt.stocks = [Stock("SPX", 1.0)]
    bt.stocks_data = stocks_data
    bt.options_data = opts_data
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (sch.underlying == "SPX") & (sch.dte >= dte_lo) & (sch.dte <= dte_hi)
        & (sch.strike <= sch.underlying_last * (1 - otm_lo))
        & (sch.strike >= sch.underlying_last * (1 - otm_hi))
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = sch.dte <= exit_dte
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    bt.options_strategy = st
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fill-rate reported explicitly below
        bt.run(rebalance_freq=2, rebalance_unit="BMS")
    b = bt.balance["total capital"]
    return _ann(b), _maxdd(b), _sharpe(b), len(bt.trade_log), bt.option_fill_rate


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "full"
    lo, hi = REGIMES[regime]
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")

    opt = pd.read_parquet(OPTS)
    stk = pd.read_csv(STK, parse_dates=["date"])
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
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
    bh_ann = _ann(bh)
    print(f"=== SPX {regime} {lo.date()}..{hi.date()}  ({len(opt):,} rows, {len(stk):,} days) ===")
    print(f"buy & hold: CAGR {bh_ann:+.2f}%  maxDD {_maxdd(bh):.1f}%  Sharpe {_sharpe(bh):.3f}\n")
    print(f"  {'OTM':9s} {'DTE/exit':12s} {'bud':>5s} {'CAGR%':>8s} {'Excess':>8s} {'MaxDD%':>8s} {'Sharpe':>7s} {'trd':>5s} {'fill':>5s}")

    results = []
    for (olo, ohi) in OTM_BANDS:
        for (dlabel, dlo, dhi, edte) in DTE_CONFIGS:
            for bud in BUDGETS:
                a, dd, sh, n, fill = _run(o, s, olo, ohi, dlo, dhi, edte, bud)
                ex = a - bh_ann
                flag = "" if fill >= 0.9 else " LOW-FILL"  # NaN = no entries
                results.append((ex, a, dd, sh, n, f"{round(olo*100)}-{round(ohi*100)}%", dlabel, bud, fill))
                print(f"  {round(olo*100)}-{round(ohi*100)}%   {dlabel:12s} {bud*100:>4.1f} "
                      f"{a:>8.2f} {ex:>+8.2f} {dd:>8.1f} {sh:>7.3f} {n:>5d} {fill:>4.0%}{flag}", flush=True)

    # Only configs that actually filled (>=90%) are a trustworthy hedged result;
    # the rest are partial buy-and-hold because the requested band is too deep
    # for the chain in this window (see findings/DATA_COVERAGE.md).
    trustworthy = [r for r in results if r[8] >= 0.9]
    print(f"\n=== TOP 10 by excess CAGR (regime={regime}, fill>=90% only) ===")
    for ex, a, dd, sh, n, otm, dlabel, bud, fill in sorted(trustworthy, reverse=True)[:10]:
        print(f"  {otm:8s} {dlabel:12s} {bud*100:>4.1f}%  excess {ex:+.2f}pp  CAGR {a:+.2f}%  DD {dd:.1f}%  Sharpe {sh:.3f}  fill {fill:.0%}")
    print(f"\n=== TOP 10 by Sharpe (fill>=90% only) ===")
    for ex, a, dd, sh, n, otm, dlabel, bud, fill in sorted(trustworthy, key=lambda r: r[3], reverse=True)[:10]:
        print(f"  {otm:8s} {dlabel:12s} {bud*100:>4.1f}%  Sharpe {sh:.3f}  excess {ex:+.2f}pp  DD {dd:.1f}%  fill {fill:.0%}")


if __name__ == "__main__":
    main()
