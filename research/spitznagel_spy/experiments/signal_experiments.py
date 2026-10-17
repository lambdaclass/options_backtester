"""Signal-driven tail-hedge experiments on SPX.

Gate *when* puts are bought on a regime signal (run(rebalance_dates=...)), and
optionally harvest on a profit multiple (profit_pct). Compare each vs always-on
and buy & hold.

Hard lessons baked in after the first round produced a +12pp false discovery:
  * FIXED, absolute thresholds — never expanding-median (it makes "high" mean
    "early in the sample", which coincided with the GFC and faked an edge).
  * DECOMPOSE every promising signal — does it hedge *before* crashes, or just
    happen to overlap one? A signal that only fires during the single big crash
    in-sample is hindsight, not prediction.
  * Prefer PRICE-derived signals (Part 1) — point-in-time by construction, no
    macro-revision lookahead, and testable over the full 1996-2025 incl. the
    dot-com slow grind. Macro signals (Part 2) only cover 2007-2025 and the
    Tobin's Q column is a known-broken proxy (see scripts/fetch_signals.py).

DATA: purchased DeltaNeutral ALLSPX, local-only; skips if absent. SPX is
price-only (no dividends); excess vs buy & hold is like-for-like.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from options_portfolio_backtester import (
    BacktestEngine, Direction, OptionType, Stock, Strategy, StrategyLeg,
)
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData, TiingoData,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
DATA = ROOT / "data" / "processed"
OPTS, STK, SIG = DATA / "spx_full_options.parquet", DATA / "spx_full_stocks.csv", DATA / "signals.csv"
OTM_LO, OTM_HI, DTE_LO, DTE_HI, EXIT_DTE, BUDGET = 0.25, 0.30, 90, 180, 30, 0.033
CRASHES = {"GFC": ("2007-10-09", "2009-03-09"), "COVID": ("2020-02-19", "2020-03-23"),
           "2022": ("2022-01-03", "2022-10-12"), "dotcom": ("2000-03-24", "2002-10-09")}


def _ann(s):
    y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) * 100 - 100 if y > 0 else float("nan")


def _maxdd(s):
    cm = s.cummax(); return ((s - cm) / cm).min() * 100


def _sharpe(s):
    d = s.pct_change().dropna()
    return _ann(s) / (d.std() * math.sqrt(252) * 100) if d.std() > 0 else 0.0


def _load(lo, hi):
    opt = pd.read_parquet(OPTS); stk = pd.read_csv(STK, parse_dates=["date"])
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    opt = opt[(opt["quotedate"] >= lo) & (opt["quotedate"] <= hi)]
    stk = stk[(stk["date"] >= lo) & (stk["date"] <= hi)]
    shared = set(opt["quotedate"]) & set(stk["date"])
    opt = opt[opt["quotedate"].isin(shared)].sort_values("quotedate", kind="stable")
    stk = stk[stk["date"].isin(shared)].sort_values("date", kind="stable")
    td = tempfile.mkdtemp()
    opt.to_parquet(f"{td}/o.parquet", index=False); stk.to_csv(f"{td}/s.csv", index=False)
    o, s = HistoricalOptionsData(f"{td}/o.parquet"), TiingoData(f"{td}/s.csv")
    shutil.rmtree(td)  # providers load eagerly; don't leave a chain copy in /tmp
    return o, s, stk


def _bimonthly_dates(stk):
    d = pd.DataFrame(index=pd.DatetimeIndex(sorted(stk["date"])))
    return list(pd.to_datetime(
        d.groupby(pd.Grouper(freq="2BMS")).apply(lambda x: x.index.min()).dropna().values))


def _run(o, s, profit_pct=math.inf, rebalance_dates=None):
    sch = o.schema
    bt = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
    bt.options_budget_annual_pct = BUDGET
    bt.check_exits_daily = True; bt.rebalance_stocks_on_exit = True; bt.assert_invariants = True
    bt.stocks = [Stock("SPX", 1.0)]; bt.stocks_data = s; bt.options_data = o
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (sch.underlying == "SPX") & (sch.dte >= DTE_LO) & (sch.dte <= DTE_HI)
        & (sch.strike <= sch.underlying_last * (1 - OTM_LO))
        & (sch.strike >= sch.underlying_last * (1 - OTM_HI)))
    leg.entry_sort = ("strike", False); leg.exit_filter = sch.dte <= EXIT_DTE
    st = Strategy(sch); st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=profit_pct, loss_pct=math.inf)
    bt.options_strategy = st
    bt.run(rebalance_freq=2, rebalance_unit="BMS", rebalance_dates=rebalance_dates)
    return bt


def _price_signals(stk):
    """Point-in-time price-derived signals on the SPX index series."""
    p = stk.set_index("date")["adjClose"].sort_index()
    ret = p.pct_change()
    df = pd.DataFrame(index=p.index)
    df["px"] = p
    df["ma200"] = p.rolling(200).mean()
    df["hi252"] = p.rolling(252).max()
    df["dd"] = p / df["hi252"] - 1.0                       # drawdown from 1y high (<=0)
    df["rvol"] = ret.rolling(60).std() * math.sqrt(252)    # 60d realized vol, annualized
    df["rvol_med3y"] = df["rvol"].rolling(756).median()
    return df


def _report(title, o, s, bh, cands, variants):
    """variants: list of (label, mask_at_cands_or_None, profit_pct)."""
    cs = pd.Series(cands); bh_ann = _ann(bh)
    print(f"\n=== {title} — buy&hold {bh_ann:+.2f}%/yr, maxDD {_maxdd(bh):.1f}%, Sharpe {_sharpe(bh):.3f} ===")
    print(f"  {'strategy':30s} {'CAGR%':>7s} {'Excess':>7s} {'MaxDD%':>7s} {'Sharpe':>7s} {'on%':>5s} {'trd':>5s}")
    out = {}
    for label, mask, pt in variants:
        if mask is None:
            rd = None; onpct = 100
        else:
            m = mask.reindex(pd.DatetimeIndex(cands)).fillna(False)
            rd = list(cs[m.values]); onpct = int(round(100 * m.mean()))
        bt = _run(o, s, profit_pct=pt, rebalance_dates=rd)
        b = bt.balance["total capital"]; a = _ann(b)
        out[label] = rd
        print(f"  {label:30s} {a:>7.2f} {a-bh_ann:>+7.2f} {_maxdd(b):>7.1f} {_sharpe(b):>7.3f} "
              f"{onpct:>5d} {len(bt.trade_log):>5d}", flush=True)
    return out


def _decompose(name, dates):
    dates = pd.DatetimeIndex(sorted(dates))
    print(f"\n  [decompose] '{name}': {len(dates)} hedge-dates — pre-crash (12mo) / during:")
    for cn, (lo, hi) in CRASHES.items():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        pre = dates[(dates >= lo - pd.Timedelta(days=365)) & (dates < lo)]
        dur = dates[(dates >= lo) & (dates <= hi)]
        print(f"    {cn:7s} {lo.date()}: {len(pre)} before / {len(dur)} during")


def main():
    if not OPTS.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")

    # ===== PART 1: price-derived signals, FULL 1996-2025 (incl. dot-com) =====
    o, s, stk = _load("1996-01-01", "2025-12-31")
    bh = stk.set_index("date")["adjClose"]
    cands = _bimonthly_dates(stk)
    ps = _price_signals(stk)
    at = lambda col: ps[col].reindex(pd.DatetimeIndex(cands), method="ffill")
    px, ma, dd, rvol, rvolmed = at("px"), at("ma200"), at("dd"), at("rvol"), at("rvol_med3y")

    sigs_full = {
        "below 200d MA":        px < ma,
        "above 200d MA":        px > ma,
        "drawdown > 10%":       dd < -0.10,
        "within 3% of 1y high": dd > -0.03,
        "realized-vol LOW":     rvol < rvolmed,
        "realized-vol HIGH":    rvol > rvolmed,
    }
    sel = _report("SPX 1996-2025 PRICE signals", o, s, bh, cands,
                  [("always-on", None, math.inf)]
                  + [(k, v, math.inf) for k, v in sigs_full.items()])
    for k in ("below 200d MA", "realized-vol LOW", "within 3% of 1y high"):
        _decompose(k, sel[k])

    # ===== PART 2: macro signals, 2007-2025, FIXED thresholds =====
    o2, s2, stk2 = _load("2007-01-01", "2025-12-31")
    bh2 = stk2.set_index("date")["adjClose"]
    cands2 = _bimonthly_dates(stk2)
    sig = pd.read_csv(SIG, parse_dates=["date"]).set_index("date").sort_index()
    sig["tobin_q"] = sig["nfc_equity_mv"] / sig["nfc_net_worth"]   # corrected formula (proxy)
    at2 = lambda col: sig[col].reindex(pd.DatetimeIndex(cands2), method="ffill")
    vix, buff, hy, curve = at2("vix"), at2("buffett_indicator"), at2("hy_spread"), at2("yield_curve_10y2y")

    sigs_macro = {
        "VIX < 15 (very calm)": vix < 15,
        "VIX < 18":             vix < 18,
        "VIX > 25 (stress)":    vix > 25,
        "Buffett > 150":        buff > 150,
        "Buffett > 180":        buff > 180,
        "HY spread < 4 (tight)": hy < 4,
        "curve inverted (<0)":  curve < 0,
    }
    selm = _report("SPX 2007-2025 MACRO signals (fixed thresholds)", o2, s2, bh2, cands2,
                   [("always-on", None, math.inf)]
                   + [(k, v, math.inf) for k, v in sigs_macro.items()])
    for k in ("VIX < 15 (very calm)", "Buffett > 180", "HY spread < 4 (tight)"):
        _decompose(k, selm[k])


if __name__ == "__main__":
    main()
