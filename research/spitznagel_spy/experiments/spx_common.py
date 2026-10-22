"""Shared setup for the calendar-scored SPX tail-hedge experiments.

Every experiment runs the same machinery: load the cleaned SPX chain, keep
SPX puts on the dates shared with the index, build a put leg from an OTM band
and a DTE window, run one engine per job in a process pool, and score each
balance curve the same way. This module holds that machinery so each
experiment script only states its pre-registered grid and its report.

Needs the local licensed SPX parquet under data/processed/.
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
FINDINGS = Path(__file__).resolve().parents[1] / "findings"
TBILL = ROOT / "data" / "processed" / "dtb3.csv"
TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
TOBIN_Q = ROOT / "data" / "processed" / "tobin_q.csv"
SHILLER = ROOT / "data" / "processed" / "shiller_dividends.csv"
SHILLER_URL = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
               "1449043e-a1ed-4e2e-8e9c-7b9a2a064493/ie_data.xls")

START, END = "1996-01-01", "2025-12-31"
GFC = (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-06-30"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
FILL_FLOOR = 0.90

# (schedule, label, months between entries, month parity, trading-day offset).
# Every calendar also enters on the first trading day (see `schedule`).
CALENDARS = [
    ("bimonthly", "odd/1st", 2, 0, 0), ("bimonthly", "even/1st", 2, 1, 0),
    ("bimonthly", "odd/10th", 2, 0, 9), ("bimonthly", "even/10th", 2, 1, 9),
    ("monthly", "1st", 1, 0, 0), ("monthly", "10th", 1, 0, 9),
]
BIMONTHLY = [c for c in CALENDARS if c[0] == "bimonthly"]

W = {}  # per-worker state: "o"/"s" data providers plus the caller's extras


# ---------------------------------------------------------------- metrics

def ann(s):
    """Annualized growth of a balance curve, in percent."""
    y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) * 100 - 100 if y > 0 else float("nan")


def maxdd(s):
    cm = s.cummax()
    return ((s - cm) / cm).min() * 100


def sharpe(s):
    d = s.pct_change().dropna()
    return ann(s) / (d.std() * math.sqrt(252) * 100) if d.std() > 0 else 0.0


def wdd(b, lo, hi):
    """Max drawdown inside [lo, hi]."""
    s = b.loc[(b.index >= lo) & (b.index <= hi)]
    return maxdd(s) if len(s) > 2 else float("nan")


def metrics(b, bh_cagr):
    """The standard score of one run's total-capital curve."""
    return {"cagr": ann(b), "excess": ann(b) - bh_cagr, "dd": maxdd(b), "sharpe": sharpe(b),
            "gfc": wdd(b, *GFC), "covid": wdd(b, *COVID)}


# ---------------------------------------------------------------- data

def schedule(days, step, parity, offset):
    """Entry dates: the offset-th trading day of every step-th month, plus the
    first trading day so every calendar starts invested."""
    days = pd.Series(pd.DatetimeIndex(sorted(days)))
    months = days.dt.year * 12 + days.dt.month
    keep = (months - months.iloc[0]) % step == parity
    dates = [g.iloc[min(offset, len(g) - 1)] for _, g in days[keep].groupby(months[keep])]
    return dates if dates[0] == days.iloc[0] else [days.iloc[0]] + dates


def load_spx(start=START, end=END):
    """(puts, stocks) on shared dates: cleaned SPX puts and the index."""
    if not OPTS.exists() or not STK.exists():
        raise SystemExit("Local SPX data absent (licensed; not in repo).")
    opt = clean_spx_chain(pd.read_parquet(OPTS))
    stk = pd.read_csv(STK, parse_dates=["date"])
    opt["quotedate"] = pd.to_datetime(opt["quotedate"])
    opt["expiration"] = pd.to_datetime(opt["expiration"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    opt = opt[(opt["quotedate"] >= lo) & (opt["quotedate"] <= hi)]
    stk = stk[(stk["date"] >= lo) & (stk["date"] <= hi)]
    shared = set(opt["quotedate"]) & set(stk["date"])
    opt = opt[opt["quotedate"].isin(shared)]
    stk = stk[stk["date"].isin(shared)].sort_values("date", kind="stable")
    # SPX puts only; never filter by strike/DTE (held puts need every row until
    # exit -- see parallel_sweep.py).
    puts = opt[(opt["type"].astype(str).str.lower().str[0] == "p")
               & (opt["underlying"] == "SPX")].sort_values("quotedate", kind="stable")
    return puts, stk


def tbill():
    """Daily 3-month T-bill rate (decimal), cached from FRED DTB3."""
    if not TBILL.exists():
        pd.read_csv(TBILL_URL).to_csv(TBILL, index=False)
    r = pd.read_csv(TBILL, parse_dates=["observation_date"], na_values=".")
    return r.set_index("observation_date")["DTB3"].ffill() / 100


def tobin_q(lag_days=75):
    """Tobin's Q as it was KNOWN on each date: market value of nonfinancial
    corporate equities / their net worth (Fed Z.1 via FRED NCBEILQ027S and
    TNWMVBSNNCB), quarterly, re-indexed to quarter end + `lag_days` (the Z.1
    publication lag). Cached in data/processed/tobin_q.csv."""
    if not TOBIN_Q.exists():
        def fred(sid):
            r = pd.read_csv(FRED_CSV.format(sid), parse_dates=["observation_date"])
            return r.set_index("observation_date").iloc[:, 0]
        (fred("NCBEILQ027S") / fred("TNWMVBSNNCB")).dropna().rename("q").to_csv(TOBIN_Q)
    q = pd.read_csv(TOBIN_Q, parse_dates=["observation_date"], index_col=0)["q"]
    q.index = q.index + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=lag_days)
    return q


def q_gate_on(dates, pct):
    """For each date, whether known Q >= its expanding `pct` quantile (all
    history since 1945 known at that date)."""
    q = tobin_q()
    thr = q.expanding().quantile(pct)
    idx = pd.DatetimeIndex(dates)
    known = q.reindex(idx, method="ffill")
    return pd.Series((known >= thr.reindex(idx, method="ffill")).to_numpy(), index=idx)


def shiller_dividends():
    """Monthly S&P composite dividends (trailing 12m, index points) from
    Robert Shiller's ie_data. Cached in data/processed/shiller_dividends.csv;
    the first download needs xlrd (`uv run --with xlrd ...` or pip)."""
    if not SHILLER.exists():
        raw = pd.read_excel(SHILLER_URL, sheet_name="Data", skiprows=7).iloc[:, :3]
        raw.columns = ["date", "P", "D"]
        raw = raw[pd.to_numeric(raw["D"], errors="coerce").notna()]
        year = raw["date"].astype(float).astype(int)
        month = ((raw["date"].astype(float) - year) * 100).round().astype(int)
        out = pd.DataFrame({"month": pd.to_datetime(dict(year=year, month=month, day=1)),
                            "D": raw["D"].astype(float)})
        out.to_csv(SHILLER, index=False)
    return pd.read_csv(SHILLER, parse_dates=["month"]).set_index("month")["D"]


def with_total_return(stk):
    """`stk` with adjClose replaced by a total-return SPX index (dividends
    reinvested daily: D/252 per trading day of each month). close stays the
    price index -- option strikes and intrinsic value use it."""
    d = shiller_dividends()
    out = stk.sort_values("date").copy()
    month = out["date"].dt.to_period("M").dt.to_timestamp()
    daily_div = month.map(d).ffill().to_numpy() / 252
    px = out["close"].to_numpy()
    growth = (px[1:] + daily_div[1:]) / px[:-1]
    tr = px[0] * pd.Series([1.0, *growth]).cumprod().to_numpy()
    out["adjClose"] = tr
    return out


# ---------------------------------------------------------------- engine

def put_strategy(sch, otm, dte, exit_dte, monetize=None, profit_x=math.inf):
    """Long SPX puts `otm` = (lo, hi) below spot, `dte` = (lo, hi) days out,
    nearest-to-spot strike first; sold at DTE <= exit_dte, once the strike is
    within `monetize` of spot (strike >= m * spot), or at (1 + profit_x) x
    premium."""
    from options_portfolio_backtester import Strategy, StrategyLeg
    from options_portfolio_backtester.core.types import Direction, OptionType
    (olo, ohi), (dlo, dhi) = otm, dte
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (sch.underlying == "SPX") & (sch.dte >= dlo) & (sch.dte <= dhi)
        & (sch.strike <= sch.underlying_last * (1 - olo))
        & (sch.strike >= sch.underlying_last * (1 - ohi))
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = sch.dte <= exit_dte
    if monetize is not None:
        leg.exit_filter = leg.exit_filter | (sch.strike >= sch.underlying_last * monetize)
    st = Strategy(sch)
    st.add_leg(leg)
    st.add_exit_thresholds(profit_pct=profit_x, loss_pct=math.inf)
    return st


def backtest(strategy, dates, annual=None, per_entry=None, leverage=1.0, roll=None,
             fill_model=None):
    """Run SPX + `strategy` on the worker's data, entering on `dates`.

    The premium budget is `annual` (%/yr, split over the schedule) or
    `per_entry` (% of NAV at each entry), spent in full each time. Stocks are
    `leverage` x capital, financed by negative cash (no interest: callers that
    lever adjust for financing themselves). `roll` is the engine's
    roll_exit_proceeds_pct; `fill_model` defaults to bid/ask."""
    from options_portfolio_backtester import BacktestEngine, Stock
    kwargs = {"fill_model": fill_model} if fill_model is not None else {}
    bt = BacktestEngine({"stocks": leverage, "options": 0.0, "cash": 1.0 - leverage},
                        initial_capital=1_000_000, **kwargs)
    bt.options_budget_annual_pct = annual
    bt.options_budget_per_rebalance_pct = per_entry
    bt.options_budget_fresh_spend = True
    bt.roll_exit_proceeds_pct = roll
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.assert_invariants = True
    bt.stocks = [Stock("SPX", 1.0)]
    bt.stocks_data = W["s"]
    bt.options_data = W["o"]
    bt.options_strategy = strategy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.run(rebalance_dates=dates)
    return bt


# ---------------------------------------------------------------- pool

def n_workers():
    """Worker count from argv[1], else up to 8."""
    return int(sys.argv[1]) if len(sys.argv) > 1 else min(8, max(1, (os.cpu_count() or 4) - 1))


def _init(op, sp, extra):
    """Load data once per worker process."""
    warnings.simplefilter("ignore")
    # Fill shortfalls are reported in the tables; silence the engine's per-run log.
    logging.getLogger("options_portfolio_backtester.engine.engine").setLevel(logging.ERROR)
    from options_portfolio_backtester import HistoricalOptionsData, TiingoData
    W["o"] = HistoricalOptionsData(op)
    W["s"] = TiingoData(sp)
    W.update(extra)


def run_pool(fn, jobs, puts, stk, workers, extra=None, every=20):
    """Map `fn` over `jobs` in `workers` processes. Workers read the data from
    temp files (fork-safe, loaded once each); `extra` lands in W. Results come
    back in job order, so reruns write identical files; a job may return a
    dict or a list of them."""
    results = []
    with tempfile.TemporaryDirectory() as td:
        op, sp = os.path.join(td, "o.parquet"), os.path.join(td, "s.csv")
        puts.to_parquet(op, index=False)
        stk.to_csv(sp, index=False)
        with Pool(workers, initializer=_init, initargs=(op, sp, extra or {})) as pool:
            for i, r in enumerate(pool.imap(fn, jobs, chunksize=1), 1):
                results += r if isinstance(r, list) else [r]
                if i % every == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done", flush=True)
    print()
    return results


def write(df, name):
    out = FINDINGS / name
    df.to_csv(out, index=False)
    print(f"All {len(df)} rows written to {out.relative_to(ROOT)}\n")
