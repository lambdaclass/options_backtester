"""Reproduce every table in the Spitznagel tail-hedge article.

Article: https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/

Usage:
    python research/spitznagel_spy/reproduce_article.py

Prints each table to stdout in the order it appears in the article. Numbers
match what tests/oracles/test_article_reproduction.py pins, with the same engine
configuration:

    - Strike-based 40-45% OTM puts (strike between 55% and 60% of spot)
    - DTE 90-180 at entry, exit when DTE drops to 30
    - Bi-monthly rebalance
    - Daily exit checks, monetize-and-reinvest on
    - Externally-funded put budget on top of 100% SPY
    - No profit target, no IV filter

Requires the SPY parquet at data/processed/options.parquet, fetched via
    python scripts/fetch_data.py all --symbols SPY

Runs in roughly two minutes on a laptop.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

warnings.filterwarnings("ignore")

from options_portfolio_backtester import (
    BacktestEngine,
    Direction,
    OptionType,
    Stock,
    Strategy,
    StrategyLeg,
)
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData,
    TiingoData,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data" / "processed"
OPTIONS_PATH = DATA / "options.parquet"
STOCKS_PATH = DATA / "stocks.csv"

INITIAL_CAPITAL = 1_000_000


def build_article_engine(
    opts_data,
    stocks_data,
    schema,
    *,
    otm_lo: float = 0.40,
    otm_hi: float = 0.45,
    dte_lo: int = 90,
    dte_hi: int = 180,
    exit_dte: int = 30,
    budget: float = 0.033,
    profit_pct: float = math.inf,
) -> BacktestEngine:
    """Configure (but don't run) the article's engine.

    Shared by the table reproduction below and make_figures.py, so the
    figures are generated from the exact pinned configuration.
    """
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget_annual_pct = budget
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks_data
    bt.options_data = opts_data
    leg = StrategyLeg(
        "leg_1", schema,
        option_type=OptionType.PUT, direction=Direction.BUY,
    )
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= dte_lo)
        & (schema.dte <= dte_hi)
        & (schema.strike <= schema.underlying_last * (1 - otm_lo))
        & (schema.strike >= schema.underlying_last * (1 - otm_hi))
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = schema.dte <= exit_dte
    strat = Strategy(schema)
    strat.add_leg(leg)
    strat.add_exit_thresholds(profit_pct=profit_pct, loss_pct=math.inf)
    bt.options_strategy = strat
    return bt


def build_spy_engine(opts_data, stocks_data, schema) -> BacktestEngine:
    """Configure (but don't run) the SPY buy-and-hold baseline."""
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks_data
    bt.options_data = opts_data
    bt.options_strategy = Strategy(schema)
    return bt


def _run(
    opts_data,
    stocks_data,
    schema,
    *,
    rebal_freq: int = 2,
    rebal_unit: str = "BMS",
    **engine_kwargs,
):
    """Run a single backtest. Returns the daily total-capital series."""
    bt = build_article_engine(opts_data, stocks_data, schema, **engine_kwargs)
    bt.run(rebalance_freq=rebal_freq, rebalance_unit=rebal_unit)
    series = bt.balance["total capital"].copy()
    series.index = pd.to_datetime(series.index)
    return series


def _run_spy(opts_data, stocks_data, schema):
    """SPY buy-and-hold baseline."""
    bt = build_spy_engine(opts_data, stocks_data, schema)
    bt.run(rebalance_freq=1, rebalance_unit="BMS")
    series = bt.balance["total capital"].copy()
    series.index = pd.to_datetime(series.index)
    return series


def _ann(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def _maxdd(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    return ((series - series.cummax()) / series.cummax()).min() * 100


def _sharpe(series: pd.Series) -> float:
    daily = series.pct_change().dropna()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return (_ann(series) / (daily.std() * math.sqrt(252) * 100))


def _sortino(series: pd.Series) -> float:
    daily = series.pct_change().dropna()
    down = daily[daily < 0]
    if len(down) < 2 or down.std() == 0:
        return 0.0
    return (_ann(series) / (down.std() * math.sqrt(252) * 100))


def _calmar(series: pd.Series) -> float:
    dd = _maxdd(series)
    return _ann(series) / abs(dd) if dd < 0 else 0.0


def _row(label: str, fields: Iterable):
    print(f"  {label:38s} " + "  ".join(f"{f:>10s}" for f in fields))


def _print_fingerprint():
    """Engine version + commit + data hashes, so every published table is
    traceable to the exact code and the exact bytes that produced it."""
    import subprocess

    from options_portfolio_backtester.analytics.results import hash_data_file
    try:
        from importlib.metadata import version
        engine_version = version("options_portfolio_backtester")
    except Exception:
        engine_version = "unknown"
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    print("=== Reproduction fingerprint ===")
    print(f"  engine:  options_portfolio_backtester {engine_version} @ {commit}")
    print(f"  options: sha256:{hash_data_file(OPTIONS_PATH)[:16]}…  ({OPTIONS_PATH.name})")
    print(f"  stocks:  sha256:{hash_data_file(STOCKS_PATH)[:16]}…  ({STOCKS_PATH.name})")


def main():
    if not OPTIONS_PATH.exists() or not STOCKS_PATH.exists():
        raise SystemExit(
            f"Need processed SPY data at {DATA}. Run:\n"
            f"    python scripts/fetch_data.py all --symbols SPY"
        )

    _print_fingerprint()

    opts_data = HistoricalOptionsData(str(OPTIONS_PATH))
    stocks_data = TiingoData(str(STOCKS_PATH))
    schema = opts_data.schema
    spy = _run_spy(opts_data, stocks_data, schema)

    # ---------- Table: SPY baseline ----------
    print("\n=== SPY baseline (2008-2024) ===")
    print(f"  CAGR:       {_ann(spy):+6.2f}%")
    print(f"  Max DD:     {_maxdd(spy):+6.1f}%")
    print(f"  Sharpe:     {_sharpe(spy):+6.3f}")
    print(f"  Final:      ${spy.iloc[-1]:>13,.0f}")

    # ---------- Table: Spitznagel framing budget sweep ----------
    print("\n=== Spitznagel framing: budget sweep (40-45% OTM, DTE 90-180/30, bi-monthly) ===")
    _row("config", ["CAGR%", "Excess", "MaxDD%", "Sharpe", "Final"])
    spy_ann = _ann(spy)
    runs = {}
    for budget in [0.005, 0.010, 0.020, 0.033, 0.050, 0.070, 0.100]:
        b = _run(opts_data, stocks_data, schema, budget=budget)
        runs[budget] = b
        _row(
            f"100% SPY + {budget*100:.1f}%/yr deep OTM",
            [
                f"{_ann(b):+.2f}",
                f"{_ann(b) - spy_ann:+.2f}",
                f"{_maxdd(b):+.1f}",
                f"{_sharpe(b):+.3f}",
                f"${b.iloc[-1]:>12,.0f}",
            ],
        )

    # ---------- Table: Convexity breakdown ----------
    print("\n=== Convexity breakdown (full risk-adjusted picture) ===")
    _row("config", ["Premium%", "CAGR%", "Excess", "Ret/1%", "MaxDD%", "Sharpe"])
    _row("100% SPY (baseline)",
         ["0.00", f"{spy_ann:+.2f}", "+0.00", "—",
          f"{_maxdd(spy):+.1f}", f"{_sharpe(spy):+.3f}"])
    for budget in [0.005, 0.010, 0.020, 0.033, 0.050, 0.070, 0.100]:
        b = runs[budget]
        excess = _ann(b) - spy_ann
        _row(
            f"+ {budget*100:.1f}%/yr deep OTM",
            [
                f"{budget*100:.2f}",
                f"{_ann(b):+.2f}",
                f"{excess:+.2f}",
                f"{excess/(budget*100):.2f}",
                f"{_maxdd(b):+.1f}",
                f"{_sharpe(b):+.3f}",
            ],
        )

    # ---------- Table: Beyond Sharpe ----------
    print("\n=== Beyond Sharpe: downside-focused metrics ===")
    _row("config", ["Sharpe", "Sortino", "Calmar", "MaxDD%"])
    _row("100% SPY",
         [f"{_sharpe(spy):+.3f}", f"{_sortino(spy):+.3f}",
          f"{_calmar(spy):+.3f}", f"{_maxdd(spy):+.1f}"])
    for budget in [0.005, 0.010, 0.020, 0.033, 0.050, 0.070]:
        b = runs[budget]
        _row(
            f"+ {budget*100:.1f}%/yr deep OTM",
            [
                f"{_sharpe(b):+.3f}",
                f"{_sortino(b):+.3f}",
                f"{_calmar(b):+.3f}",
                f"{_maxdd(b):+.1f}",
            ],
        )

    # ---------- Table: Cross-validation halves ----------
    print("\n=== Cross-validation: halves (article default 3.3%/yr) ===")
    halves = {
        "H1 2008-2016": ("2008-01-02", "2016-12-31"),
        "H2 2017-2024": ("2017-01-01", "2024-12-31"),
    }
    b033 = runs[0.033]
    _row("period", ["SPY%", "Strat%", "SPY DD", "Strat DD", "Excess"])
    for label, (lo, hi) in halves.items():
        ss, bs = spy.loc[lo:hi], b033.loc[lo:hi]
        _row(label, [
            f"{_ann(ss):+.2f}", f"{_ann(bs):+.2f}",
            f"{_maxdd(ss):+.1f}", f"{_maxdd(bs):+.1f}",
            f"{_ann(bs) - _ann(ss):+.2f}",
        ])

    # ---------- Table: Cross-validation quarters ----------
    print("\n=== Cross-validation: quarters (article default 3.3%/yr) ===")
    quarters = {
        "Q1 2008-2011 (GFC)": ("2008-01-02", "2011-12-31"),
        "Q2 2012-2015 (calm)": ("2012-01-01", "2015-12-31"),
        "Q3 2016-2019 (calm + Q4'18)": ("2016-01-01", "2019-12-31"),
        "Q4 2020-2024 (COVID + bear)": ("2020-01-01", "2024-12-31"),
    }
    _row("quarter", ["SPY%", "Strat%", "Excess"])
    for label, (lo, hi) in quarters.items():
        ss, bs = spy.loc[lo:hi], b033.loc[lo:hi]
        _row(label, [
            f"{_ann(ss):+.2f}", f"{_ann(bs):+.2f}",
            f"{_ann(bs) - _ann(ss):+.2f}",
        ])

    # ---------- Table: Rolling 5-year windows ----------
    print("\n=== Cross-validation: rolling 5-year windows ===")
    _row("window", ["SPY%", "Strat%", "Excess"])
    for start_year in range(2008, 2021):
        lo = f"{start_year}-01-02"
        hi = f"{start_year + 4}-12-31"
        ss, bs = spy.loc[lo:hi], b033.loc[lo:hi]
        if len(ss) < 100:
            continue
        _row(
            f"{start_year}-{start_year+4}",
            [
                f"{_ann(ss):+.2f}",
                f"{_ann(bs):+.2f}",
                f"{_ann(bs) - _ann(ss):+.2f}",
            ],
        )

    # ---------- Table: Rebalance frequency ----------
    print("\n=== Rebalance frequency at the article default (3.3%/yr budget) ===")
    _row("frequency", ["CAGR%", "Excess", "MaxDD%", "Final"])
    for freq, unit, label in [
        (1, "W", "weekly"), (2, "W", "biweekly"),
        (1, "BMS", "monthly"), (2, "BMS", "bi-monthly"),
        (3, "BMS", "quarterly"),
    ]:
        b = _run(opts_data, stocks_data, schema, rebal_freq=freq, rebal_unit=unit)
        _row(label, [
            f"{_ann(b):+.2f}", f"{_ann(b) - spy_ann:+.2f}",
            f"{_maxdd(b):+.1f}", f"${b.iloc[-1]:>12,.0f}",
        ])

    # ---------- Table: Profit targets ----------
    print("\n=== Profit targets at the article default (3.3%/yr budget, bi-monthly) ===")
    _row("PT", ["CAGR%", "Excess", "MaxDD%"])
    _row("none (roll at DTE 30)",
         [f"{_ann(b033):+.2f}", f"{_ann(b033) - spy_ann:+.2f}", f"{_maxdd(b033):+.1f}"])
    for pt in [20.0, 10.0, 5.0, 3.0]:
        b = _run(opts_data, stocks_data, schema, profit_pct=pt)
        _row(f"{pt:.0f}x premium",
             [f"{_ann(b):+.2f}", f"{_ann(b) - spy_ann:+.2f}", f"{_maxdd(b):+.1f}"])

    print("\nAll tables produced. Numbers should match the article within "
          "the tolerances in tests/oracles/test_article_reproduction.py "
          "(0.5pp annual, 1.0pp max DD, 0.05 Sharpe).")


if __name__ == "__main__":
    main()
