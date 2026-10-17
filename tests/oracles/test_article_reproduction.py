"""Regression tests for the canonical reproductions of published articles.

These tests pin the headline numbers from articles that depend on this
backtester. If an upstream change in the engine moves a published table
by more than the tolerance, the test fails and either the article needs
re-verification or the engine change needs review. The intent is to
prevent the kind of silent drift that left the Spitznagel article
publishing pre-fix numbers for an unknown period after the
externally-funded budget path was repaired.

Article-reproduction tests should:
  1. Pin SPY baseline values (annual return, max DD) so changes to the
     data pipeline or compounding math are caught directly.
  2. Pin the published table's Spitznagel-framing numbers (annual,
     max DD, Sharpe) at the budgets the article calls out.
  3. Tolerate small numerical drift (~0.5pp on returns, ~0.05 on
     Sharpe) but flag larger moves.

The companion script outside this repo,
``unbalancedparentheses/finance_research:scripts/verify_blog_numbers.py``,
exercises the same reproduction in human-readable form; this file is
the CI-friendly version.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
# Parquet is the canonical processed format: ~30x faster to load than the
# CSV (~0.4s vs ~15s for the 17-year chain) and kept date-aligned by
# fetch_data.align_dates() since PR #107.
OPTIONS_PARQUET = DATA_ROOT / "options.parquet"
STOCKS_CSV = DATA_ROOT / "stocks.csv"

requires_data = pytest.mark.skipif(
    not (OPTIONS_PARQUET.exists() and STOCKS_CSV.exists()),
    reason=(
        "Article-reproduction tests need processed SPY data. Run "
        "'python scripts/fetch_data.py all --symbols SPY' before invoking them."
    ),
)


# --- Spitznagel article -----------------------------------------------------
#
# Article: https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/
# Configuration: strike-based 40-45% OTM puts (strike between 55% and 60% of
# spot), DTE 90-180 at entry, exit when DTE drops to 30 (≈ 60-150 day hold),
# bi-monthly rebalance, daily exit checks, monetize-and-reinvest on, no profit
# target, no IV filter, externally-funded annual put budget on top of 100% SPY.
# Data window: 2008-01-02 to 2024-12-31.

SPITZNAGEL_SPY_BASELINE = {
    "annual": 10.68,
    "max_dd": -51.9,
}

# Spitznagel framing (100% stocks + annual put budget on top).
# Tolerance: 0.5pp annual return, 1.0pp max DD, 0.05 Sharpe.
#
# ERRATA vs the published table. The article's numbers were produced while
# daily-exit runs back-filled each rebalance period's balance rows with the
# end-of-period holdings (a look-ahead in the NAV path), and before premium
# was debited at entry. Annual returns were essentially right (all within
# 0.13pp); the path statistics were not. Published -> corrected:
#   0.5%: max DD -41.3 -> -41.2, Sharpe 0.613 -> 0.661
#   1.0%: max DD -31.2 -> -31.1, Sharpe 0.636 -> 0.749
#   2.0%: max DD -30.8 -> -26.0, Sharpe 0.597 -> 0.797
#   3.3%: max DD -30.2 -> -31.8, Sharpe 0.519 -> 0.740
# The values below are the corrected ones.
SPITZNAGEL_TABLE = {
    0.005: {"annual": 12.03, "max_dd": -41.2, "sharpe": 0.661},
    0.010: {"annual": 13.17, "max_dd": -31.1, "sharpe": 0.749},
    0.020: {"annual": 14.96, "max_dd": -26.0, "sharpe": 0.797},
    0.033: {"annual": 16.61, "max_dd": -31.8, "sharpe": 0.740},
}

INITIAL_CAPITAL = 1_000_000
RETURN_TOLERANCE_PP = 0.5
DRAWDOWN_TOLERANCE_PP = 1.0
SHARPE_TOLERANCE = 0.05


# The pinned tables above are only valid for this exact window. Clamp the
# loaded data to it so fetching a longer range (e.g. `fetch_data.py` with a
# later --end) cannot masquerade as an engine regression — an extra year of
# data once shifted the most-leveraged row past tolerance and looked exactly
# like a backtest bug.
ARTICLE_WINDOW_END = pd.Timestamp("2024-12-31")


@pytest.fixture(scope="module")
def options_data():
    d = HistoricalOptionsData(str(OPTIONS_PARQUET))
    date_col = d.schema["date"]
    d._data = d._data[d._data[date_col] <= ARTICLE_WINDOW_END]
    d.end_date = d._data[date_col].max()
    return d


@pytest.fixture(scope="module")
def stocks_data():
    d = TiingoData(str(STOCKS_CSV))
    date_col = d.schema["date"]
    d._data = d._data[d._data[date_col] <= ARTICLE_WINDOW_END]
    d.end_date = d._data[date_col].max()
    return d


def _make_deep_otm_put_strategy(schema):
    """Strike-based 40-45% OTM puts at DTE 90-180 entry, exit at DTE 30.

    The article filters by strike-to-spot ratio (not delta) so the depth is
    constant across volatility regimes — delta-based filtering produces puts
    that are 7% OTM in high-vol periods and 30% OTM in calm periods even at
    the same delta.
    """
    leg = StrategyLeg("leg_1", schema, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= 90)
        & (schema.dte <= 180)
        & (schema.strike <= schema.underlying_last * 0.60)   # strike <= 60% of spot = >= 40% OTM
        & (schema.strike >= schema.underlying_last * 0.55)   # strike >= 55% of spot = <= 45% OTM
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = schema.dte <= 30
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def _run_spitznagel(options_data, stocks_data, schema, budget_pct):
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget_annual_pct = budget_pct
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks_data
    bt.options_data = options_data
    bt.options_strategy = _make_deep_otm_put_strategy(schema)
    # Bi-monthly rebalance — better than monthly across all budgets in our
    # cross-validation; less premium decay per year for the same total budget.
    bt.run(rebalance_freq=2, rebalance_unit="BMS")
    return bt.balance


def _compute_stats(balance):
    bal = balance["total capital"]
    years = (bal.index[-1] - bal.index[0]).days / 365.25
    total = bal.iloc[-1] / bal.iloc[0] - 1
    annual = ((1 + total) ** (1 / years) - 1) * 100
    daily = bal.pct_change().dropna()
    vol = daily.std() * math.sqrt(252) * 100
    sharpe = annual / vol if vol > 0 else 0.0
    cummax = bal.cummax()
    max_dd = ((bal - cummax) / cummax).min() * 100
    return annual, max_dd, sharpe


def _spy_series(stocks_data):
    """SPY adjusted-close series indexed by date (buy-and-hold baseline)."""
    df = stocks_data._data.sort_values("date")
    df = df[df["symbol"] == "SPY"]
    return df.set_index("date")["adjClose"]


@requires_data
def test_spitznagel_spy_baseline(stocks_data):
    """SPY buy-and-hold over the article's 2008-2024 window."""
    df = stocks_data._data.sort_values("date")
    df = df[df["symbol"] == "SPY"]
    prices = df["adjClose"].values
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    annual = ((prices[-1] / prices[0]) ** (1 / years) - 1) * 100
    series = df.set_index("date")["adjClose"]
    cummax = series.cummax()
    max_dd = ((series - cummax) / cummax).min() * 100

    assert abs(annual - SPITZNAGEL_SPY_BASELINE["annual"]) < RETURN_TOLERANCE_PP, (
        f"SPY baseline annual return drifted: "
        f"expected {SPITZNAGEL_SPY_BASELINE['annual']:.2f}%, got {annual:.2f}%"
    )
    assert abs(max_dd - SPITZNAGEL_SPY_BASELINE["max_dd"]) < DRAWDOWN_TOLERANCE_PP, (
        f"SPY baseline max DD drifted: "
        f"expected {SPITZNAGEL_SPY_BASELINE['max_dd']:.1f}%, got {max_dd:.1f}%"
    )


@requires_data
@pytest.mark.parametrize("budget", sorted(SPITZNAGEL_TABLE.keys()))
def test_spitznagel_framing(options_data, stocks_data, budget):
    """One row of the article's Spitznagel-framing table."""
    schema = options_data.schema
    balance = _run_spitznagel(options_data, stocks_data, schema, budget)
    annual, max_dd, sharpe = _compute_stats(balance)
    expected = SPITZNAGEL_TABLE[budget]

    assert abs(annual - expected["annual"]) < RETURN_TOLERANCE_PP, (
        f"budget {budget*100:.2f}%: annual return drifted from article: "
        f"expected {expected['annual']:.2f}%, got {annual:.2f}%"
    )
    assert abs(max_dd - expected["max_dd"]) < DRAWDOWN_TOLERANCE_PP, (
        f"budget {budget*100:.2f}%: max DD drifted from article: "
        f"expected {expected['max_dd']:.1f}%, got {max_dd:.1f}%"
    )
    assert abs(sharpe - expected["sharpe"]) < SHARPE_TOLERANCE, (
        f"budget {budget*100:.2f}%: Sharpe drifted from article: "
        f"expected {expected['sharpe']:.3f}, got {sharpe:.3f}"
    )


@requires_data
def test_spitznagel_monotone_excess_and_dd_improvement(options_data, stocks_data):
    """At the strike-based 40-45% OTM, DTE 90-180/exit 30, bi-monthly configuration
    in the article, every budget from 0.5% to 3.3%/yr beats SPY on annual
    return, and the excess grows monotonically with budget (linearly within
    rounding). Max drawdown improves through the working range — every tested
    budget has a smaller drawdown than SPY's. If a future engine change reverts
    either property, the article's core claim breaks and CI should fail.
    """
    schema = options_data.schema
    spy_series = _spy_series(stocks_data).to_frame("total capital")
    spy_annual, spy_max_dd, _ = _compute_stats(spy_series)

    excesses, drawdowns = {}, {}
    for budget in (0.005, 0.010, 0.020, 0.033):
        balance = _run_spitznagel(options_data, stocks_data, schema, budget)
        annual, max_dd, _ = _compute_stats(balance)
        excesses[budget] = annual - spy_annual
        drawdowns[budget] = max_dd

    for budget, ex in excesses.items():
        assert ex > 0.5, (
            f"budget {budget*100:.1f}%: excess over SPY ({ex:+.2f}pp) "
            f"must exceed +0.5pp — article claims this config beats SPY"
        )

    budgets = [0.005, 0.010, 0.020, 0.033]
    for lo, hi in zip(budgets, budgets[1:]):
        assert excesses[hi] > excesses[lo] - 0.5, (
            f"excess should grow with budget: {lo*100:.1f}% gave "
            f"{excesses[lo]:+.2f}pp but {hi*100:.1f}% gave {excesses[hi]:+.2f}pp"
        )

    for budget, dd in drawdowns.items():
        assert dd > spy_max_dd + 5.0, (
            f"budget {budget*100:.1f}%: max DD ({dd:.1f}%) must improve "
            f"SPY's max DD ({spy_max_dd:.1f}%) by at least 5pp"
        )


# --- Fast smoke variant -----------------------------------------------------
# The tests above need the full 17-year SPY chain and take ~3 minutes to run.
# This fast variant runs only the 0.5%/yr Spitznagel budget against the full
# sample and asserts the corrected qualitative shape — overlay TRACKS SPY on
# return and is no worse on max drawdown — without pinning specific numbers.
# Suitable as a pre-commit / fast-CI smoke that catches "Spitznagel framing
# fundamentally broken" without paying for the full parametrized run.

@requires_data
@pytest.mark.smoke
def test_spitznagel_smoke_qualitative(options_data, stocks_data):
    """Single-budget qualitative check: at 0.5%/yr Universa-described scale,
    the strategy beats SPY by at least 0.5pp annual and improves max drawdown
    by at least 5pp. Faster than the full parametrized table; catches
    engine-side regressions that flip the sign of the trade.
    """
    schema = options_data.schema

    series = _spy_series(stocks_data)
    spy_annual, spy_max_dd, _ = _compute_stats(series.to_frame("total capital"))

    balance = _run_spitznagel(options_data, stocks_data, schema, 0.005)
    annual, max_dd, _ = _compute_stats(balance)

    assert annual > spy_annual + 0.5, (
        f"0.5%/yr Spitznagel annual ({annual:.2f}%) should beat "
        f"SPY annual ({spy_annual:.2f}%) by at least 0.5pp"
    )
    assert max_dd > spy_max_dd + 5.0, (
        f"0.5%/yr Spitznagel max DD ({max_dd:.1f}%) should be at least "
        f"5pp less severe than SPY max DD ({spy_max_dd:.1f}%)"
    )
