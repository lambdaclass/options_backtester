"""CI pin for the article's *figures*, companion to test_article_reproduction.

The reproduction test pins the published numbers; this pins the inputs to
the published charts: the tearsheet panel inventory for the article-default
configuration, the structure each panel draws from, and serializability on
real 17-year data (the altair row-limit class of regression). An engine or
analytics change that would alter a published figure fails here the same
way numeric drift fails the reproduction test.

Runs one backtest at the article-default 3.3%/yr budget; skipped (like the
reproduction test) when the SPY data is not present.
"""

from __future__ import annotations

import pytest

from options_portfolio_backtester.analytics.options_charts import normalize_trade_log
from options_portfolio_backtester.analytics.tearsheet import build_tearsheet
from options_portfolio_backtester.core.types import Stock
from options_portfolio_backtester.engine.engine import BacktestEngine
from tests.oracles.test_article_reproduction import (  # noqa: F401  (fixtures)
    INITIAL_CAPITAL,
    OPTIONS_PARQUET,
    STOCKS_CSV,
    _make_deep_otm_put_strategy,
    options_data,
    requires_data,
    stocks_data,
)

ARTICLE_BUDGET = 0.033

EXPECTED_PANELS = [
    "Equity curve",
    "Underwater plot",
    "Rolling Sharpe",
    "Rolling volatility",
    "Return distribution",
    "Annual returns",
    "Monthly returns heatmap",
    "Capital allocation",
    "Options exposure",
    "Crash windows",
    "Options P&L decomposition",
    "Per-trade P&L",
    "Trade payoff distribution",
    "Holding periods",
    "Realized P&L by year",
    "Premium spend",
]


@pytest.fixture(scope="module")
def article_run(options_data, stocks_data):  # noqa: F811 (fixtures)
    """The article-default engine run, shared across this module's tests."""
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget_annual_pct = ARTICLE_BUDGET
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks_data
    bt.options_data = options_data
    bt.options_strategy = _make_deep_otm_put_strategy(options_data.schema)
    bt.run(rebalance_freq=2, rebalance_unit="BMS")
    return bt


@pytest.fixture(scope="module")
def article_report(article_run):
    import pandas as pd
    balance = article_run.balance.copy()
    balance.index = pd.to_datetime(balance.index)
    return build_tearsheet(
        balance,
        trade_log=article_run.trade_log,
        budget_annual_pct=ARTICLE_BUDGET,
    )


@requires_data
def test_panel_inventory_is_pinned(article_report):
    assert list(article_report.charts().keys()) == EXPECTED_PANELS


@requires_data
def test_all_panels_serialize_on_real_data(article_report):
    """Guards the altair 5000-row class of regression on the full sample."""
    for title, chart in article_report.charts().items():
        chart.to_json()


@requires_data
def test_crash_windows_cover_all_three_events(article_report):
    crash = article_report.charts()["Crash windows"]
    assert len(crash.vconcat) == 3


@requires_data
def test_trade_log_recovers_round_trips(article_run):
    """The string-order regression in from_legacy_trade_log silently
    produced zero trades from any Rust-engine run; pin a sane count.
    Bi-monthly rolls over 17 years ≈ 94 round trips (open positions at
    sample end are not round trips yet)."""
    trades = normalize_trade_log(article_run.trade_log)
    assert 50 < len(trades) < 150


@requires_data
def test_premium_spend_within_budget_envelope(article_report):
    """Realized rolling spend stays positive and never exceeds the budget by
    more than 0.5pp. (It systematically *under*-spends — a separate open
    question — so the lower bound is deliberately loose.)"""
    chart = article_report.charts()["Premium spend"]
    spend = chart.layer[0].data["spend"]
    assert spend.max() > 0.005
    assert spend.max() < ARTICLE_BUDGET + 0.005


@requires_data
def test_decomposition_net_matches_trade_pnl(article_run, article_report):
    """The decomposition's final net line must equal the summed per-trade
    net P&L — both derive from the same trades by different paths."""
    trades = normalize_trade_log(article_run.trade_log)
    initial = article_report.balance["total capital"].dropna().iloc[0]
    expected_net = trades["gross_pnl"].sum() / initial
    chart = article_report.charts()["Options P&L decomposition"]
    net = chart.data[chart.data["series"] == "Net options P&L"]["value"].iloc[-1]
    assert abs(net - expected_net) < 1e-6
