from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.analytics.options_charts import (
    DEFAULT_CRASH_WINDOWS,
    crash_window_chart,
    exposure_chart,
    holding_period_chart,
    normalize_trade_log,
    options_pnl_decomposition_chart,
    premium_spend_chart,
    trade_pnl_chart,
    trade_return_histogram,
    yearly_pnl_chart,
)
from options_portfolio_backtester.analytics.trade_log import Trade, TradeLog
from options_portfolio_backtester.core.types import Order


def _balance(start: str = "2020-01-01", periods: int = 200) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="B")
    rng = np.random.default_rng(7)
    total = 100_000 * np.cumprod(1 + rng.normal(0.0004, 0.01, periods))
    options = total * 0.02
    cash = total * 0.03
    bal = pd.DataFrame({
        "total capital": total,
        "options capital": options,
        "cash": cash,
        "stocks capital": total - options - cash,
    }, index=idx)
    bal["% change"] = bal["total capital"].pct_change()
    return bal


def _trade(entry="2020-02-03", exit_="2020-04-01", entry_price=1.5,
           exit_price=0.5, qty=10) -> Trade:
    return Trade(
        contract="SPY200417P00200000", underlying="SPY", option_type="put",
        strike=200.0, entry_date=pd.Timestamp(entry), exit_date=pd.Timestamp(exit_),
        entry_price=entry_price, exit_price=exit_price, quantity=qty,
        shares_per_contract=100, entry_order=Order.BTO, exit_order=Order.STC,
    )


def _trade_log() -> TradeLog:
    tl = TradeLog()
    tl.add_trade(_trade())
    tl.add_trade(_trade(entry="2020-03-02", exit_="2020-03-20",
                        entry_price=1.0, exit_price=25.0))
    return tl


# ---------------------------------------------------------------------------
# normalize_trade_log
# ---------------------------------------------------------------------------

def test_normalize_none_and_empty():
    assert normalize_trade_log(None).empty
    assert normalize_trade_log(pd.DataFrame()).empty


def test_normalize_tradelog_object():
    df = normalize_trade_log(_trade_log())
    assert len(df) == 2
    assert "net_pnl" in df.columns


def test_normalize_flat_frame_passthrough():
    flat = _trade_log().to_dataframe()
    out = normalize_trade_log(flat)
    pd.testing.assert_frame_equal(out, flat)


def test_normalize_rejects_unknown_type():
    with pytest.raises(TypeError):
        normalize_trade_log(42)


# ---------------------------------------------------------------------------
# chart builders return non-empty specs on valid input
# ---------------------------------------------------------------------------

def test_options_pnl_decomposition_known_trade():
    """BTO 10×$1.50 then STC at $0.50 on $100k initial capital:
    paid 1.5%, received 0.5%, net −1.0%."""
    tl = TradeLog()
    tl.add_trade(_trade(entry="2020-02-03", exit_="2020-04-01",
                        entry_price=1.5, exit_price=0.5, qty=10))
    chart = options_pnl_decomposition_chart(tl.to_dataframe(), _flat_balance())
    data = chart.data.set_index(["date", "series"])["value"]
    exit_day = pd.Timestamp("2020-04-01")
    assert abs(data.loc[(exit_day, "Premium paid (drag)")] - (-0.015)) < 1e-9
    assert abs(data.loc[(exit_day, "Payoffs received")] - 0.005) < 1e-9
    assert abs(data.loc[(exit_day, "Net options P&L")] - (-0.010)) < 1e-9


def test_options_pnl_decomposition_empty_inputs():
    chart = options_pnl_decomposition_chart(pd.DataFrame(), _balance())
    assert len(chart.data) == 0


def test_trade_return_histogram_mass_at_loss():
    """A −67% trade and a +24x trade land in the right bins."""
    chart = trade_return_histogram(_trade_log().to_dataframe())
    data = chart.data
    assert data["count"].sum() == 2
    lows = data[data["bin_start"] < 0]
    highs = data[data["bin_start"] > 10]
    assert lows["count"].sum() == 1
    assert highs["count"].sum() == 1


def test_premium_spend_chart_with_budget_layers_rule():
    chart = premium_spend_chart(_trade_log().to_dataframe(), _balance(),
                                budget_annual_pct=0.033)
    # budget rule makes it a LayerChart
    assert hasattr(chart, "layer")


def test_premium_spend_chart_empty_inputs():
    chart = premium_spend_chart(pd.DataFrame(), _balance())
    assert len(chart.data) == 0


def test_crash_window_chart_covid_overlap():
    chart = crash_window_chart(_balance("2020-01-01", 200))
    assert chart is not None  # COVID window overlaps the sample


def test_crash_window_chart_no_overlap_returns_none():
    assert crash_window_chart(_balance("2015-01-01", 100)) is None


def test_crash_window_chart_benchmark_series_present():
    bal = _balance("2020-01-01", 200)
    chart = crash_window_chart(bal, benchmark_balance=bal * 0.9)
    data = chart.data if chart.data is not None else None
    if data is not None:
        assert set(data["series"]) == {"strategy", "benchmark"}


def test_trade_pnl_chart_outcomes():
    chart = trade_pnl_chart(_trade_log().to_dataframe())
    assert set(chart.data["outcome"]) == {"win", "loss"}


def test_holding_period_chart_has_days():
    chart = holding_period_chart(_trade_log().to_dataframe())
    assert "holding_days" in chart.data.columns
    assert (chart.data["holding_days"] > 0).all()


def test_yearly_pnl_chart_groups_by_exit_year():
    chart = yearly_pnl_chart(_trade_log().to_dataframe())
    assert set(chart.data.columns) >= {"year", "net_pnl", "outcome"}


def test_exposure_chart_values_are_ratios():
    chart = exposure_chart(_balance())
    assert (chart.data["exposure"].dropna() < 1).all()


def test_default_crash_windows_cover_three_events():
    assert len(DEFAULT_CRASH_WINDOWS) == 3


# ---------------------------------------------------------------------------
# semantic checks: values, signs, alignment (not just construction)
# ---------------------------------------------------------------------------

def _flat_balance(start="2020-01-01", periods=300, capital=100_000.0):
    idx = pd.date_range(start, periods=periods, freq="B")
    bal = pd.DataFrame({"total capital": [capital] * periods}, index=idx)
    return bal


def test_premium_spend_value_known_trade():
    """One BTO trade of 10 contracts at $1.50 on $100k = 1.5% rolling spend."""
    tl = TradeLog()
    tl.add_trade(_trade(entry="2020-02-03", entry_price=1.5, qty=10))
    chart = premium_spend_chart(tl.to_dataframe(), _flat_balance())
    spend = chart.data.set_index("date")["spend"]
    assert abs(spend.loc["2020-03-02"] - 0.015) < 1e-9
    assert spend.loc["2020-01-15"] == 0.0  # before the entry


def test_premium_spend_sto_is_credit():
    """Short premium (STO entry) must count as a credit, not absolute spend."""
    tl = TradeLog()
    short = _trade(entry="2020-02-03", entry_price=2.0, qty=5)
    short.entry_order = Order.STO
    tl.add_trade(short)
    chart = premium_spend_chart(tl.to_dataframe(), _flat_balance())
    assert chart.data["spend"].min() < 0
    assert chart.data["spend"].max() <= 0


def test_premium_spend_intraday_timestamp_not_dropped():
    """Entry timestamps with a time component must still land on the day."""
    tl = TradeLog()
    tl.add_trade(_trade(entry="2020-02-03 14:30:00", entry_price=1.5, qty=10))
    chart = premium_spend_chart(tl.to_dataframe(), _flat_balance())
    assert abs(chart.data["spend"].max() - 0.015) < 1e-9


def test_premium_spend_respects_shares_per_contract_column():
    tl = TradeLog()
    mini = _trade(entry="2020-02-03", entry_price=1.5, qty=10)
    mini.shares_per_contract = 10  # mini contract
    tl.add_trade(mini)
    chart = premium_spend_chart(tl.to_dataframe(), _flat_balance())
    assert abs(chart.data["spend"].max() - 0.0015) < 1e-9


def test_crash_window_vconcat_panels_have_both_series_indexed_at_100():
    idx = pd.date_range("2008-01-01", periods=3900, freq="B")  # spans all 3 windows
    bal = pd.DataFrame({"total capital": np.linspace(100_000, 200_000, 3900)}, index=idx)
    bench = bal * 0.9
    chart = crash_window_chart(bal, benchmark_balance=bench)
    import altair as alt
    assert isinstance(chart, alt.VConcatChart)
    assert len(chart.vconcat) == 3
    for panel in chart.vconcat:
        data = panel.data
        assert set(data["series"]) == {"strategy", "benchmark"}
        for series in ("strategy", "benchmark"):
            first = data[data["series"] == series].iloc[0]
            assert abs(first["value"] - 100.0) < 1e-9
