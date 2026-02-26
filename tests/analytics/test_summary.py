"""Tests for analytics/summary.py — the legacy summary statistics function."""

import numpy as np
import pandas as pd

from options_portfolio_backtester.core.types import Order
from options_portfolio_backtester.analytics.summary import summary


def _make_trade_log_and_balance():
    """Build a minimal MultiIndex trade log and balance for testing summary()."""
    leg = "leg_1"
    entries = pd.DataFrame({
        (leg, "contract"): ["SPY_C_001", "SPY_C_002"],
        (leg, "underlying"): ["SPY", "SPY"],
        (leg, "expiration"): pd.to_datetime(["2020-03-20", "2020-03-20"]),
        (leg, "type"): ["call", "call"],
        (leg, "strike"): [320.0, 325.0],
        (leg, "cost"): [-500.0, -400.0],
        (leg, "order"): [Order.BTO, Order.BTO],
        ("totals", "cost"): [-500.0, -400.0],
        ("totals", "qty"): [2, 3],
        ("totals", "date"): pd.to_datetime(["2020-01-15", "2020-01-20"]),
    })
    exits = pd.DataFrame({
        (leg, "contract"): ["SPY_C_001", "SPY_C_002"],
        (leg, "underlying"): ["SPY", "SPY"],
        (leg, "expiration"): pd.to_datetime(["2020-03-20", "2020-03-20"]),
        (leg, "type"): ["call", "call"],
        (leg, "strike"): [320.0, 325.0],
        (leg, "cost"): [600.0, 350.0],
        (leg, "order"): [Order.STC, Order.STC],
        ("totals", "cost"): [600.0, 350.0],
        ("totals", "qty"): [2, 3],
        ("totals", "date"): pd.to_datetime(["2020-02-15", "2020-02-20"]),
    })
    entries.columns = pd.MultiIndex.from_tuples(entries.columns)
    exits.columns = pd.MultiIndex.from_tuples(exits.columns)
    trade_log = pd.concat([entries, exits], ignore_index=True)

    dates = pd.date_range("2020-01-10", periods=30, freq="B")
    capital = np.linspace(1_000_000, 1_050_000, 30)
    balance = pd.DataFrame({"total capital": capital}, index=dates)
    balance["% change"] = balance["total capital"].pct_change()

    return trade_log, balance


class TestSummary:
    def test_returns_styler(self):
        trade_log, balance = _make_trade_log_and_balance()
        result = summary(trade_log, balance)
        assert isinstance(result, pd.io.formats.style.Styler)

    def test_summary_has_expected_rows(self):
        trade_log, balance = _make_trade_log_and_balance()
        result = summary(trade_log, balance)
        df = result.data
        assert "Total trades" in df.index
        assert "Win %" in df.index
        assert "Average P&L %" in df.index
        assert "Total P&L %" in df.index

    def test_total_trades_count(self):
        trade_log, balance = _make_trade_log_and_balance()
        result = summary(trade_log, balance)
        df = result.data
        total_trades = df.loc["Total trades", "Strategy"]
        assert total_trades == 2

    def test_win_metrics(self):
        trade_log, balance = _make_trade_log_and_balance()
        result = summary(trade_log, balance)
        df = result.data
        wins = df.loc["Number of wins", "Strategy"]
        losses = df.loc["Number of losses", "Strategy"]
        assert wins + losses == df.loc["Total trades", "Strategy"]

    def test_summary_with_missing_exit(self):
        """When an exit is missing for a contract, IndexError branch is hit."""
        leg = "leg_1"
        # Entry for contract A, but no matching exit
        entries = pd.DataFrame({
            (leg, "contract"): ["SPY_C_ORPHAN"],
            (leg, "underlying"): ["SPY"],
            (leg, "expiration"): pd.to_datetime(["2020-03-20"]),
            (leg, "type"): ["call"],
            (leg, "strike"): [320.0],
            (leg, "cost"): [-500.0],
            (leg, "order"): [Order.BTO],
            ("totals", "cost"): [-500.0],
            ("totals", "qty"): [2],
            ("totals", "date"): pd.to_datetime(["2020-01-15"]),
        })
        # Exit for a different contract
        exits = pd.DataFrame({
            (leg, "contract"): ["SPY_C_OTHER"],
            (leg, "underlying"): ["SPY"],
            (leg, "expiration"): pd.to_datetime(["2020-03-20"]),
            (leg, "type"): ["call"],
            (leg, "strike"): [325.0],
            (leg, "cost"): [600.0],
            (leg, "order"): [Order.STC],
            ("totals", "cost"): [600.0],
            ("totals", "qty"): [2],
            ("totals", "date"): pd.to_datetime(["2020-02-15"]),
        })
        entries.columns = pd.MultiIndex.from_tuples(entries.columns)
        exits.columns = pd.MultiIndex.from_tuples(exits.columns)
        trade_log = pd.concat([entries, exits], ignore_index=True)

        dates = pd.date_range("2020-01-10", periods=10, freq="B")
        balance = pd.DataFrame({
            "total capital": np.linspace(1_000_000, 1_010_000, 10),
        }, index=dates)
        balance["% change"] = balance["total capital"].pct_change()

        result = summary(trade_log, balance)
        df = result.data
        # Should still produce output — the orphan contract is skipped
        assert "Total trades" in df.index
