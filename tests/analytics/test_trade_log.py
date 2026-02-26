"""Tests for structured TradeLog."""

import numpy as np
import pandas as pd

from options_portfolio_backtester.analytics.trade_log import Trade, TradeLog
from options_portfolio_backtester.core.types import Order


def _make_trade(pnl_sign: float = 1.0) -> Trade:
    return Trade(
        contract="SPY_C_500",
        underlying="SPY",
        option_type="call",
        strike=500.0,
        entry_date=pd.Timestamp("2024-01-01"),
        exit_date=pd.Timestamp("2024-02-01"),
        entry_price=5.0,
        exit_price=5.0 + pnl_sign * 2.0,
        quantity=10,
        shares_per_contract=100,
        entry_order=Order.BTO,
        exit_order=Order.STC,
    )


class TestTrade:
    def test_gross_pnl(self):
        t = _make_trade(pnl_sign=1.0)
        # (7-5) * 10 * 100 = 2000
        assert t.gross_pnl == 2000.0

    def test_gross_pnl_loss(self):
        t = _make_trade(pnl_sign=-1.0)
        # (3-5) * 10 * 100 = -2000
        assert t.gross_pnl == -2000.0

    def test_net_pnl_with_commission(self):
        t = Trade(
            contract="X", underlying="SPY", option_type="call",
            strike=500.0, entry_date="2024-01-01", exit_date="2024-02-01",
            entry_price=5.0, exit_price=7.0,
            quantity=10, shares_per_contract=100,
            entry_order=Order.BTO, exit_order=Order.STC,
            entry_commission=6.50, exit_commission=6.50,
        )
        assert t.net_pnl == 2000.0 - 13.0

    def test_return_pct(self):
        t = _make_trade(pnl_sign=1.0)
        # gross=2000, entry_cost=5*10*100=5000, return=40%
        assert abs(t.return_pct - 0.40) < 1e-10


class TestTradeLog:
    def test_add_and_len(self):
        tl = TradeLog()
        tl.add_trade(_make_trade(1.0))
        tl.add_trade(_make_trade(-1.0))
        assert len(tl) == 2

    def test_winners_losers(self):
        tl = TradeLog()
        tl.add_trade(_make_trade(1.0))
        tl.add_trade(_make_trade(-1.0))
        tl.add_trade(_make_trade(1.0))
        assert len(tl.winners) == 2
        assert len(tl.losers) == 1

    def test_net_pnls(self):
        tl = TradeLog()
        tl.add_trade(_make_trade(1.0))
        tl.add_trade(_make_trade(-1.0))
        pnls = tl.net_pnls
        assert len(pnls) == 2
        assert pnls[0] == 2000.0
        assert pnls[1] == -2000.0

    def test_to_dataframe(self):
        tl = TradeLog()
        tl.add_trade(_make_trade(1.0))
        df = tl.to_dataframe()
        assert "net_pnl" in df.columns
        assert "return_pct" in df.columns
        assert len(df) == 1

    def test_empty_to_dataframe(self):
        tl = TradeLog()
        df = tl.to_dataframe()
        assert len(df) == 0

    def test_from_legacy_empty(self):
        tl = TradeLog.from_legacy_trade_log(pd.DataFrame())
        assert len(tl) == 0

    def test_from_legacy_trade_log(self):
        """Build a MultiIndex trade log and verify round-trip parsing."""
        leg = "leg_1"
        entries = pd.DataFrame({
            (leg, "contract"): ["SPY_C_001"],
            (leg, "underlying"): ["SPY"],
            (leg, "expiration"): pd.to_datetime(["2024-03-15"]),
            (leg, "type"): ["call"],
            (leg, "strike"): [450.0],
            (leg, "cost"): [-500.0],
            (leg, "order"): [Order.BTO],
            ("totals", "cost"): [-500.0],
            ("totals", "qty"): [5],
            ("totals", "date"): pd.to_datetime(["2024-01-15"]),
        })
        exits = pd.DataFrame({
            (leg, "contract"): ["SPY_C_001"],
            (leg, "underlying"): ["SPY"],
            (leg, "expiration"): pd.to_datetime(["2024-03-15"]),
            (leg, "type"): ["call"],
            (leg, "strike"): [450.0],
            (leg, "cost"): [600.0],
            (leg, "order"): [Order.STC],
            ("totals", "cost"): [600.0],
            ("totals", "qty"): [5],
            ("totals", "date"): pd.to_datetime(["2024-02-15"]),
        })
        entries.columns = pd.MultiIndex.from_tuples(entries.columns)
        exits.columns = pd.MultiIndex.from_tuples(exits.columns)
        trade_log_df = pd.concat([entries, exits], ignore_index=True)

        tl = TradeLog.from_legacy_trade_log(trade_log_df)
        assert len(tl) == 1
        assert tl.trades[0].contract == "SPY_C_001"
        assert tl.trades[0].quantity == 5

    def test_return_pct_zero_entry(self):
        t = Trade(
            contract="X", underlying="SPY", option_type="call",
            strike=500.0, entry_date="2024-01-01", exit_date="2024-02-01",
            entry_price=0.0, exit_price=1.0,
            quantity=10, shares_per_contract=100,
            entry_order=Order.BTO, exit_order=Order.STC,
        )
        assert t.return_pct == 0.0
