"""Tests for structured TradeLog."""

import pandas as pd
import pytest

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


def test_from_legacy_accepts_rust_string_orders():
    """The Rust engine emits orders as strings ('BTO'), not Order enums."""
    cols = pd.MultiIndex.from_tuples(
        [("leg_1", f) for f in
         ("contract", "underlying", "type", "strike", "cost", "order")]
        + [("totals", f) for f in ("cost", "qty", "date")])
    # Engine sign convention: cost > 0 is paid (BTO), < 0 received (STC).
    rows = [
        ["SPY240119P00300000", "SPY", "put", 300.0, 500.0, "BTO",
         500.0, 1, pd.Timestamp("2023-06-01")],
        ["SPY240119P00300000", "SPY", "put", 300.0, -250.0, "STC",
         -250.0, 1, pd.Timestamp("2023-09-01")],
    ]
    legacy = pd.DataFrame(rows, columns=cols)
    tl = TradeLog.from_legacy_trade_log(legacy)
    assert len(tl) == 1
    trade = tl.trades[0]
    assert trade.entry_price == 5.0
    assert trade.exit_price == 2.5
    assert trade.net_pnl == -250.0


def _log(rows, legs=("leg_1",)):
    """Engine-shaped trade log: per leg (contract, cost, order), totals."""
    cols = [(leg, f) for leg in legs for f in ("contract", "underlying", "type", "strike", "cost", "order")]
    cols += [("totals", f) for f in ("cost", "qty", "date")]
    return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(cols))


def _leg(contract, cost, order, typ="put", strike=300.0):
    return [contract, "SPY", typ, strike, cost, order]


class TestRoundTripPnL:
    def test_short_put_winner_is_a_gain(self):
        # Sold at 5.00 (credit 500), bought back at 1.00 (debit 100), 10 lots.
        log = _log([
            _leg("P300", -500.0, "STO") + [-500.0, 10, pd.Timestamp("2024-01-02")],
            _leg("P300", 100.0, "BTC") + [100.0, 10, pd.Timestamp("2024-02-01")],
        ])
        (t,) = TradeLog.from_legacy_trade_log(log).trades
        assert t.gross_pnl == 4_000.0
        assert t.is_short

    def test_spread_is_one_round_trip_with_net_pnl(self):
        # Put spread: buy P300 at 5.00, sell P250 at 2.00 (net debit 300/unit);
        # close P300 at 9.00, P250 at 4.00 (net credit 500/unit); 2 units.
        legs = ("leg_1", "leg_2")
        log = _log([
            _leg("P300", 500.0, "BTO") + _leg("P250", -200.0, "STO", strike=250.0)
            + [300.0, 2, pd.Timestamp("2024-01-02")],
            _leg("P300", -900.0, "STC") + _leg("P250", 400.0, "BTC", strike=250.0)
            + [-500.0, 2, pd.Timestamp("2024-02-01")],
        ], legs)
        (t,) = TradeLog.from_legacy_trade_log(log).trades
        assert t.contract == "P300/P250"
        assert t.gross_pnl == 400.0  # (500 - 300) * 2

    def test_reentries_pair_fifo(self):
        log = _log([
            _leg("P300", 100.0, "BTO") + [100.0, 1, pd.Timestamp("2024-01-02")],
            _leg("P300", -300.0, "STC") + [-300.0, 1, pd.Timestamp("2024-01-10")],
            _leg("P300", 200.0, "BTO") + [200.0, 1, pd.Timestamp("2024-02-01")],
            _leg("P300", -100.0, "STC") + [-100.0, 1, pd.Timestamp("2024-02-10")],
        ])
        assert [t.gross_pnl for t in TradeLog.from_legacy_trade_log(log).trades] == [200.0, -100.0]

    def test_commissions_from_cost_model(self):
        from options_portfolio_backtester.execution.cost_model import PerContractCommission
        log = _log([
            _leg("P300", 100.0, "BTO") + [100.0, 10, pd.Timestamp("2024-01-02")],
            _leg("P300", -300.0, "STC") + [-300.0, 10, pd.Timestamp("2024-01-10")],
        ])
        (t,) = TradeLog.from_legacy_trade_log(log, cost_model=PerContractCommission(rate=0.65)).trades
        assert (t.entry_commission, t.exit_commission) == (6.5, 6.5)
        assert t.net_pnl == 2_000.0 - 13.0


class TestSummary:
    def _log_and_balance(self):
        log = _log([
            _leg("P300", 100.0, "BTO") + [100.0, 10, pd.Timestamp("2024-01-02")],
            _leg("P300", -300.0, "STC") + [-300.0, 10, pd.Timestamp("2024-01-10")],
            _leg("P290", 200.0, "BTO") + [200.0, 10, pd.Timestamp("2024-02-01")],
            _leg("P290", -100.0, "STC") + [-100.0, 10, pd.Timestamp("2024-02-10")],
        ])
        balance = pd.DataFrame({"total capital": [100_000.0, 100_500.0, 100_750.0]},
                               index=pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-29"]))
        return log, balance

    def test_counts_round_trips_and_profit_factor(self):
        from options_portfolio_backtester.analytics.summary import summary
        log, balance = self._log_and_balance()
        before = log.copy()
        table = summary(log, balance).data["Strategy"]
        assert table["Total trades"] == 2
        assert (table["Number of wins"], table["Number of losses"]) == (1, 1)
        assert table["Profit factor"] == 2.0          # 2,000 gained / 1,000 lost
        assert table["Largest loss"] == 1_000.0
        assert table["Total P&L %"] == pytest.approx(0.75)
        pd.testing.assert_frame_equal(log, before)    # input untouched
