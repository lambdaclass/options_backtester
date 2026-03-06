"""Balance sheet and trade log invariants.

Tests run each backtest ONCE and verify structural invariants.
Covers small, generated, and production datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.bench._test_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    IVY_STOCKS_TUPLES,
    ivy_stocks,
    generated_stocks,
    prod_spy_stocks,
    load_generated_stocks,
    load_generated_options,
    load_prod_stocks,
    load_prod_options,
    buy_put_strategy,
    sell_put_strategy,
    strangle_strategy,
    run_backtest,
    assert_invariants,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# ── Small dataset invariants ───────────────────────────────────────────

class TestBalanceSheetInvariants:

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.eng = run_backtest()

    def test_total_capital_equals_parts(self):
        assert_invariants(self.eng)

    def test_capital_never_negative(self):
        tc = self.eng.balance["total capital"]
        assert (tc >= -1.0).all()

    def test_initial_capital_correct(self):
        first_tc = self.eng.balance["total capital"].iloc[0]
        assert abs(first_tc - DEFAULT_CAPITAL) < 1.0

    def test_balance_dates_monotonic(self):
        assert self.eng.balance.index.is_monotonic_increasing

    def test_balance_not_empty(self):
        assert len(self.eng.balance) > 1

    def test_cash_column_exists(self):
        assert "cash" in self.eng.balance.columns


class TestTradeLogInvariants:

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.eng = run_backtest()

    def test_trade_log_not_empty(self):
        assert not self.eng.trade_log.empty

    def test_entry_costs_nonzero(self):
        if self.eng.trade_log.empty:
            pytest.skip("No trades")
        costs = self.eng.trade_log["totals"]["cost"].values
        assert all(c != 0 for c in costs)

    def test_qty_positive_on_entry(self):
        if self.eng.trade_log.empty:
            pytest.skip("No trades")
        qtys = self.eng.trade_log["totals"]["qty"].values
        assert all(q > 0 for q in qtys)

    def test_trade_dates_within_data_range(self):
        if self.eng.trade_log.empty:
            pytest.skip("No trades")
        trade_dates = pd.to_datetime(self.eng.trade_log["totals"]["date"]).unique()
        data_start = pd.Timestamp(self.eng.options_data._data["quotedate"].min())
        data_end = pd.Timestamp(self.eng.options_data._data["quotedate"].max())
        for td in trade_dates:
            assert data_start <= td <= data_end


class TestBalanceColumns:

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.eng = run_backtest()

    def test_required_columns(self):
        required = {
            "cash", "stocks capital", "options capital",
            "total capital", "calls capital", "puts capital",
            "% change", "accumulated return",
        }
        actual = set(self.eng.balance.columns)
        missing = required - actual
        assert not missing, f"Missing columns: {missing}"

    def test_per_stock_columns(self):
        for sym, _ in IVY_STOCKS_TUPLES:
            assert sym in self.eng.balance.columns
            assert f"{sym} qty" in self.eng.balance.columns


# ── Generated dataset invariants ───────────────────────────────────────

class TestGeneratedDataInvariants:

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.eng = run_backtest(
            stocks=generated_stocks(),
            stocks_data=load_generated_stocks(),
            options_data=load_generated_options(),
        )

    def test_invariants(self):
        assert_invariants(self.eng, min_trades=5, label="generated")

    def test_many_balance_rows(self):
        assert len(self.eng.balance) >= 10

    def test_initial_capital(self):
        first_tc = self.eng.balance["total capital"].iloc[0]
        assert abs(first_tc - DEFAULT_CAPITAL) < 1.0


# ── Production SPY invariants ──────────────────────────────────────────

class TestProductionDataInvariants:

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.eng = run_backtest(
            strategy_fn=lambda schema: buy_put_strategy(schema, underlying="SPY"),
            stocks=prod_spy_stocks(),
            stocks_data=load_prod_stocks(),
            options_data=load_prod_options(),
        )

    def test_invariants(self):
        assert_invariants(self.eng, min_trades=3, label="production")

    def test_capital_never_negative(self):
        tc = self.eng.balance["total capital"]
        assert (tc >= -1.0).all()
