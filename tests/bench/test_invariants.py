"""Balance sheet and trade log invariants that must hold for both paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    IVY_STOCKS_TUPLES,
    ivy_stocks,
    load_small_stocks,
    load_small_options,
    generated_stocks,
    load_generated_stocks,
    load_generated_options,
    prod_spy_stocks,
    load_prod_stocks,
    load_prod_options,
    buy_put_strategy,
    sell_put_strategy,
    strangle_strategy,
    run_rust,
    run_python,
    run_python_ex,
    run_rust_ex,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


def _run_both():
    """Run default config on both paths, return (py_eng, rs_eng)."""
    py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)
    rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy)
    return py, rs


class TestBalanceSheetInvariants:
    """Properties that must hold for the balance DataFrame on both paths."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_both()

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_total_capital_equals_parts(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        bal = eng.balance

        if "options capital" in bal.columns and "stocks capital" in bal.columns:
            reconstructed = (
                bal["cash"] + bal["stocks capital"] + bal["options capital"]
            )
            assert np.allclose(
                bal["total capital"].values, reconstructed.values,
                rtol=1e-4, atol=1.0,
            ), f"[{path}] total capital != cash + stocks + options"

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_capital_never_negative(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        tc = eng.balance["total capital"]
        assert (tc >= -1.0).all(), (
            f"[{path}] negative total capital found: min={tc.min()}"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_initial_capital_correct(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        first_tc = eng.balance["total capital"].iloc[0]
        assert abs(first_tc - DEFAULT_CAPITAL) < 1.0, (
            f"[{path}] first row capital={first_tc}, expected={DEFAULT_CAPITAL}"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_balance_dates_monotonic(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        idx = eng.balance.index
        assert idx.is_monotonic_increasing, (
            f"[{path}] balance index is not monotonically increasing"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_balance_not_empty(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        assert len(eng.balance) > 1, (
            f"[{path}] balance has only {len(eng.balance)} rows"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_cash_column_exists(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        assert "cash" in eng.balance.columns, (
            f"[{path}] 'cash' column missing from balance"
        )


class TestTradeLogInvariants:
    """Properties that must hold for the trade log on both paths."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_both()

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_trade_log_not_empty(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        assert not eng.trade_log.empty, (
            f"[{path}] trade log is empty with default config"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_entry_costs_nonzero(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        if eng.trade_log.empty:
            pytest.skip("No trades to check")
        costs = eng.trade_log["totals"]["cost"].values
        assert all(c != 0 for c in costs), (
            f"[{path}] found zero-cost entries: {costs}"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_qty_positive_on_entry(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        if eng.trade_log.empty:
            pytest.skip("No trades to check")
        qtys = eng.trade_log["totals"]["qty"].values
        assert all(q > 0 for q in qtys), (
            f"[{path}] found non-positive qty: {qtys}"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_trade_dates_within_data_range(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        if eng.trade_log.empty:
            pytest.skip("No trades to check")
        # Dates live in the ("totals", "date") column, not the index
        trade_dates = pd.to_datetime(eng.trade_log["totals"]["date"]).unique()
        data_start = pd.Timestamp(eng.options_data._data["quotedate"].min())
        data_end = pd.Timestamp(eng.options_data._data["quotedate"].max())
        for td in trade_dates:
            assert data_start <= td <= data_end, (
                f"[{path}] trade date {td} outside data range "
                f"[{data_start}, {data_end}]"
            )


class TestBalanceColumnCompleteness:
    """Verify balance DataFrames have correct columns on both paths."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_both()

    def test_balance_columns_match(self):
        """Python and Rust balance DataFrames have identical column sets."""
        py_cols = set(self.py_eng.balance.columns)
        rs_cols = set(self.rs_eng.balance.columns)
        assert py_cols == rs_cols, (
            f"Column mismatch:\n"
            f"  in py only: {py_cols - rs_cols}\n"
            f"  in rs only: {rs_cols - py_cols}"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_required_balance_columns(self, path):
        """Assert presence of all required balance columns."""
        eng = self.py_eng if path == "python" else self.rs_eng
        required = {
            "cash", "stocks capital", "options capital",
            "total capital", "calls capital", "puts capital",
            "% change", "accumulated return",
        }
        actual = set(eng.balance.columns)
        missing = required - actual
        assert not missing, (
            f"[{path}] Missing required columns: {missing}"
        )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_per_stock_columns(self, path):
        """Assert per-stock value and qty columns exist for each stock."""
        eng = self.py_eng if path == "python" else self.rs_eng
        for sym, _ in IVY_STOCKS_TUPLES:
            assert sym in eng.balance.columns, (
                f"[{path}] Missing stock value column: {sym}"
            )
            assert f"{sym} qty" in eng.balance.columns, (
                f"[{path}] Missing stock qty column: {sym} qty"
            )


class TestBalanceColumnCompletenessParametrized:
    """Verify balance DataFrames have the expected columns (parametrized variant)."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_both()

    def test_balance_columns_match(self):
        """Python and Rust balance DataFrames have identical column sets."""
        py_cols = set(self.py_eng.balance.columns)
        rs_cols = set(self.rs_eng.balance.columns)
        assert py_cols == rs_cols, (
            f"Column mismatch:\n"
            f"  in py only: {py_cols - rs_cols}\n"
            f"  in rs only: {rs_cols - py_cols}"
        )

    @pytest.mark.parametrize("col", [
        "cash", "stocks capital", "options capital", "total capital",
        "calls capital", "puts capital", "% change", "accumulated return",
    ])
    def test_required_balance_columns(self, col):
        """Assert presence of all required balance columns in both paths."""
        assert col in self.py_eng.balance.columns, (
            f"'{col}' missing from Python balance"
        )
        assert col in self.rs_eng.balance.columns, (
            f"'{col}' missing from Rust balance"
        )

    def test_per_stock_columns(self):
        """Assert per-stock value and qty columns exist for each stock."""
        for stock in ivy_stocks():
            sym = stock.symbol
            for eng, path in [(self.py_eng, "python"), (self.rs_eng, "rust")]:
                assert sym in eng.balance.columns, (
                    f"[{path}] '{sym}' value column missing"
                )
                assert f"{sym} qty" in eng.balance.columns, (
                    f"[{path}] '{sym} qty' column missing"
                )


class TestCrossPathConsistency:
    """Invariants that must hold between the two paths."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_both()

    def test_same_number_of_trades(self):
        assert self.py_eng.trade_log.shape == self.rs_eng.trade_log.shape

    def test_same_balance_length(self):
        assert len(self.py_eng.balance) == len(self.rs_eng.balance)

    def test_same_final_capital(self):
        py_final = self.py_eng.balance["total capital"].iloc[-1]
        rs_final = self.rs_eng.balance["total capital"].iloc[-1]
        assert abs(py_final - rs_final) < 1.0, (
            f"py={py_final}, rs={rs_final}"
        )


# ══════════════════════════════════════════════════════════════════════
# Large-dataset invariants (generated synthetic + production SPY data)
# ══════════════════════════════════════════════════════════════════════

def _run_gen_pair():
    """Run Python + Rust on generated data with monthly rebalancing."""
    common = dict(
        alloc=DEFAULT_ALLOC, capital=DEFAULT_CAPITAL,
        strategy_fn=buy_put_strategy,
        rebalance_freq=1, rebalance_unit='BMS',
        stocks=generated_stocks(),
        stocks_data=load_generated_stocks(),
        options_data=load_generated_options(),
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


def _run_prod_pair():
    """Run Python + Rust on production SPY data with monthly rebalancing."""
    common = dict(
        alloc=DEFAULT_ALLOC, capital=DEFAULT_CAPITAL,
        strategy_fn=lambda schema: buy_put_strategy(schema, underlying="SPY"),
        rebalance_freq=1, rebalance_unit='BMS',
        stocks=prod_spy_stocks(),
        stocks_data=load_prod_stocks(),
        options_data=load_prod_options(),
    )
    py = run_python_ex(**common)
    rs = run_rust_ex(**common)
    return py, rs


class TestLargeDataInvariants:
    """Invariants on the large generated dataset."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_gen_pair()

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_total_capital_equals_parts(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        bal = eng.balance
        if "options capital" in bal.columns and "stocks capital" in bal.columns:
            reconstructed = bal["cash"] + bal["stocks capital"] + bal["options capital"]
            # Skip first row (initial row has zero stocks/options)
            assert np.allclose(
                bal["total capital"].values[1:], reconstructed.values[1:],
                rtol=1e-4, atol=1.0,
            ), f"[{path}] total capital != cash + stocks + options"

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_capital_never_negative(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        tc = eng.balance["total capital"]
        assert (tc >= -1.0).all(), f"[{path}] negative total capital: min={tc.min()}"

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_no_negative_stock_qty(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        bal = eng.balance
        for col in bal.columns:
            if col.endswith(" qty"):
                vals = pd.to_numeric(bal[col], errors="coerce").dropna()
                assert (vals >= -0.01).all(), (
                    f"[{path}] negative qty in '{col}': min={vals.min()}"
                )

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_balance_dates_monotonic(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        idx = eng.balance.index
        assert idx.is_monotonic_increasing

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_initial_capital_correct(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        first_tc = eng.balance["total capital"].iloc[0]
        assert abs(first_tc - DEFAULT_CAPITAL) < 1.0

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_balance_has_many_rows(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        assert len(eng.balance) >= 10, (
            f"[{path}] expected 10+ balance rows, got {len(eng.balance)}"
        )

    def test_trade_log_not_empty(self):
        assert not self.py_eng.trade_log.empty
        assert not self.rs_eng.trade_log.empty

    def test_trade_counts_match(self):
        assert self.py_eng.trade_log.shape == self.rs_eng.trade_log.shape


class TestProductionDataInvariants:
    """Invariants on production SPY data."""

    @pytest.fixture(autouse=True)
    def _engines(self):
        self.py_eng, self.rs_eng = _run_prod_pair()

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_total_capital_equals_parts(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        bal = eng.balance
        if "options capital" in bal.columns and "stocks capital" in bal.columns:
            reconstructed = bal["cash"] + bal["stocks capital"] + bal["options capital"]
            assert np.allclose(
                bal["total capital"].values[1:], reconstructed.values[1:],
                rtol=1e-4, atol=1.0,
            ), f"[{path}] total capital != cash + stocks + options"

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_capital_never_negative(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        tc = eng.balance["total capital"]
        assert (tc >= -1.0).all(), f"[{path}] negative total capital: min={tc.min()}"

    @pytest.mark.parametrize("path", ["python", "rust"])
    def test_cash_column_always_present(self, path):
        eng = self.py_eng if path == "python" else self.rs_eng
        assert "cash" in eng.balance.columns

    def test_trade_counts_match(self):
        assert self.py_eng.trade_log.shape == self.rs_eng.trade_log.shape
