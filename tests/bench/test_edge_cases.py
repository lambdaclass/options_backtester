"""Edge-case regression tests.

Each test runs the backtest ONCE and checks invariants.
"""

from __future__ import annotations

import pytest

from tests.bench._test_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    _make_engine,
    ivy_stocks,
    load_small_stocks,
    load_small_options,
    buy_put_strategy,
    buy_call_strategy,
    sell_put_strategy,
    run_backtest,
    assert_invariants,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# ── Allocation edge cases ─────────────────────────────────────────────

class TestAllocationEdgeCases:

    def test_zero_options(self):
        alloc = {"stocks": 0.9, "options": 0.0, "cash": 0.1}
        eng = run_backtest(alloc=alloc)
        assert len(eng.balance) > 0
        assert eng.trade_log.empty

    def test_high_options(self):
        alloc = {"stocks": 0.09, "options": 0.90, "cash": 0.01}
        eng = run_backtest(alloc=alloc)
        assert_invariants(eng, label="high_options")

    def test_tiny_stocks(self):
        alloc = {"stocks": 0.01, "options": 0.89, "cash": 0.10}
        eng = run_backtest(alloc=alloc)
        assert_invariants(eng, label="tiny_stocks")


# ── Capital edge cases ────────────────────────────────────────────────

class TestCapitalEdgeCases:

    def test_tiny_capital(self):
        eng = run_backtest(capital=1_000)
        assert_invariants(eng, label="tiny_capital")

    def test_huge_capital(self):
        eng = run_backtest(capital=100_000_000)
        assert_invariants(eng, label="huge_capital")


# ── Rebalance edge cases ─────────────────────────────────────────────

class TestRebalanceEdgeCases:

    def test_weekly_rebalance(self):
        eng = run_backtest(rebalance_unit="W-MON")
        assert_invariants(eng, min_trades=1, label="weekly_rebalance")


# ── Direction and type ────────────────────────────────────────────────

class TestDirectionAndType:

    def test_sell_put(self):
        eng = run_backtest(strategy_fn=sell_put_strategy)
        # Sell strategies can go deeply negative (unlimited downside risk)
        assert len(eng.balance) > 0
        assert not eng.trade_log.empty

    def test_buy_call(self):
        eng = run_backtest(strategy_fn=buy_call_strategy)
        assert_invariants(eng, label="buy_call")


# ── SMA gating ───────────────────────────────────────────────────────

class TestSMAGating:

    def test_sma_50(self):
        eng = run_backtest(sma_days=50)
        assert_invariants(eng, label="sma_50")


# ── Options budget pct ───────────────────────────────────────────────

class TestOptionsBudgetPct:

    def test_budget_limits_spending(self):
        eng = _make_engine(
            DEFAULT_ALLOC, DEFAULT_CAPITAL,
            ivy_stocks(), load_small_stocks(), load_small_options(),
            buy_put_strategy,
        )
        eng.options_budget_pct = 0.005
        eng.run(rebalance_freq=1, rebalance_unit="BMS")
        assert_invariants(eng, label="budget_pct")


# ── No matching entries ──────────────────────────────────────────────

class TestNoMatchingEntries:

    def test_filter_matches_nothing(self):
        eng = run_backtest(
            strategy_fn=lambda schema: buy_put_strategy(
                schema, underlying="NONEXISTENT"
            ),
        )
        assert len(eng.balance) > 0
        assert eng.trade_log.empty
