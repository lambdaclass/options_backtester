"""Regression guards for the bug classes that have actually burned us.

Each test here encodes an *independent oracle* — a fact about the economics
or the API that must hold regardless of how the engine is implemented — so
the bug classes documented in CHANGELOG.md cannot quietly return:

- silent dead config (`check_exits_daily` attribute no-op),
- cash-flow asymmetry / phantom money (the "free puts" bug, class A),
- mispriced positions (the adjClose intrinsic bug, class B),
- budget framing confusion (per-rebalance treated as annual).

Everything in this file runs on the small bundled dataset so it executes in
the default suite (and therefore CI) on every run — unlike the opt-in bench
suite, which needs the full SPY chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_portfolio_backtester import BacktestEngine
from tests.heavy._test_helpers import (
    DEFAULT_CAPITAL,
    buy_put_strategy,
    ivy_stocks,
    load_small_options,
    load_small_stocks,
)


def _engine(alloc, **attrs):
    from options_portfolio_backtester.execution.cost_model import NoCosts
    from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk

    opts = load_small_options()
    eng = BacktestEngine(
        alloc, initial_capital=DEFAULT_CAPITAL,
        cost_model=NoCosts(), fill_model=MarketAtBidAsk(),
    )
    eng.stocks = ivy_stocks()
    eng.stocks_data = load_small_stocks()
    eng.options_data = opts
    eng.options_strategy = buy_put_strategy(opts.schema)
    for k, v in attrs.items():
        setattr(eng, k, v)
    return eng


# ── Silent dead config ─────────────────────────────────────────────────

class TestUnknownConfigRejected:
    """`engine.check_exits_daily = True` was silently ignored for months
    because run() never read the attribute. The engine is now sealed: any
    unknown attribute raises instead of becoming dead config."""

    def test_typo_raises_attribute_error(self):
        eng = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        with pytest.raises(AttributeError, match="unknown config"):
            eng.check_exit_daily = True  # missing 's' — the classic typo

    def test_suggestion_in_error(self):
        eng = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        with pytest.raises(AttributeError, match="check_exits_daily"):
            eng.check_exits_dailyy = True

    def test_known_attributes_still_settable(self):
        eng = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        eng.check_exits_daily = True
        eng.assert_invariants = True
        eng.rebalance_stocks_on_exit = True
        eng.options_budget_annual_pct = 0.01
        assert eng.check_exits_daily and eng.assert_invariants


# ── Runtime invariants exercised on every CI run ───────────────────────

class TestRuntimeInvariantsInDefaultSuite:
    """The class-A (cash-flow) and class-B (valuation) in-engine guards must
    hold on a standard run. The SPY-scale version lives in tests/heavy; this
    small-data version makes sure every default `pytest` run exercises the
    invariant code path at all."""

    def test_self_funded_run_passes_invariants(self):
        eng = _engine({"stocks": 0.6, "options": 0.3, "cash": 0.1},
                      assert_invariants=True)
        eng.run(rebalance_freq=1, rebalance_unit="BMS")
        assert len(eng.balance) > 0

    def test_externally_funded_run_passes_invariants(self):
        eng = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0},
                      assert_invariants=True)
        eng.options_budget_pct = 0.05
        eng.run(rebalance_freq=1, rebalance_unit="BMS", check_exits_daily=True)
        assert len(eng.balance) > 0


# ── Phantom money / budget oracles ─────────────────────────────────────

class TestBudgetOracles:

    def test_zero_budget_equals_stock_only(self):
        """An options budget of zero must produce *exactly* the stock-only
        result. Any divergence means option positions created value or cash
        out of nothing — the phantom-money class."""
        base = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        base.run(rebalance_freq=1, rebalance_unit="BMS")

        zero = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        zero.options_budget_annual_pct = 0.0
        zero.run(rebalance_freq=1, rebalance_unit="BMS")

        a = base.balance["total capital"].to_numpy(dtype=float)
        b = zero.balance["total capital"].to_numpy(dtype=float)
        assert a.shape == b.shape
        assert np.allclose(a, b, rtol=1e-9, atol=1e-6), (
            "zero options budget diverged from stock-only run"
        )

    def test_unfilled_budget_is_fully_clawed_back(self):
        """A positive external budget whose entry filter never matches must
        leave the portfolio EXACTLY equal to the stock-only run. The engine
        temporarily injects the budget into cash before attempting entries
        and must claw back every cent on no-fill — a leak in that branch is
        free money every rebalance (the cash-flow bug class)."""
        base = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        base.run(rebalance_freq=1, rebalance_unit="BMS")

        nofill = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        nofill.options_budget_pct = 0.05
        # Impossible entry filter: nothing has 10,000+ days to expiry.
        schema = nofill.options_data.schema
        for leg in nofill.options_strategy.legs:
            leg.entry_filter = schema.dte >= 10_000
        nofill.run(rebalance_freq=1, rebalance_unit="BMS")

        a = base.balance["total capital"].to_numpy(dtype=float)
        b = nofill.balance["total capital"].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-6), (
            "unfilled external budget leaked into the portfolio: "
            f"final {b[-1]:,.2f} vs stock-only {a[-1]:,.2f}"
        )

    def test_per_rebalance_vs_annual_budget_distinct(self):
        """`options_budget_pct` is per-rebalance; `options_budget_annual_pct`
        is per-year. With monthly rebalancing the same numeric value must
        spend ~12x more premium in per-rebalance mode. If the two runs come
        out (near-)identical, one knob has been silently rewired to the
        other's semantics."""
        per_reb = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        per_reb.options_budget_pct = 0.05
        per_reb.run(rebalance_freq=1, rebalance_unit="BMS")

        annual = _engine({"stocks": 1.0, "options": 0.0, "cash": 0.0})
        annual.options_budget_annual_pct = 0.05
        annual.run(rebalance_freq=1, rebalance_unit="BMS")

        per_reb_trades = len(per_reb.trade_log)
        annual_trades = len(annual.trade_log)
        # The per-rebalance run has 12x the budget per rebalance; on the
        # small dataset that must show up as at least as many trades and a
        # different balance path.
        assert per_reb_trades >= annual_trades
        a = per_reb.balance["total capital"].to_numpy(dtype=float)
        b = annual.balance["total capital"].to_numpy(dtype=float)
        assert not np.allclose(a, b, rtol=1e-6), (
            "per-rebalance and annual budgets produced identical runs — "
            "one knob is mis-wired to the other's semantics"
        )
