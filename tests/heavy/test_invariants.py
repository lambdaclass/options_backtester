"""Balance sheet and trade log invariants.

Tests run each backtest ONCE and verify structural invariants.
Covers small, generated, and production datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.heavy._test_helpers import (
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    IVY_STOCKS_TUPLES,
    RUST_AVAILABLE,
    assert_invariants,
    buy_put_strategy,
    generated_stocks,
    ivy_stocks,
    load_generated_options,
    load_generated_stocks,
    load_prod_options,
    load_prod_stocks,
    prod_spy_stocks,
    run_backtest,
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

import os as _os_guard

from tests.heavy._test_helpers import _DATA_DIR as _GUARD_DATA_DIR

_needs_generated = pytest.mark.skipif(
    not _os_guard.path.exists(_os_guard.path.join(_GUARD_DATA_DIR, "large_stocks.csv")),
    reason="needs generated data: python tests/heavy/generate_test_data.py",
)
_needs_prod_slice = pytest.mark.skipif(
    not _os_guard.path.exists(_os_guard.path.join(_GUARD_DATA_DIR, "prod_stocks_1y.csv")),
    reason="needs prod slice data: python tests/heavy/extract_prod_slices.py",
)


@_needs_generated
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

@_needs_prod_slice
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


# ── Runtime invariant self-checks (assert_invariants flag) ─────────────
#
# Exercise the in-engine cash-flow (class A) and valuation (class B)
# invariants by enabling `assert_invariants` and confirming the CORRECTED
# engine does not trip them across the configurations where the "free puts"
# (class A) and adjClose-intrinsic (class B) bugs originally lived. If a
# regression reintroduces either bug the Rust engine returns an error, which
# surfaces here as an exception and fails the test.

import math as _math
import os as _os

from options_portfolio_backtester.core.types import Direction as _Dir
from options_portfolio_backtester.core.types import OptionType as _OptType
from options_portfolio_backtester.core.types import Stock as _Stock
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData as _Opts,
)
from options_portfolio_backtester.data.providers import (
    TiingoData as _Stx,
)
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk
from options_portfolio_backtester.strategy.strategy import Strategy as _Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg as _Leg

_PROCESSED = _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "processed")
# Parquet is the canonical processed format (~30x faster to load than the
# CSV). fetch_data.align_dates() keeps it date-aligned with stocks.csv
# since PR #107; before that fix the parquet carried an extra quotedate.
_SPY_OPTS = _os.path.join(_PROCESSED, "options.parquet")
_SPY_STX = _os.path.join(_PROCESSED, "stocks.csv")
_needs_spy = pytest.mark.skipif(
    not (_os.path.exists(_SPY_OPTS) and _os.path.exists(_SPY_STX)),
    reason="needs processed SPY data: python scripts/fetch_data.py all --symbols SPY",
)


def _deep_otm_put(schema):
    leg = _Leg("leg_1", schema, option_type=_OptType.PUT, direction=_Dir.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY") & (schema.dte >= 90) & (schema.dte <= 180)
        & (schema.delta >= -0.10) & (schema.delta <= -0.02)
    )
    leg.entry_sort = ("delta", False)
    leg.exit_filter = schema.dte <= 14
    s = _Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=_math.inf, loss_pct=_math.inf)
    return s


class TestRuntimeInvariantChecks:
    """The corrected engine must satisfy its own cash-flow and valuation
    invariants. These tests turn the checks on and assert no violation."""

    @pytest.fixture(scope="class")
    def spy_data(self):
        # Load the 3.3 GB option chain once for the whole class.
        return _Opts(_SPY_OPTS), _Stx(_SPY_STX)

    def _budget_engine(self, spy_data):
        opts, stx = spy_data
        eng = BacktestEngine(
            {"stocks": 1.0, "options": 0.0, "cash": 0.0},
            initial_capital=1_000_000,
            cost_model=NoCosts(),
            fill_model=MarketAtBidAsk(),
        )
        eng.assert_invariants = True
        eng.stocks = [_Stock("SPY", 1.0)]
        eng.stocks_data = stx
        eng.options_data = opts
        eng.options_strategy = _deep_otm_put(opts.schema)
        return eng

    def test_small_dataset_invariants_hold(self):
        # Fast self-funded-path check on the bundled small dataset: enabling the
        # invariants must not false-positive on a standard backtest.
        from tests.heavy._test_helpers import (
            buy_put_strategy,
            load_small_options,
            load_small_stocks,
        )
        opts = load_small_options()
        eng = BacktestEngine(
            DEFAULT_ALLOC, initial_capital=DEFAULT_CAPITAL,
            cost_model=NoCosts(), fill_model=MarketAtBidAsk(),
        )
        eng.assert_invariants = True
        eng.stocks = ivy_stocks()
        eng.stocks_data = load_small_stocks()
        eng.options_data = opts
        eng.options_strategy = buy_put_strategy(opts.schema)
        eng.run(rebalance_freq=1, rebalance_unit="BMS")
        assert len(eng.balance) > 0

    @_needs_spy
    @pytest.mark.parametrize("annual_budget", [0.005, 0.020, 0.033])
    def test_spy_externally_funded_annual_invariants_hold(self, spy_data, annual_budget):
        eng = self._budget_engine(spy_data)
        eng.options_budget_annual_pct = annual_budget
        # No exception == both invariants held across the whole run.
        eng.run(rebalance_freq=1, rebalance_unit="BMS", check_exits_daily=True)
        assert len(eng.balance) > 1

    @_needs_spy
    def test_spy_per_rebalance_budget_invariants_hold(self, spy_data):
        # The per-rebalance framing (heavier premium spend) exercises far more
        # exits and intrinsic fallbacks — the regime where the bugs were loudest.
        eng = self._budget_engine(spy_data)
        eng.options_budget_pct = 0.033
        eng.run(rebalance_freq=1, rebalance_unit="BMS", check_exits_daily=False)
        assert len(eng.balance) > 1


class TestExitPriceEnvelope:
    """Truly independent class-B oracle (no shared code with the engine).

    Every exit in the trade log must price inside the day's quote envelope,
    reconstructed here in pandas directly from the raw CSVs:

    - contract quoted on exit date (non-null bid) -> exit price within
      [min(bid, ask), max(bid, ask)];
    - contract unquoted (expired / missing / null bid) -> exit price no
      greater than intrinsic value computed from the UNADJUSTED close.

    The in-engine class-B invariant recomputes intrinsic via the same
    helpers the engine uses, so a bug inside those helpers passes it. This
    test rebuilds the bound from the raw data instead — the pre-fix
    adjClose bug (phantom intrinsic ~ strike-minus-adjusted-spot) fails it
    immediately, no matter where in the engine the wrong price came from.
    """

    SPC = 100

    @pytest.fixture(scope="class")
    def envelope_run(self):
        opts = _Opts(_SPY_OPTS)
        stx = _Stx(_SPY_STX)
        eng = BacktestEngine(
            {"stocks": 1.0, "options": 0.0, "cash": 0.0},
            initial_capital=1_000_000,
            cost_model=NoCosts(),
            fill_model=MarketAtBidAsk(),
        )
        # Per-rebalance budget, NO daily exits: contracts routinely expire
        # before the next rebalance, so the intrinsic fallback is exercised
        # heavily — the exact regime where the adjClose bug lived.
        eng.options_budget_pct = 0.033
        eng.stocks = [_Stock("SPY", 1.0)]
        eng.stocks_data = stx
        eng.options_data = opts
        eng.options_strategy = _deep_otm_put(opts.schema)
        eng.run(rebalance_freq=1, rebalance_unit="BMS", check_exits_daily=False)
        return eng, opts, stx

    @_needs_spy
    def test_every_exit_prices_inside_envelope(self, envelope_run):
        eng, opts, stx = envelope_run
        tl = eng.trade_log
        exits = tl[tl[("leg_1", "order")].isin(["STC", "BTC"])]
        assert len(exits) > 0, "no exits to check"

        # Quote lookup for exited contracts only: (contract, date) -> bid, ask
        od = opts._data
        schema = opts.schema
        c_col, d_col = schema["contract"], schema["date"]
        contracts = set(exits[("leg_1", "contract")])
        sub = od[od[c_col].isin(contracts)][[c_col, d_col, "bid", "ask"]]
        quotes = {
            (r[0], r[1]): (r[2], r[3])
            for r in sub.itertuples(index=False, name=None)
        }

        # Unadjusted close by date (raw column, not DayStocks/adjClose).
        sd = stx._data
        spy = sd[sd[stx.schema["symbol"]] == "SPY"]
        unadj = dict(zip(spy[stx.schema["date"]], spy["close"]))

        eps = 1e-6
        checked_quoted = checked_fallback = 0
        for _, row in exits.iterrows():
            # Trade-log dates are serialized as strings; quote/stock keys are
            # Timestamps.
            date = pd.Timestamp(row[("totals", "date")])
            contract = row[("leg_1", "contract")]
            strike = float(row[("leg_1", "strike")])
            price = abs(float(row[("leg_1", "cost")])) / self.SPC

            bid_ask = quotes.get((contract, date))
            bid = bid_ask[0] if bid_ask is not None else None
            if bid is not None and not np.isnan(bid):
                ask = bid_ask[1]
                hi = max(bid, ask) if not np.isnan(ask) else bid
                lo = min(bid, ask) if not np.isnan(ask) else bid
                assert lo - eps <= price <= hi + eps, (
                    f"{contract} exit {date}: price {price} outside "
                    f"quote envelope [{lo}, {hi}]"
                )
                checked_quoted += 1
            else:
                spot = float(unadj.get(date, 0.0))
                intrinsic = max(strike - spot, 0.0)  # long puts only here
                assert -eps <= price <= intrinsic + eps, (
                    f"{contract} exit {date}: unquoted exit priced {price}, "
                    f"unadjusted intrinsic bound {intrinsic} "
                    f"(strike {strike}, raw close {spot})"
                )
                checked_fallback += 1

        # The oracle is only meaningful if both branches were exercised.
        assert checked_quoted > 0, "no quoted exits checked"
        assert checked_fallback > 0, (
            "no intrinsic-fallback exits occurred — config no longer "
            "exercises the regime the adjClose bug lived in"
        )
