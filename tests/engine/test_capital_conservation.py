"""Capital conservation invariant: no money should be created or destroyed.

At every row in the balance sheet:
    cash + stocks_capital + options_capital ≈ total_capital

This catches bugs like the one where _execute_option_entries unconditionally
added options_allocation to current_cash, creating money from thin air in
AQR framing.
"""

import math
import os

import numpy as np
import pytest

from options_portfolio_backtester.core.types import (
    Direction,
    OptionType as Type,
    Stock,
)
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData,
    TiingoData,
)
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
STOCKS_FILE = os.path.join(TEST_DIR, "ivy_5assets_data.csv")
OPTIONS_FILE = os.path.join(TEST_DIR, "options_data.csv")


def _ivy_stocks():
    return [
        Stock("VTI", 0.2), Stock("VEU", 0.2), Stock("BND", 0.2),
        Stock("VNQ", 0.2), Stock("DBC", 0.2),
    ]


def _stocks_data():
    data = TiingoData(STOCKS_FILE)
    data._data["adjClose"] = 10
    return data


def _options_data():
    data = HistoricalOptionsData(OPTIONS_FILE)
    data._data.at[2, "ask"] = 1
    data._data.at[2, "bid"] = 0.5
    data._data.at[51, "ask"] = 1.5
    data._data.at[50, "bid"] = 0.5
    data._data.at[130, "bid"] = 0.5
    data._data.at[131, "bid"] = 1.5
    data._data.at[206, "bid"] = 0.5
    data._data.at[207, "bid"] = 1.5
    return data


def _buy_strategy(schema):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 60)
    leg.exit_filter = schema.dte <= 30
    strat.add_legs([leg])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return strat


def _assert_balance_components_sum(balance, rtol=1e-6):
    """Assert cash + stocks + options = total at every row."""
    component_sum = (
        balance["cash"]
        + balance["stocks capital"]
        + balance["options capital"]
    )
    total = balance["total capital"]
    mismatches = ~np.isclose(component_sum, total, rtol=rtol, atol=0.01)
    if mismatches.any():
        bad = balance[mismatches][["cash", "stocks capital", "options capital", "total capital"]].head(5)
        sums = component_sum[mismatches].head(5)
        raise AssertionError(
            f"Components don't sum to total at {mismatches.sum()} rows.\n"
            f"First mismatches:\n{bad}\nComponent sums:\n{sums}"
        )


def _assert_no_capital_spike(balance, initial_capital, max_first_day_ratio=1.01):
    """Assert total capital never jumps above initial on the first day.

    The first rebalance should not create money — total capital on day 1
    should be ≤ initial_capital (plus a small tolerance for rounding).
    """
    first_total = balance["total capital"].iloc[1] if len(balance) > 1 else balance["total capital"].iloc[0]
    assert first_total <= initial_capital * max_first_day_ratio, (
        f"Capital spiked on first day: {first_total:.2f} > {initial_capital * max_first_day_ratio:.2f}. "
        f"Possible money creation."
    )


class TestCapitalConservationAQR:
    """AQR framing: sell stocks to fund puts. No external money."""

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        self.engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.engine.stocks = _ivy_stocks()
        self.engine.stocks_data = stocks_data
        self.engine.options_data = options_data
        self.engine.options_strategy = _buy_strategy(schema)
        self.engine.run(rebalance_freq=1)

    def test_components_sum_to_total(self):
        _assert_balance_components_sum(self.engine.balance)

    def test_no_first_day_spike(self):
        _assert_no_capital_spike(self.engine.balance, 100_000)

    def test_final_capital_plausible(self):
        """With NoCosts and OTM puts, total capital should stay near initial."""
        final = self.engine.balance["total capital"].iloc[-1]
        # Should not grow by more than 50% from options alone on small test data
        assert final < 100_000 * 1.5, f"Suspiciously high final capital: {final}"
        # Should not go negative
        assert final > 0


class TestCapitalConservationSpitznagel:
    """Spitznagel framing: 100% stocks + external put budget."""

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        self.engine = BacktestEngine(
            {"stocks": 1.0, "options": 0.0, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.engine.options_budget_pct = 0.03
        self.engine.stocks = _ivy_stocks()
        self.engine.stocks_data = stocks_data
        self.engine.options_data = options_data
        self.engine.options_strategy = _buy_strategy(schema)
        self.engine.run(rebalance_freq=1)

    def test_components_sum_to_total(self):
        _assert_balance_components_sum(self.engine.balance)


class TestCapitalConservationNoTrades:
    """With impossible entry filter, no trades should happen and capital should be stable."""

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        # Impossible filter: delta > 0 for puts (never true)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.delta > 0)
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])

        self.engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.engine.stocks = _ivy_stocks()
        self.engine.stocks_data = stocks_data
        self.engine.options_data = options_data
        self.engine.options_strategy = strat
        self.engine.run(rebalance_freq=1)

    def test_components_sum_to_total(self):
        _assert_balance_components_sum(self.engine.balance)

    def test_options_capital_always_zero(self):
        assert (self.engine.balance["options capital"] == 0).all()

    def test_no_trades(self):
        assert self.engine.trade_log.empty


class TestCapitalConservationHighBudget:
    """Stress test: 50% AQR allocation. Should NOT create money."""

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        self.engine = BacktestEngine(
            {"stocks": 0.50, "options": 0.50, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.engine.stocks = _ivy_stocks()
        self.engine.stocks_data = stocks_data
        self.engine.options_data = options_data
        self.engine.options_strategy = _buy_strategy(schema)
        self.engine.run(rebalance_freq=1)

    def test_components_sum_to_total(self):
        _assert_balance_components_sum(self.engine.balance)

    def test_no_first_day_spike(self):
        _assert_no_capital_spike(self.engine.balance, 100_000)


# ---------------------------------------------------------------------------
# New tests for the skip-day cash conservation fix
# ---------------------------------------------------------------------------


class TestSkipDayCashConservation:
    """When _execute_option_entries returns early (no candidates), the options
    allocation money must stay as cash -- it must not be destroyed.

    Uses an impossible entry filter (delta > 0 for puts) so puts are never
    found.  Verifies:
      - cash + stocks = total at every step (options capital is always 0)
      - cash is never lower than the stocks-only floor (i.e. options money
        is always preserved in cash on skip days)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        # Impossible filter: delta > 0 for puts (never satisfied)
        strat = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.delta > 0)
        leg.exit_filter = schema.dte <= 30
        strat.add_legs([leg])
        strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

        self.engine = BacktestEngine(
            {"stocks": 0.90, "options": 0.10, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.engine.stocks = _ivy_stocks()
        self.engine.stocks_data = stocks_data
        self.engine.options_data = options_data
        self.engine.options_strategy = strat
        self.engine.run(rebalance_freq=1)

    def test_components_sum_to_total(self):
        _assert_balance_components_sum(self.engine.balance)

    def test_options_capital_always_zero(self):
        assert (self.engine.balance["options capital"] == 0).all(), (
            "Options capital should be zero when no puts are ever entered."
        )

    def test_cash_never_below_options_floor(self):
        """Cash should always hold at least the options portion of total
        capital (since puts were never bought, the money stays as cash)."""
        bal = self.engine.balance
        total = bal["total capital"]
        # On skip days, cash = total - stocks.  Since options_allocation
        # was 10% of total, cash should be >= 10% of total (minus rounding).
        expected_min_cash = total * 0.10 - 0.01
        # Skip the first row (initial balance row, before any rebalance)
        actual_cash = bal["cash"].iloc[1:]
        expected = expected_min_cash.iloc[1:]
        violations = actual_cash < expected
        if violations.any():
            bad = bal.iloc[1:][violations][["cash", "stocks capital", "total capital"]].head(5)
            raise AssertionError(
                f"Cash fell below expected options floor at {violations.sum()} rows.\n"
                f"First violations:\n{bad}"
            )

    def test_no_trades(self):
        assert self.engine.trade_log.empty

    def test_total_stable_with_flat_prices(self):
        """With constant stock prices and no options, total capital should be
        approximately constant (NoCosts means no transaction fees)."""
        bal = self.engine.balance
        total = bal["total capital"].iloc[1:]  # skip pre-rebalance row
        assert total.max() <= 100_000 * 1.001, (
            f"Total grew unexpectedly: {total.max()}"
        )
        assert total.min() >= 100_000 * 0.999, (
            f"Total shrunk unexpectedly: {total.min()}"
        )


class TestAQRDeploymentNeverExceedsTotal:
    """At every point: stocks_capital + options_capital + cash == total_capital.
    Total capital must never exceed what is explained by stock returns and
    option P&L.  This catches the original 'money from thin air' bug where
    options_allocation was double-counted.

    Tests across several AQR allocation ratios.
    """

    @pytest.fixture(
        params=[
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            {"stocks": 0.80, "options": 0.20, "cash": 0},
            {"stocks": 0.50, "options": 0.50, "cash": 0},
            {"stocks": 0.50, "options": 0.30, "cash": 0.20},
        ],
        ids=["97/3/0", "80/20/0", "50/50/0", "50/30/20"],
    )
    def engine(self, request):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        eng = BacktestEngine(
            request.param,
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        eng.stocks = _ivy_stocks()
        eng.stocks_data = stocks_data
        eng.options_data = options_data
        eng.options_strategy = _buy_strategy(schema)
        eng.run(rebalance_freq=1)
        return eng

    def test_components_sum_to_total(self, engine):
        _assert_balance_components_sum(engine.balance)

    def test_total_never_above_initial_on_flat_prices(self, engine):
        """Stock prices are flat at 10 (set by _stocks_data), so stocks
        generate no return.  Total capital should never materially exceed
        initial capital by an unreasonable amount.  With large options
        allocations, option MTM can legitimately swing, so we use a generous
        bound (3x) that catches runaway money creation but not legitimate
        option P&L."""
        bal = engine.balance
        total = bal["total capital"]
        assert total.max() < 100_000 * 3.0, (
            f"Total capital suspiciously high: {total.max():.2f}. "
            f"Possible money creation."
        )

    def test_deployment_never_exceeds_total(self, engine):
        """stocks_capital + options_capital should never exceed total_capital.
        If it does, cash would be negative, meaning we spent money we didn't
        have."""
        bal = engine.balance
        deployed = bal["stocks capital"] + bal["options capital"]
        total = bal["total capital"]
        overdeployed = deployed > total + 0.01
        if overdeployed.any():
            bad = bal[overdeployed][
                ["cash", "stocks capital", "options capital", "total capital"]
            ].head(5)
            raise AssertionError(
                f"Deployed capital exceeds total at {overdeployed.sum()} rows.\n"
                f"First overdeployments:\n{bad}"
            )

    def test_cash_never_negative(self, engine):
        """Cash should never go meaningfully negative (small float noise OK)."""
        bal = engine.balance
        cash = bal["cash"]
        bad_cash = cash < -0.01
        if bad_cash.any():
            bad = bal[bad_cash][
                ["cash", "stocks capital", "options capital", "total capital"]
            ].head(5)
            raise AssertionError(
                f"Cash went negative at {bad_cash.sum()} rows.\n"
                f"First violations:\n{bad}"
            )


class TestAQRRebalanceCycleAccounting:
    """After a full cycle (buy puts -> puts expire/exit -> rebalance), verify
    that total_capital change is explained only by stock price moves and
    option P&L -- not by cash appearing or disappearing.

    Since _stocks_data() sets all prices to 10 (flat), and we use NoCosts,
    the total capital change should come purely from option value changes.
    We verify this by checking that cash + stocks + options = total at every
    row, and that the net change in total capital matches the net change in
    the component sum.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        self.engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.engine.stocks = _ivy_stocks()
        self.engine.stocks_data = stocks_data
        self.engine.options_data = options_data
        self.engine.options_strategy = _buy_strategy(schema)
        self.engine.run(rebalance_freq=1)

    def test_components_sum_to_total(self):
        _assert_balance_components_sum(self.engine.balance)

    def test_total_change_equals_component_change(self):
        """Delta(total) == Delta(cash) + Delta(stocks) + Delta(options).
        If not, money leaked in or out."""
        bal = self.engine.balance
        d_total = bal["total capital"].diff().iloc[1:]
        d_cash = bal["cash"].diff().iloc[1:]
        d_stocks = bal["stocks capital"].diff().iloc[1:]
        d_options = bal["options capital"].diff().iloc[1:]
        d_components = d_cash + d_stocks + d_options
        mismatches = ~np.isclose(d_total, d_components, rtol=1e-6, atol=0.01)
        if mismatches.any():
            bad_idx = d_total[mismatches].index[:5]
            raise AssertionError(
                f"Total capital change does not match component changes at "
                f"{mismatches.sum()} rows.\n"
                f"Dates: {bad_idx.tolist()}\n"
                f"d_total: {d_total[mismatches].head(5).tolist()}\n"
                f"d_components: {d_components[mismatches].head(5).tolist()}"
            )

    def test_no_cash_leak_over_full_run(self):
        """Over the entire run, the total change in capital should equal
        the sum of: stock returns + option P&L.  We verify this by checking
        that [total_final - total_initial] == [sum of period-by-period
        component changes]."""
        bal = self.engine.balance
        total_change = bal["total capital"].iloc[-1] - bal["total capital"].iloc[0]
        component_changes = (
            bal["cash"].diff().sum()
            + bal["stocks capital"].diff().sum()
            + bal["options capital"].diff().sum()
        )
        assert np.isclose(total_change, component_changes, rtol=1e-6, atol=0.01), (
            f"Cumulative total change {total_change:.4f} != "
            f"cumulative component changes {component_changes:.4f}. "
            f"Cash leaked: {total_change - component_changes:.4f}"
        )


class TestAQRvsSpitznagelZeroBudget:
    """With an impossible filter (no puts ever bought), AQR has less equity
    exposure than Spitznagel.  AQR allocates (1 - options_pct) to stocks,
    while Spitznagel allocates 100% to stocks.

    When puts are never bought:
    - AQR 97/3: only 97% in stocks, 3% idle in cash
    - Spitznagel 100/0 + 3% budget: 100% in stocks, budget never spent

    With flat prices both should preserve capital, but Spitznagel should
    have higher (or equal) equity exposure and thus higher (or equal)
    returns in a rising market.  With flat prices (all = 10), they should
    be approximately equal, but AQR should never beat Spitznagel since
    AQR holds less stock.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        stocks_data = _stocks_data()
        options_data = _options_data()
        schema = options_data.schema

        # Impossible filter so no puts are ever entered
        strat_impossible = Strategy(schema)
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        leg.entry_filter = (schema.underlying == "SPX") & (schema.delta > 0)
        leg.exit_filter = schema.dte <= 30
        strat_impossible.add_legs([leg])
        strat_impossible.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

        # AQR framing: 97% stocks, 3% options (from stock allocation)
        self.aqr_engine = BacktestEngine(
            {"stocks": 0.97, "options": 0.03, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.aqr_engine.stocks = _ivy_stocks()
        self.aqr_engine.stocks_data = stocks_data
        self.aqr_engine.options_data = options_data
        self.aqr_engine.options_strategy = strat_impossible
        self.aqr_engine.run(rebalance_freq=1)

        # Spitznagel framing: 100% stocks + external 3% budget
        self.spitz_engine = BacktestEngine(
            {"stocks": 1.0, "options": 0.0, "cash": 0},
            cost_model=NoCosts(),
            initial_capital=100_000,
        )
        self.spitz_engine.options_budget_pct = 0.03
        self.spitz_engine.stocks = _ivy_stocks()
        self.spitz_engine.stocks_data = stocks_data
        self.spitz_engine.options_data = options_data
        self.spitz_engine.options_strategy = strat_impossible
        self.spitz_engine.run(rebalance_freq=1)

    def test_both_conserve_capital(self):
        _assert_balance_components_sum(self.aqr_engine.balance)
        _assert_balance_components_sum(self.spitz_engine.balance)

    def test_aqr_less_equity_than_spitznagel(self):
        """AQR should have strictly less stock capital than Spitznagel,
        since AQR reserves 3% for options (held as cash when puts not found)
        while Spitznagel puts 100% in stocks."""
        aqr_stocks = self.aqr_engine.balance["stocks capital"].iloc[1:]
        spitz_stocks = self.spitz_engine.balance["stocks capital"].iloc[1:]
        # Align on common dates
        common = aqr_stocks.index.intersection(spitz_stocks.index)
        assert len(common) > 0, "No overlapping dates between AQR and Spitznagel"
        assert (aqr_stocks.loc[common] <= spitz_stocks.loc[common] + 0.01).all(), (
            "AQR has more stock capital than Spitznagel -- allocation is wrong."
        )

    def test_aqr_return_leq_spitznagel(self):
        """With flat prices and no puts bought, AQR total return should be
        less than or equal to Spitznagel (AQR holds less stock)."""
        aqr_final = self.aqr_engine.balance["total capital"].iloc[-1]
        spitz_final = self.spitz_engine.balance["total capital"].iloc[-1]
        assert aqr_final <= spitz_final + 0.01, (
            f"AQR final ({aqr_final:.2f}) > Spitznagel final ({spitz_final:.2f}). "
            f"AQR should not outperform with less equity and no options."
        )

    def test_no_trades_in_either(self):
        assert self.aqr_engine.trade_log.empty, "AQR should have no trades"
        assert self.spitz_engine.trade_log.empty, "Spitznagel should have no trades"

    def test_aqr_has_cash_from_unspent_options(self):
        """In AQR framing with impossible filter, the 3% options allocation
        should remain as cash since it was never spent on puts."""
        bal = self.aqr_engine.balance
        # After first rebalance, cash should be roughly 3% of total
        cash = bal["cash"].iloc[1:]
        total = bal["total capital"].iloc[1:]
        ratio = cash / total
        # Should be close to 3% (the unspent options allocation)
        assert (ratio > 0.02).all(), (
            f"AQR cash ratio too low -- options money was destroyed.\n"
            f"Min ratio: {ratio.min():.4f}, expected ~0.03"
        )
