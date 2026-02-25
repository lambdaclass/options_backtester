"""Property-based and fuzz tests for core components.

Uses hypothesis to generate random inputs and verify invariants hold.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from options_backtester.analytics.stats import BacktestStats, PeriodStats
from options_backtester.engine.pipeline import (
    AlgoPipelineBacktester,
    PipelineContext,
    Rebalance,
    RunDaily,
    RunMonthly,
    RunWeekly,
    SelectAll,
    SelectThese,
    SelectRegex,
    WeighEqually,
    WeighSpecified,
    LimitWeights,
    ScaleWeights,
    StepDecision,
)
from options_backtester.execution.cost_model import (
    NoCosts, PerContractCommission, TieredCommission,
)
from options_backtester.execution.fill_model import (
    MarketAtBidAsk, MidPrice, VolumeAwareFill,
)
from options_backtester.execution.signal_selector import (
    FirstMatch, NearestDelta, MaxOpenInterest,
)


# ---------------------------------------------------------------------------
# Strategies (hypothesis)
# ---------------------------------------------------------------------------

daily_return = st.floats(min_value=-0.20, max_value=0.20, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.01, max_value=1e8, allow_nan=False, allow_infinity=False)
price = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)
quantity = st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
rate = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
weight = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _make_balance(returns: list[float], initial: float = 100_000.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(returns) + 1, freq="B")
    capital = [initial]
    for r in returns:
        capital.append(capital[-1] * (1 + r))
    df = pd.DataFrame({"total capital": capital}, index=dates)
    df["% change"] = df["total capital"].pct_change()
    return df


# ---------------------------------------------------------------------------
# BacktestStats invariants
# ---------------------------------------------------------------------------

class TestStatsInvariants:
    @given(st.lists(daily_return, min_size=10, max_size=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_max_drawdown_non_negative(self, returns):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown >= 0

    @given(st.lists(daily_return, min_size=10, max_size=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_max_drawdown_at_most_one(self, returns):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.max_drawdown <= 1.0 + 1e-10

    @given(st.lists(daily_return, min_size=10, max_size=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_avg_drawdown_leq_max(self, returns):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.avg_drawdown <= stats.max_drawdown + 1e-10

    @given(st.lists(daily_return, min_size=10, max_size=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_volatility_non_negative(self, returns):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        assert stats.volatility >= 0

    @given(st.lists(daily_return, min_size=10, max_size=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_total_return_consistent(self, returns):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        # total_return = (final / initial) - 1
        expected = balance["total capital"].iloc[-1] / balance["total capital"].iloc[0] - 1
        assert abs(stats.total_return - expected) < 1e-10

    @given(
        st.lists(daily_return, min_size=10, max_size=100),
        st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_trade_stats_consistent(self, returns, pnls):
        balance = _make_balance(returns)
        trade_pnls = np.array(pnls)
        stats = BacktestStats.from_balance(balance, trade_pnls)
        assert stats.wins + stats.losses == stats.total_trades
        if stats.total_trades > 0:
            assert 0 <= stats.win_pct <= 100
        assert stats.profit_factor >= 0

    @given(st.lists(daily_return, min_size=10, max_size=500))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_dataframe_has_all_rows(self, returns):
        balance = _make_balance(returns)
        stats = BacktestStats.from_balance(balance)
        df = stats.to_dataframe()
        assert df.shape[0] >= 30
        assert df.shape[1] == 1


# ---------------------------------------------------------------------------
# Cost model invariants
# ---------------------------------------------------------------------------

class TestCostModelInvariants:
    @given(price, quantity)
    @settings(max_examples=100)
    def test_no_costs_always_zero(self, p, q):
        assert NoCosts().option_cost(p, q, 100) == 0.0
        assert NoCosts().stock_cost(p, q) == 0.0

    @given(price, quantity, rate)
    @settings(max_examples=100)
    def test_per_contract_non_negative(self, p, q, r):
        model = PerContractCommission(rate=r)
        assert model.option_cost(p, q, 100) >= 0
        assert model.stock_cost(p, q) >= 0

    @given(price, quantity, rate)
    @settings(max_examples=100)
    def test_per_contract_proportional(self, p, q, r):
        model = PerContractCommission(rate=r)
        cost = model.option_cost(p, q, 100)
        expected = r * abs(q)
        assert abs(cost - expected) < 1e-8

    @given(price, quantity)
    @settings(max_examples=100)
    def test_per_contract_symmetric(self, p, q):
        """Commission should be same for buy (+q) and sell (-q)."""
        model = PerContractCommission(rate=0.65)
        assert model.option_cost(p, q, 100) == model.option_cost(p, -q, 100)


# ---------------------------------------------------------------------------
# Fill model invariants
# ---------------------------------------------------------------------------

class TestFillModelInvariants:
    @given(
        st.floats(min_value=0.01, max_value=100, allow_nan=False),
        st.floats(min_value=0.01, max_value=100, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_mid_price_between_bid_ask(self, bid, ask):
        assume(bid <= ask)
        from options_backtester.core.types import Direction
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        model = MidPrice()
        mid = model.get_fill_price(row, Direction.BUY)
        assert bid - 1e-10 <= mid <= ask + 1e-10

    @given(
        st.floats(min_value=0.01, max_value=100, allow_nan=False),
        st.floats(min_value=0.01, max_value=100, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_market_bid_ask_buy_at_ask(self, bid, ask):
        assume(bid <= ask)
        from options_backtester.core.types import Direction
        row = pd.Series({"bid": bid, "ask": ask, "volume": 100})
        model = MarketAtBidAsk()
        assert model.get_fill_price(row, Direction.BUY) == ask
        assert model.get_fill_price(row, Direction.SELL) == bid


# ---------------------------------------------------------------------------
# Pipeline algo invariants
# ---------------------------------------------------------------------------

class TestWeightInvariants:
    @given(st.integers(min_value=2, max_value=20))
    @settings(max_examples=30)
    def test_weigh_equally_sums_to_one(self, n):
        symbols = [f"S{i}" for i in range(n)]
        prices = pd.Series({s: 100.0 for s in symbols})
        ctx = PipelineContext(
            date=pd.Timestamp("2024-01-01"),
            prices=prices,
            total_capital=1_000_000.0,
            cash=1_000_000.0,
            positions={},
        )
        ctx.selected_symbols = symbols
        WeighEqually()(ctx)
        total = sum(ctx.target_weights.values())
        assert abs(total - 1.0) < 1e-10

    def test_limit_weights_caps_with_many_assets(self):
        """With enough assets, LimitWeights caps each at the limit."""
        ctx = PipelineContext(
            date=pd.Timestamp("2024-01-01"),
            prices=pd.Series({f"S{i}": 100 for i in range(10)}),
            total_capital=1_000_000.0,
            cash=1_000_000.0,
            positions={},
        )
        # One huge weight, rest small
        ctx.target_weights = {"S0": 0.91}
        for i in range(1, 10):
            ctx.target_weights[f"S{i}"] = 0.01
        LimitWeights(0.20)(ctx)
        # After clip+renormalize, S0 should be capped at 0.20
        assert ctx.target_weights["S0"] <= 0.20 + 1e-10

    @given(st.floats(min_value=0.01, max_value=5.0, allow_nan=False))
    @settings(max_examples=30)
    def test_scale_weights_multiplies(self, scale):
        ctx = PipelineContext(
            date=pd.Timestamp("2024-01-01"),
            prices=pd.Series({"A": 100, "B": 100}),
            total_capital=1_000_000.0,
            cash=1_000_000.0,
            positions={},
        )
        ctx.target_weights = {"A": 0.3, "B": 0.2}
        ScaleWeights(scale)(ctx)
        assert abs(ctx.target_weights["A"] - 0.3 * scale) < 1e-10
        assert abs(ctx.target_weights["B"] - 0.2 * scale) < 1e-10


# ---------------------------------------------------------------------------
# Pipeline end-to-end fuzz: random prices, verify no crashes + capital > 0
# ---------------------------------------------------------------------------

class TestPipelineFuzz:
    @given(
        st.lists(
            st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
            min_size=5,
            max_size=50,
        ),
        st.floats(min_value=1000.0, max_value=1e7, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_pipeline_no_crash_on_random_prices(self, spy_prices, capital):
        dates = pd.date_range("2024-01-01", periods=len(spy_prices), freq="B")
        prices = pd.DataFrame({"SPY": spy_prices}, index=dates)
        bt = AlgoPipelineBacktester(
            prices=prices,
            initial_capital=capital,
            algos=[RunDaily(), SelectAll(), WeighEqually(), Rebalance()],
        )
        bal = bt.run()
        assert len(bal) > 0
        assert bal["total capital"].iloc[-1] > 0

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=10, max_value=500, allow_nan=False),
                st.floats(min_value=10, max_value=500, allow_nan=False),
            ),
            min_size=5,
            max_size=30,
        ),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_pipeline_multi_asset_no_crash(self, price_pairs):
        spy_prices = [p[0] for p in price_pairs]
        qqq_prices = [p[1] for p in price_pairs]
        dates = pd.date_range("2024-01-01", periods=len(price_pairs), freq="B")
        prices = pd.DataFrame({"SPY": spy_prices, "QQQ": qqq_prices}, index=dates)
        bt = AlgoPipelineBacktester(
            prices=prices,
            initial_capital=100_000.0,
            algos=[RunDaily(), SelectAll(), WeighEqually(), Rebalance()],
        )
        bal = bt.run()
        assert len(bal) > 0
        assert bal["total capital"].iloc[-1] > 0

    @given(
        st.lists(
            st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
            min_size=10,
            max_size=50,
        ),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_stats_from_pipeline_no_crash(self, spy_prices):
        dates = pd.date_range("2024-01-01", periods=len(spy_prices), freq="B")
        prices = pd.DataFrame({"SPY": spy_prices}, index=dates)
        bt = AlgoPipelineBacktester(
            prices=prices,
            initial_capital=100_000.0,
            algos=[RunDaily(), SelectAll(), WeighEqually(), Rebalance()],
        )
        bal = bt.run()
        stats = BacktestStats.from_balance(bal)
        assert stats.max_drawdown >= 0
        assert stats.volatility >= 0
        df = stats.to_dataframe()
        assert df.shape[0] >= 30


# ---------------------------------------------------------------------------
# to_rust_config round-trip invariants
# ---------------------------------------------------------------------------

class TestRustConfigRoundTrip:
    def test_cost_models_have_rust_config(self):
        for model in [NoCosts(), PerContractCommission(0.65), TieredCommission([(10000, 0.65)])]:
            cfg = model.to_rust_config()
            assert isinstance(cfg, dict)
            assert "type" in cfg

    def test_fill_models_have_rust_config(self):
        for model in [MarketAtBidAsk(), MidPrice(), VolumeAwareFill(full_volume_threshold=100)]:
            cfg = model.to_rust_config()
            assert isinstance(cfg, dict)
            assert "type" in cfg

    def test_signal_selectors_have_rust_config(self):
        for model in [FirstMatch(), NearestDelta(-0.30), MaxOpenInterest()]:
            cfg = model.to_rust_config()
            assert isinstance(cfg, dict)
            assert "type" in cfg
