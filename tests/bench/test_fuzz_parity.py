"""Property-based tests and fuzzing for Rust/Python parity.

Uses hypothesis to generate random inputs and verify that Rust and Python
implementations produce identical results across a wide range of scenarios.
"""

import math
import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

try:
    import polars as pl
    from options_portfolio_backtester._ob_rust import (
        compute_stats as rust_compute_stats,
        compute_exit_mask as rust_compute_exit_mask,
        apply_filter as rust_apply_filter,
        compile_filter as rust_compile_filter,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed"),
]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_floats = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
small_positive = st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False)
pct_floats = st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False)
daily_returns = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Stats parity fuzzing
# ---------------------------------------------------------------------------

class TestStatsFuzz:
    """Fuzz compute_stats: random returns and PnLs must match Python exactly."""

    def _python_stats(self, returns, pnls, rf):
        n = len(returns)
        if n == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                    "profit_factor": 0.0, "win_rate": 0.0, "total_trades": 0}
        total_ret = 1.0
        for r in returns:
            total_ret *= (1 + r)
        total_ret -= 1.0

        daily_rf = rf / 252.0
        excess = [r - daily_rf for r in returns]
        mean_ex = np.mean(excess)
        std_ex = np.std(excess, ddof=1) if len(excess) > 1 else 0
        sharpe = mean_ex / std_ex * np.sqrt(252) if std_ex > 0 else 0

        peak = eq = 1.0
        max_dd = 0.0
        for r in returns:
            eq *= (1 + r)
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)

        gp = sum(p for p in pnls if p > 0)
        gl = sum(abs(p) for p in pnls if p <= 0)
        pf = gp / gl if gl > 0 else 0
        wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0

        return {"total_return": total_ret, "sharpe_ratio": sharpe,
                "max_drawdown": max_dd, "profit_factor": pf, "win_rate": wr,
                "total_trades": len(pnls)}

    @given(
        returns=st.lists(daily_returns, min_size=1, max_size=500),
        pnls=st.lists(finite_floats, min_size=0, max_size=100),
        rf=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_stats_match(self, returns, pnls, rf):
        rust = rust_compute_stats(returns, pnls, rf)
        py = self._python_stats(returns, pnls, rf)

        assert np.isclose(rust["total_return"], py["total_return"], rtol=1e-6, atol=1e-12), \
            f"total_return: rust={rust['total_return']} py={py['total_return']}"
        assert np.isclose(rust["max_drawdown"], py["max_drawdown"], rtol=1e-6, atol=1e-12), \
            f"max_drawdown: rust={rust['max_drawdown']} py={py['max_drawdown']}"
        assert np.isclose(rust["profit_factor"], py["profit_factor"], rtol=1e-6, atol=1e-12), \
            f"profit_factor: rust={rust['profit_factor']} py={py['profit_factor']}"
        assert rust["total_trades"] == py["total_trades"]
        assert np.isclose(rust["win_rate"], py["win_rate"], rtol=1e-6, atol=1e-12)

        if py["sharpe_ratio"] != 0 and abs(py["sharpe_ratio"]) < 1e10:
            assert np.isclose(rust["sharpe_ratio"], py["sharpe_ratio"], rtol=1e-4), \
                f"sharpe: rust={rust['sharpe_ratio']} py={py['sharpe_ratio']}"

    @given(returns=st.lists(daily_returns, min_size=0, max_size=0))
    def test_empty_returns(self, returns):
        rust = rust_compute_stats(returns, [], 0.0)
        assert rust["total_return"] == 0.0
        assert rust["total_trades"] == 0


# ---------------------------------------------------------------------------
# Exit mask parity fuzzing
# ---------------------------------------------------------------------------

class TestExitMaskFuzz:
    """Fuzz threshold exit mask: random costs must match Python exactly."""

    def _python_mask(self, entries, currents, profit, loss):
        mask = []
        for e, c in zip(entries, currents):
            should_exit = False
            if e != 0:
                pnl = (c - e) / abs(e)
                if profit is not None and pnl >= profit:
                    should_exit = True
                if loss is not None and pnl <= -loss:
                    should_exit = True
            mask.append(should_exit)
        return mask

    @given(
        n=st.integers(min_value=1, max_value=500),
        profit=st.one_of(st.none(), pct_floats),
        loss=st.one_of(st.none(), pct_floats),
        data=st.data(),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_exit_mask_match(self, n, profit, loss, data):
        assume(profit is not None or loss is not None)
        entries = data.draw(st.lists(
            positive_floats, min_size=n, max_size=n
        ))
        currents = data.draw(st.lists(
            positive_floats, min_size=n, max_size=n
        ))

        rust = rust_compute_exit_mask(entries, currents,
                                       profit_pct=profit, loss_pct=loss)
        py = self._python_mask(entries, currents, profit, loss)
        assert rust == py, f"Mismatch at profit={profit} loss={loss}"

    @given(
        entries=st.lists(positive_floats, min_size=1, max_size=50),
    )
    def test_no_thresholds_no_exits(self, entries):
        """With no thresholds set, nothing should exit."""
        currents = entries  # same as entry = 0% PnL
        rust = rust_compute_exit_mask(entries, currents)
        assert all(not x for x in rust)


# ---------------------------------------------------------------------------
# Filter parity fuzzing
# ---------------------------------------------------------------------------

class TestFilterFuzz:
    """Fuzz filter application: random DataFrames must match pandas eval."""

    @given(
        n=st.integers(min_value=1, max_value=200),
        threshold=st.integers(min_value=0, max_value=500),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_numeric_filter_match(self, n, threshold, data):
        values = data.draw(st.lists(
            st.integers(min_value=0, max_value=1000), min_size=n, max_size=n
        ))
        df = pd.DataFrame({"dte": values})
        query = f"dte >= {threshold}"

        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count, \
            f"query={query} py={py_count} rust={rust_result.height}"

    @given(
        n=st.integers(min_value=1, max_value=200),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_string_eq_filter_match(self, n, data):
        types = data.draw(st.lists(
            st.sampled_from(["call", "put"]), min_size=n, max_size=n
        ))
        df = pd.DataFrame({"type": types})
        query = "type == 'put'"

        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count

    @given(
        n=st.integers(min_value=1, max_value=200),
        lo=st.integers(min_value=0, max_value=500),
        hi=st.integers(min_value=0, max_value=500),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_range_filter_match(self, n, lo, hi, data):
        assume(lo <= hi)
        values = data.draw(st.lists(
            st.integers(min_value=0, max_value=1000), min_size=n, max_size=n
        ))
        df = pd.DataFrame({"dte": values})
        query = f"(dte >= {lo}) & (dte <= {hi})"

        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count

    @given(
        n=st.integers(min_value=1, max_value=200),
        threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_compound_filter_match(self, n, threshold, data):
        types = data.draw(st.lists(
            st.sampled_from(["call", "put"]), min_size=n, max_size=n
        ))
        asks = data.draw(st.lists(
            st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n,
        ))
        df = pd.DataFrame({"type": types, "ask": asks})
        query = f"(type == 'put') & (ask > {threshold})"

        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count


# ---------------------------------------------------------------------------
# Compiled filter reuse fuzzing
# ---------------------------------------------------------------------------

class TestFilterKnownEdgeCases:
    """Edge cases discovered by fuzzing."""

    def test_scientific_notation_in_query(self):
        df = pd.DataFrame({"ask": [0.0, 1.0]})
        query = "ask > 1e-5"
        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count

    def test_scientific_notation_negative_exp(self):
        df = pd.DataFrame({"val": [0.0, 1e-10, 1e-3, 1.0, 1e5]})
        query = "val > 1e-4"
        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count

    def test_scientific_notation_large(self):
        df = pd.DataFrame({"strike": [100.0, 1500.0, 5000.0]})
        query = "strike >= 1.5E3"
        py_count = len(df[df.eval(query)])
        rust_result = rust_apply_filter(query, pl.from_pandas(df))
        assert rust_result.height == py_count


class TestCompiledFilterFuzz:
    """Compiled filters must give same results when reused across DataFrames."""

    @given(
        threshold=st.integers(min_value=0, max_value=500),
        data=st.data(),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compiled_reuse(self, threshold, data):
        query = f"dte >= {threshold}"
        filt = rust_compile_filter(query)

        for _ in range(3):
            n = data.draw(st.integers(min_value=1, max_value=100))
            values = data.draw(st.lists(
                st.integers(min_value=0, max_value=1000), min_size=n, max_size=n
            ))
            df = pd.DataFrame({"dte": values})
            py_count = len(df[df.eval(query)])
            rust_result = filt.apply(pl.from_pandas(df))
            assert rust_result.height == py_count


# ---------------------------------------------------------------------------
# Full engine fuzzing -- randomize allocations and capital
# ---------------------------------------------------------------------------

class TestEngineAllocationFuzz:
    """Fuzz engine with varying allocations. Rust and Python must agree on
    trade log when both use FirstMatch + NoCosts (Rust-eligible config)."""

    @staticmethod
    def _setup_data():
        import os
        from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
        TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
        s = TiingoData(os.path.join(TEST_DIR, "ivy_5assets_data.csv"))
        s._data["adjClose"] = 10
        o = HistoricalOptionsData(os.path.join(TEST_DIR, "options_data.csv"))
        o._data.at[2, "ask"] = 1; o._data.at[2, "bid"] = 0.5
        o._data.at[51, "ask"] = 1.5; o._data.at[50, "bid"] = 0.5
        o._data.at[130, "bid"] = 0.5; o._data.at[131, "bid"] = 1.5
        o._data.at[206, "bid"] = 0.5; o._data.at[207, "bid"] = 1.5
        return s, o

    @given(
        stocks_pct=st.floats(min_value=0.1, max_value=0.95, allow_nan=False),
        options_pct=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
        capital=st.integers(min_value=10_000, max_value=10_000_000),
    )
    @settings(max_examples=20, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_rust_matches_python(self, stocks_pct, options_pct, capital):
        from options_portfolio_backtester.engine.engine import BacktestEngine
        from options_portfolio_backtester.execution.cost_model import NoCosts
        from options_portfolio_backtester.strategy.strategy import Strategy
        from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
        from options_portfolio_backtester.core.types import Stock, OptionType as Type, Direction
        import options_portfolio_backtester.engine._dispatch as _dispatch

        assume(stocks_pct + options_pct <= 1.0)
        cash_pct = 1.0 - stocks_pct - options_pct
        alloc = {"stocks": stocks_pct, "options": options_pct, "cash": cash_pct}

        stocks = [Stock("VTI", 0.2), Stock("VEU", 0.2), Stock("BND", 0.2),
                  Stock("VNQ", 0.2), Stock("DBC", 0.2)]

        # Python path
        s1, o1 = self._setup_data()
        eng1 = BacktestEngine(alloc, initial_capital=capital, cost_model=NoCosts())
        eng1.stocks = stocks; eng1.stocks_data = s1; eng1.options_data = o1
        schema1 = o1.schema
        strat1 = Strategy(schema1)
        leg1 = StrategyLeg("leg_1", schema1, option_type=Type.PUT, direction=Direction.BUY)
        leg1.entry_filter = (schema1.underlying == "SPX") & (schema1.dte >= 60)
        leg1.exit_filter = schema1.dte <= 30
        strat1.add_legs([leg1])
        eng1.options_strategy = strat1
        orig = _dispatch.RUST_AVAILABLE
        try:
            _dispatch.RUST_AVAILABLE = False
            eng1.run(rebalance_freq=1)
        finally:
            _dispatch.RUST_AVAILABLE = orig

        # Rust path
        s2, o2 = self._setup_data()
        eng2 = BacktestEngine(alloc, initial_capital=capital, cost_model=NoCosts())
        eng2.stocks = stocks; eng2.stocks_data = s2; eng2.options_data = o2
        strat2 = Strategy(o2.schema)
        leg2 = StrategyLeg("leg_1", o2.schema, option_type=Type.PUT, direction=Direction.BUY)
        leg2.entry_filter = (o2.schema.underlying == "SPX") & (o2.schema.dte >= 60)
        leg2.exit_filter = o2.schema.dte <= 30
        strat2.add_legs([leg2])
        eng2.options_strategy = strat2
        eng2.run(rebalance_freq=1)

        # Compare trade log
        assert eng1.trade_log.shape == eng2.trade_log.shape, \
            f"shape: py={eng1.trade_log.shape} rs={eng2.trade_log.shape} alloc={alloc} cap={capital}"

        if not eng1.trade_log.empty:
            py_costs = eng1.trade_log["totals"]["cost"].values
            rs_costs = eng2.trade_log["totals"]["cost"].values
            assert np.allclose(py_costs, rs_costs, rtol=1e-4), \
                f"costs: py={py_costs} rs={rs_costs}"

            py_qtys = eng1.trade_log["totals"]["qty"].values
            rs_qtys = eng2.trade_log["totals"]["qty"].values
            assert np.allclose(py_qtys, rs_qtys, rtol=1e-4), \
                f"qtys: py={py_qtys} rs={rs_qtys}"

        # Compare final capital
        py_final = eng1.balance["total capital"].iloc[-1]
        rs_final = eng2.balance["total capital"].iloc[-1]
        assert abs(py_final - rs_final) < 1.0, \
            f"final_capital: py={py_final} rs={rs_final} alloc={alloc} cap={capital}"
