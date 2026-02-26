"""Shared helpers for Rust-vs-Python parity tests."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

try:
    from options_portfolio_backtester.engine._dispatch import use_rust
    RUST_AVAILABLE = use_rust()
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)

_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Stock list ──────────────────────────────────────────────────────────

IVY_STOCKS_TUPLES = [
    ("VTI", 0.2), ("VEU", 0.2), ("BND", 0.2), ("VNQ", 0.2), ("DBC", 0.2),
]


def ivy_stocks():
    """Return standard 5-asset IVY stock list as Stock objects."""
    from options_portfolio_backtester.core.types import Stock
    return [Stock(sym, pct) for sym, pct in IVY_STOCKS_TUPLES]


# ── Data loaders ────────────────────────────────────────────────────────

def load_small_stocks():
    """Load ivy_5assets_data.csv with adjClose fixed to 10."""
    from options_portfolio_backtester.data.providers import TiingoData
    s = TiingoData(os.path.join(_TEST_DIR, "ivy_5assets_data.csv"))
    s._data["adjClose"] = 10
    return s


def load_small_options():
    """Load options_data.csv with bid/ask fixes for reliable entries."""
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    o = HistoricalOptionsData(
        os.path.join(_TEST_DIR, "options_data.csv")
    )
    o._data.at[2, "ask"] = 1
    o._data.at[2, "bid"] = 0.5
    o._data.at[51, "ask"] = 1.5
    o._data.at[50, "bid"] = 0.5
    o._data.at[130, "bid"] = 0.5
    o._data.at[131, "bid"] = 1.5
    o._data.at[206, "bid"] = 0.5
    o._data.at[207, "bid"] = 1.5
    return o


def load_large_stocks():
    """Load test_data_stocks.csv."""
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(
        os.path.join(_TEST_DIR, "test_data_stocks.csv")
    )


def load_large_options():
    """Load test_data_options.csv."""
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(
        os.path.join(_TEST_DIR, "test_data_options.csv")
    )


# ── Strategy builders ──────────────────────────────────────────────────

def buy_put_strategy(schema, underlying="SPX", dte_min=60, dte_max=None,
                     dte_exit=30):
    """Build a single-leg BUY PUT strategy."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                      direction=Direction.BUY)
    filt = (schema.underlying == underlying) & (schema.dte >= dte_min)
    if dte_max is not None:
        filt = filt & (schema.dte <= dte_max)
    leg.entry_filter = filt
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def buy_call_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a single-leg BUY CALL strategy."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.CALL,
                      direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == underlying) & (schema.dte >= dte_min)
    )
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def sell_put_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a single-leg SELL PUT strategy."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                      direction=Direction.SELL)
    leg.entry_filter = (
        (schema.underlying == underlying) & (schema.dte >= dte_min)
    )
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def sell_call_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a single-leg SELL CALL strategy."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.CALL,
                      direction=Direction.SELL)
    leg.entry_filter = (
        (schema.underlying == underlying) & (schema.dte >= dte_min)
    )
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def buy_put_spread_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a 2-leg BUY PUT SPREAD (debit spread): buy high-strike put, sell low-strike put."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    # Long put (higher strike)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg1.exit_filter = schema.dte <= dte_exit
    # Short put (lower strike)
    leg2 = StrategyLeg("leg_2", schema, option_type=Type.PUT, direction=Direction.SELL)
    leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg2.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg1, leg2])
    return strat


def sell_call_spread_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a 2-leg SELL CALL SPREAD (credit spread): sell low-strike call, buy high-strike call."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.SELL)
    leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg1.exit_filter = schema.dte <= dte_exit
    leg2 = StrategyLeg("leg_2", schema, option_type=Type.CALL, direction=Direction.BUY)
    leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg2.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg1, leg2])
    return strat


def strangle_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a 2-leg SELL STRANGLE: sell put + sell call."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.SELL)
    leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg1.exit_filter = schema.dte <= dte_exit
    leg2 = StrategyLeg("leg_2", schema, option_type=Type.CALL, direction=Direction.SELL)
    leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg2.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg1, leg2])
    return strat


def straddle_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    """Build a 2-leg BUY STRADDLE: buy put + buy call."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg1.exit_filter = schema.dte <= dte_exit
    leg2 = StrategyLeg("leg_2", schema, option_type=Type.CALL, direction=Direction.BUY)
    leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg2.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg1, leg2])
    return strat


def two_leg_strategy(schema, dir1, type1, dir2, type2,
                     underlying="SPX", dte_min=60, dte_exit=30):
    """Build a 2-leg strategy with specified direction/type combos."""
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    type_map = {"put": Type.PUT, "call": Type.CALL}
    dir_map = {"buy": Direction.BUY, "sell": Direction.SELL}

    strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema,
                       option_type=type_map[type1], direction=dir_map[dir1])
    leg1.entry_filter = (
        (schema.underlying == underlying) & (schema.dte >= dte_min)
    )
    leg1.exit_filter = schema.dte <= dte_exit

    leg2 = StrategyLeg("leg_2", schema,
                       option_type=type_map[type2], direction=dir_map[dir2])
    leg2.entry_filter = (
        (schema.underlying == underlying) & (schema.dte >= dte_min)
    )
    leg2.exit_filter = schema.dte <= dte_exit

    strat.add_legs([leg1, leg2])
    return strat


# ── Engine runners ──────────────────────────────────────────────────────

def _make_engine(alloc, capital, stocks, stocks_data, options_data,
                 strategy_fn, **engine_kwargs):
    """Create a BacktestEngine with given config."""
    from options_portfolio_backtester.engine.engine import BacktestEngine
    from options_portfolio_backtester.execution.cost_model import NoCosts
    from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk
    from options_portfolio_backtester.execution.signal_selector import FirstMatch

    engine_kwargs.setdefault("cost_model", NoCosts())
    engine_kwargs.setdefault("fill_model", MarketAtBidAsk())
    engine_kwargs.setdefault("signal_selector", FirstMatch())

    eng = BacktestEngine(alloc, initial_capital=capital, **engine_kwargs)
    eng.stocks = stocks
    eng.stocks_data = stocks_data
    eng.options_data = options_data
    eng.options_strategy = strategy_fn(options_data.schema)
    return eng


def run_rust(alloc, capital, strategy_fn, rebalance_freq=1,
             stocks=None, stocks_data=None, options_data=None,
             sma_days=None, require_rust=True, **engine_kwargs):
    """Run backtest via Rust path. Returns engine.

    If require_rust=True (default), asserts that the engine actually
    dispatched to Rust. Set require_rust=False to allow silent fallback
    (the test can then check dispatch_mode itself).
    """
    s = stocks_data or load_small_stocks()
    o = options_data or load_small_options()
    stks = stocks or ivy_stocks()

    eng = _make_engine(alloc, capital, stks, s, o, strategy_fn, **engine_kwargs)
    eng.run(rebalance_freq=rebalance_freq, sma_days=sma_days)

    if require_rust:
        assert eng.run_metadata.get("dispatch_mode") == "rust-full", (
            f"Expected rust-full dispatch but got: "
            f"{eng.run_metadata.get('dispatch_mode')}"
        )
    return eng


def run_python(alloc, capital, strategy_fn, rebalance_freq=1,
               stocks=None, stocks_data=None, options_data=None,
               sma_days=None, **engine_kwargs):
    """Run backtest forcing Python path by temporarily disabling Rust.

    Monkeypatches _dispatch.RUST_AVAILABLE to False so the dispatch gate
    falls through to the Python code path without changing any allocation logic.
    """
    import options_portfolio_backtester.engine._dispatch as _dispatch

    s = stocks_data or load_small_stocks()
    o = options_data or load_small_options()
    stks = stocks or ivy_stocks()

    eng = _make_engine(alloc, capital, stks, s, o, strategy_fn, **engine_kwargs)

    orig = _dispatch.RUST_AVAILABLE
    try:
        _dispatch.RUST_AVAILABLE = False
        eng.run(rebalance_freq=rebalance_freq, sma_days=sma_days)
    finally:
        _dispatch.RUST_AVAILABLE = orig

    assert eng.run_metadata.get("dispatch_mode") == "python", (
        f"Expected python dispatch but got: {eng.run_metadata.get('dispatch_mode')}"
    )
    return eng


# ── Comparison assertions ───────────────────────────────────────────────

def assert_parity(py_eng, rs_eng, rtol=2e-3, atol=1.0, label=""):
    """Assert trade log and balance match between Python and Rust runs.

    rtol=0.2%: commission rounding cascades across many re-entries with
    floor()-based qty computation.  A ±1 qty difference on a $100 contract
    shifts cash by $100, which then cascades through subsequent allocations.

    Checks: trade log shape, costs, quantities, row-level content,
    final capital, AND full balance timeseries.
    """
    prefix = f"[{label}] " if label else ""

    # Trade log shape
    assert py_eng.trade_log.shape == rs_eng.trade_log.shape, (
        f"{prefix}trade_log shape: py={py_eng.trade_log.shape} "
        f"rs={rs_eng.trade_log.shape}"
    )

    # Trade log costs and quantities
    if not py_eng.trade_log.empty:
        py_costs = py_eng.trade_log["totals"]["cost"].values
        rs_costs = rs_eng.trade_log["totals"]["cost"].values
        assert np.allclose(py_costs, rs_costs, rtol=rtol), (
            f"{prefix}costs: py={py_costs} rs={rs_costs}"
        )

        py_qtys = py_eng.trade_log["totals"]["qty"].values
        rs_qtys = rs_eng.trade_log["totals"]["qty"].values
        # Quantities use floor() division, so tiny cash differences from
        # commission rounding can cascade across multiple re-entries.
        # Allow 1% relative OR ±5 absolute (covers cascading ±1-3 per trade).
        assert np.allclose(py_qtys, rs_qtys, rtol=0.01, atol=5.0), (
            f"{prefix}qtys: py={py_qtys} rs={rs_qtys}"
        )

        # Row-level trade log content verification
        _assert_trade_log_content(py_eng, rs_eng, prefix)

    # Final capital — use both absolute and relative tolerance.
    # Commission cascading can produce ~$50 diffs on $300K-$1M portfolios.
    py_final = py_eng.balance["total capital"].iloc[-1]
    rs_final = rs_eng.balance["total capital"].iloc[-1]
    abs_diff = abs(py_final - rs_final)
    rel_diff = abs_diff / max(abs(py_final), 1.0)
    assert abs_diff < atol or rel_diff < rtol, (
        f"{prefix}final_capital: py={py_final} rs={rs_final} "
        f"(abs_diff={abs_diff:.2f}, rel_diff={rel_diff:.6f})"
    )

    # Full balance timeseries comparison
    assert_balance_close(py_eng, rs_eng, rtol=rtol * 10, label=label)


def _assert_trade_log_content(py_eng, rs_eng, prefix=""):
    """Verify trade log row-level content matches: contracts, orders, strikes, dates."""
    py_tl = py_eng.trade_log
    rs_tl = rs_eng.trade_log

    # Get leg names from columns (top-level MultiIndex excluding 'totals')
    py_legs = [c for c in py_tl.columns.get_level_values(0).unique() if c != "totals"]
    rs_legs = [c for c in rs_tl.columns.get_level_values(0).unique() if c != "totals"]
    assert py_legs == rs_legs, (
        f"{prefix}trade_log leg names differ: py={py_legs} rs={rs_legs}"
    )

    for leg in py_legs:
        # Contract IDs
        py_contracts = py_tl[leg]["contract"].astype(str).values
        rs_contracts = rs_tl[leg]["contract"].astype(str).values
        assert np.array_equal(py_contracts, rs_contracts), (
            f"{prefix}trade_log {leg} contracts differ:\n"
            f"  py={py_contracts[:5]}\n  rs={rs_contracts[:5]}"
        )

        # Order types (BTO/STC/STO/BTC)
        # Python stores as enum (Order.BTO), Rust as plain string (BTO)
        def _normalize_order(val):
            s = str(val)
            for suffix in ("BTO", "STC", "STO", "BTC"):
                if s.endswith(suffix):
                    return suffix
            return s
        py_orders = np.array([_normalize_order(v) for v in py_tl[leg]["order"].values])
        rs_orders = np.array([_normalize_order(v) for v in rs_tl[leg]["order"].values])
        assert np.array_equal(py_orders, rs_orders), (
            f"{prefix}trade_log {leg} orders differ:\n"
            f"  py={py_orders[:5]}\n  rs={rs_orders[:5]}"
        )

        # Strikes
        py_strikes = py_tl[leg]["strike"].astype(float).values
        rs_strikes = rs_tl[leg]["strike"].astype(float).values
        assert np.allclose(py_strikes, rs_strikes, rtol=1e-6), (
            f"{prefix}trade_log {leg} strikes differ:\n"
            f"  py={py_strikes[:5]}\n  rs={rs_strikes[:5]}"
        )

        # Per-leg costs
        py_leg_costs = py_tl[leg]["cost"].astype(float).values
        rs_leg_costs = rs_tl[leg]["cost"].astype(float).values
        assert np.allclose(py_leg_costs, rs_leg_costs, rtol=1e-4), (
            f"{prefix}trade_log {leg} per-leg costs differ:\n"
            f"  py={py_leg_costs[:5]}\n  rs={rs_leg_costs[:5]}"
        )

    # Dates
    py_dates = pd.to_datetime(py_tl["totals"]["date"]).values
    rs_dates = pd.to_datetime(rs_tl["totals"]["date"]).values
    assert np.array_equal(py_dates, rs_dates), (
        f"{prefix}trade_log dates differ:\n"
        f"  py={py_dates[:5]}\n  rs={rs_dates[:5]}"
    )


def assert_balance_close(py_eng, rs_eng, rtol=1e-3, label=""):
    """Assert full balance timeseries matches (not just final value)."""
    prefix = f"[{label}] " if label else ""

    py_tc = py_eng.balance["total capital"].values
    rs_tc = rs_eng.balance["total capital"].values

    assert len(py_tc) == len(rs_tc), (
        f"{prefix}balance length: py={len(py_tc)} rs={len(rs_tc)}"
    )
    assert np.allclose(py_tc, rs_tc, rtol=rtol, atol=1.0), (
        f"{prefix}balance timeseries diverged.\n"
        f"Max diff: {np.max(np.abs(py_tc - rs_tc))}"
    )


# ── Default allocation ──────────────────────────────────────────────────

DEFAULT_ALLOC = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
DEFAULT_CAPITAL = 1_000_000


# ── Stock lists for generated / production data ───────────────────────

GENERATED_STOCKS_TUPLES = [
    ("VOO", 0.20), ("TLT", 0.20), ("EWY", 0.15),
    ("PDBC", 0.15), ("IAU", 0.10), ("VNQI", 0.10), ("VTIP", 0.10),
]

PROD_SPY_STOCKS_TUPLES = [("SPY", 1.0)]


def generated_stocks():
    """Return 7-asset stock list for the generated synthetic dataset."""
    from options_portfolio_backtester.core.types import Stock
    return [Stock(sym, pct) for sym, pct in GENERATED_STOCKS_TUPLES]


def prod_spy_stocks():
    """Return single SPY stock for the production dataset."""
    from options_portfolio_backtester.core.types import Stock
    return [Stock(sym, pct) for sym, pct in PROD_SPY_STOCKS_TUPLES]


# ── Data loaders for generated / production data ─────────────────────

def load_generated_stocks():
    """Load synthetic large_stocks.csv (500 days, 7 stocks)."""
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_DATA_DIR, "large_stocks.csv"))


def load_generated_options():
    """Load synthetic large_options.csv (500 days, ~38 contracts/date)."""
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(os.path.join(_DATA_DIR, "large_options.csv"))


def load_prod_stocks():
    """Load production SPY stocks (1 year, 252 days)."""
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_DATA_DIR, "prod_stocks_1y.csv"))


def load_prod_options():
    """Load production SPY options (1 year, ~5K rows, ~20 contracts/date)."""
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(os.path.join(_DATA_DIR, "prod_options_1y.csv"))


# ── Extended engine runners with rebalance_unit ──────────────────────

def run_rust_ex(alloc, capital, strategy_fn, rebalance_freq=1,
                stocks=None, stocks_data=None, options_data=None,
                sma_days=None, rebalance_unit='BMS', require_rust=True,
                **engine_kwargs):
    """Run backtest via Rust path with rebalance_unit support."""
    s = stocks_data or load_small_stocks()
    o = options_data or load_small_options()
    stks = stocks or ivy_stocks()

    eng = _make_engine(alloc, capital, stks, s, o, strategy_fn, **engine_kwargs)
    eng.run(rebalance_freq=rebalance_freq, sma_days=sma_days,
            rebalance_unit=rebalance_unit)

    if require_rust:
        assert eng.run_metadata.get("dispatch_mode") == "rust-full", (
            f"Expected rust-full dispatch but got: "
            f"{eng.run_metadata.get('dispatch_mode')}"
        )
    return eng


def run_python_ex(alloc, capital, strategy_fn, rebalance_freq=1,
                  stocks=None, stocks_data=None, options_data=None,
                  sma_days=None, rebalance_unit='BMS', **engine_kwargs):
    """Run backtest forcing Python path, with rebalance_unit support."""
    import options_portfolio_backtester.engine._dispatch as _dispatch

    s = stocks_data or load_small_stocks()
    o = options_data or load_small_options()
    stks = stocks or ivy_stocks()

    eng = _make_engine(alloc, capital, stks, s, o, strategy_fn, **engine_kwargs)

    orig = _dispatch.RUST_AVAILABLE
    try:
        _dispatch.RUST_AVAILABLE = False
        eng.run(rebalance_freq=rebalance_freq, sma_days=sma_days,
                rebalance_unit=rebalance_unit)
    finally:
        _dispatch.RUST_AVAILABLE = orig

    assert eng.run_metadata.get("dispatch_mode") == "python", (
        f"Expected python dispatch but got: {eng.run_metadata.get('dispatch_mode')}"
    )
    return eng


# ── Multi-dataset slice loaders ────────────────────────────────────────

PROD_SLICES = {
    "spy_crisis": {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "spy_lowvol": {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "spy_covid":  {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "spy_bear":   {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "iwm_2020":   {"stocks_tuples": [("IWM", 1.0)], "underlying": "IWM"},
    "qqq_2020":   {"stocks_tuples": [("QQQ", 1.0)], "underlying": "QQQ"},
}

STRATEGY_MAP = {
    "buy_put": buy_put_strategy,
    "buy_call": buy_call_strategy,
    "sell_put": sell_put_strategy,
    "sell_call": sell_call_strategy,
    "buy_put_spread": buy_put_spread_strategy,
    "sell_call_spread": sell_call_spread_strategy,
    "strangle": strangle_strategy,
    "straddle": straddle_strategy,
}


def make_cost_model(name):
    """Factory for cost models by name."""
    from options_portfolio_backtester.execution.cost_model import (
        NoCosts, PerContractCommission, TieredCommission,
    )
    if name == "NoCosts":
        return NoCosts()
    elif name == "PerContract":
        return PerContractCommission(rate=0.65)
    elif name == "Tiered":
        return TieredCommission(tiers=[(10_000, 0.65), (100_000, 0.50)])
    raise ValueError(name)


def make_fill_model(name):
    """Factory for fill models by name."""
    from options_portfolio_backtester.execution.fill_model import (
        MarketAtBidAsk, MidPrice, VolumeAwareFill,
    )
    if name == "MarketAtBidAsk":
        return MarketAtBidAsk()
    elif name == "MidPrice":
        return MidPrice()
    elif name == "VolumeAware":
        return VolumeAwareFill(full_volume_threshold=100)
    raise ValueError(name)


def make_signal_selector(name):
    """Factory for signal selectors by name."""
    from options_portfolio_backtester.execution.signal_selector import (
        FirstMatch, NearestDelta, MaxOpenInterest,
    )
    if name == "FirstMatch":
        return FirstMatch()
    elif name == "NearestDelta":
        return NearestDelta(target_delta=-0.30)
    elif name == "MaxOpenInterest":
        return MaxOpenInterest()
    raise ValueError(name)


def _slice_data_exists(slice_id):
    """Check if CSV files for a slice have been generated."""
    return (
        os.path.isfile(os.path.join(_DATA_DIR, f"{slice_id}_stocks.csv"))
        and os.path.isfile(os.path.join(_DATA_DIR, f"{slice_id}_options.csv"))
    )


def load_slice_stocks(slice_id):
    """Load stocks CSV for a production slice → TiingoData."""
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_DATA_DIR, f"{slice_id}_stocks.csv"))


def load_slice_options(slice_id):
    """Load options CSV for a production slice → HistoricalOptionsData."""
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(
        os.path.join(_DATA_DIR, f"{slice_id}_options.csv")
    )


def slice_stocks(slice_id):
    """Return Stock objects for a production slice."""
    from options_portfolio_backtester.core.types import Stock
    tuples = PROD_SLICES[slice_id]["stocks_tuples"]
    return [Stock(sym, pct) for sym, pct in tuples]


def assert_per_rebalance_parity(py_eng, rs_eng, rtol=1e-4, atol=0.01, label=""):
    """Assert Python and Rust balance match at EVERY rebalance date.

    Catches compensating errors where final capital matches but
    intermediate values diverge.
    """
    prefix = f"[{label}] " if label else ""

    py_bal = py_eng.balance
    rs_bal = rs_eng.balance

    assert len(py_bal) == len(rs_bal), (
        f"{prefix}balance length: py={len(py_bal)} rs={len(rs_bal)}"
    )

    py_tc = py_bal["total capital"].values
    rs_tc = rs_bal["total capital"].values

    for i in range(len(py_tc)):
        diff = abs(py_tc[i] - rs_tc[i])
        rel = diff / max(abs(py_tc[i]), 1.0)
        assert diff < atol or rel < rtol, (
            f"{prefix}balance diverged at row {i}: "
            f"py={py_tc[i]:.4f} rs={rs_tc[i]:.4f} diff={diff:.4f}"
        )

    # Also check cash column
    py_cash = py_bal["cash"].values
    rs_cash = rs_bal["cash"].values
    assert len(py_cash) == len(rs_cash)
    assert np.allclose(py_cash, rs_cash, rtol=rtol, atol=atol), (
        f"{prefix}cash diverged. Max diff: {np.max(np.abs(py_cash - rs_cash))}"
    )
