"""Shared helpers for bench regression tests."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

try:
    from options_portfolio_backtester._ob_rust import run_backtest_py  # noqa: F401
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_ALLOC = {"stocks": 0.6, "options": 0.3, "cash": 0.1}
DEFAULT_CAPITAL = 1_000_000

IVY_STOCKS_TUPLES = [
    ("VTI", 0.2), ("VEU", 0.2), ("BND", 0.2), ("VNQ", 0.2), ("DBC", 0.2),
]

GENERATED_STOCKS_TUPLES = [
    ("VOO", 0.20), ("TLT", 0.20), ("EWY", 0.15),
    ("PDBC", 0.15), ("IAU", 0.10), ("VNQI", 0.10), ("VTIP", 0.10),
]

PROD_SPY_STOCKS_TUPLES = [("SPY", 1.0)]

PROD_SLICES = {
    "spy_crisis": {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "spy_lowvol": {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "spy_covid":  {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "spy_bear":   {"stocks_tuples": [("SPY", 1.0)], "underlying": "SPY"},
    "iwm_2020":   {"stocks_tuples": [("IWM", 1.0)], "underlying": "IWM"},
    "qqq_2020":   {"stocks_tuples": [("QQQ", 1.0)], "underlying": "QQQ"},
}

STRATEGY_MAP: dict = {}  # populated after strategy builder definitions


# ── Stock lists ────────────────────────────────────────────────────────

def _stocks_from_tuples(tuples):
    from options_portfolio_backtester.core.types import Stock
    return [Stock(sym, pct) for sym, pct in tuples]


def ivy_stocks():
    return _stocks_from_tuples(IVY_STOCKS_TUPLES)


def generated_stocks():
    return _stocks_from_tuples(GENERATED_STOCKS_TUPLES)


def prod_spy_stocks():
    return _stocks_from_tuples(PROD_SPY_STOCKS_TUPLES)


def slice_stocks(slice_id):
    return _stocks_from_tuples(PROD_SLICES[slice_id]["stocks_tuples"])


# ── Data loaders ───────────────────────────────────────────────────────

def load_small_stocks():
    from options_portfolio_backtester.data.providers import TiingoData
    s = TiingoData(os.path.join(_TEST_DIR, "ivy_5assets_data.csv"))
    s._data["adjClose"] = 10
    return s


def load_small_options():
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    o = HistoricalOptionsData(os.path.join(_TEST_DIR, "options_data.csv"))
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
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_TEST_DIR, "test_data_stocks.csv"))


def load_large_options():
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(os.path.join(_TEST_DIR, "test_data_options.csv"))


def load_generated_stocks():
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_DATA_DIR, "large_stocks.csv"))


def load_generated_options():
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(os.path.join(_DATA_DIR, "large_options.csv"))


def load_prod_stocks():
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_DATA_DIR, "prod_stocks_1y.csv"))


def load_prod_options():
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(os.path.join(_DATA_DIR, "prod_options_1y.csv"))


def slice_data_exists(slice_id):
    return (
        os.path.isfile(os.path.join(_DATA_DIR, f"{slice_id}_stocks.csv"))
        and os.path.isfile(os.path.join(_DATA_DIR, f"{slice_id}_options.csv"))
    )


def load_slice_stocks(slice_id):
    from options_portfolio_backtester.data.providers import TiingoData
    return TiingoData(os.path.join(_DATA_DIR, f"{slice_id}_stocks.csv"))


def load_slice_options(slice_id):
    from options_portfolio_backtester.data.providers import HistoricalOptionsData
    return HistoricalOptionsData(
        os.path.join(_DATA_DIR, f"{slice_id}_options.csv")
    )


# ── Strategy builders ─────────────────────────────────────────────────

def buy_put_strategy(schema, underlying="SPX", dte_min=60, dte_max=None,
                     dte_exit=30):
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
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.CALL,
                      direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def sell_put_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT,
                      direction=Direction.SELL)
    leg.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def sell_call_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.CALL,
                      direction=Direction.SELL)
    leg.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    return strat


def strangle_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
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


def buy_put_spread_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg1.exit_filter = schema.dte <= dte_exit
    leg2 = StrategyLeg("leg_2", schema, option_type=Type.PUT, direction=Direction.SELL)
    leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg2.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg1, leg2])
    return strat


def sell_call_spread_strategy(schema, underlying="SPX", dte_min=60, dte_exit=30):
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


def two_leg_strategy(schema, dir1, type1, dir2, type2,
                     underlying="SPX", dte_min=60, dte_exit=30):
    from options_portfolio_backtester.strategy.strategy import Strategy
    from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
    from options_portfolio_backtester.core.types import OptionType as Type, Direction

    type_map = {"put": Type.PUT, "call": Type.CALL}
    dir_map = {"buy": Direction.BUY, "sell": Direction.SELL}

    strat = Strategy(schema)
    leg1 = StrategyLeg("leg_1", schema,
                       option_type=type_map[type1], direction=dir_map[dir1])
    leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg1.exit_filter = schema.dte <= dte_exit
    leg2 = StrategyLeg("leg_2", schema,
                       option_type=type_map[type2], direction=dir_map[dir2])
    leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_min)
    leg2.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg1, leg2])
    return strat


# Populate STRATEGY_MAP after definitions
STRATEGY_MAP.update({
    "buy_put": buy_put_strategy,
    "buy_call": buy_call_strategy,
    "sell_put": sell_put_strategy,
    "sell_call": sell_call_strategy,
    "buy_put_spread": buy_put_spread_strategy,
    "sell_call_spread": sell_call_spread_strategy,
    "strangle": strangle_strategy,
    "straddle": straddle_strategy,
})


# ── Execution model factories ─────────────────────────────────────────

def make_cost_model(name):
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


# ── Engine runner ──────────────────────────────────────────────────────

def _make_engine(alloc, capital, stocks, stocks_data, options_data,
                 strategy_fn, **engine_kwargs):
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


def run_backtest(alloc=None, capital=None, strategy_fn=None,
                 rebalance_freq=1, rebalance_unit='BMS',
                 stocks=None, stocks_data=None, options_data=None,
                 sma_days=None, **engine_kwargs):
    """Run a single backtest and return the engine."""
    s = stocks_data or load_small_stocks()
    o = options_data or load_small_options()
    stks = stocks or ivy_stocks()
    a = alloc or DEFAULT_ALLOC
    c = capital or DEFAULT_CAPITAL
    sf = strategy_fn or buy_put_strategy

    eng = _make_engine(a, c, stks, s, o, sf, **engine_kwargs)
    eng.run(rebalance_freq=rebalance_freq, sma_days=sma_days,
            rebalance_unit=rebalance_unit)
    return eng


# ── Invariant assertions ──────────────────────────────────────────────

def assert_invariants(eng, min_trades=0, label="", allow_negative_capital=False):
    """Assert standard invariants on a single backtest engine result."""
    prefix = f"[{label}] " if label else ""

    bal = eng.balance
    assert len(bal) > 0, f"{prefix}empty balance"

    tc = bal["total capital"]

    # Total capital never negative (sell strategies can go negative)
    if not allow_negative_capital:
        assert (tc >= -1.0).all(), f"{prefix}negative total capital: min={tc.min()}"

    # Balance dates monotonic
    assert bal.index.is_monotonic_increasing, f"{prefix}balance index not monotonic"

    # Cash column exists
    assert "cash" in bal.columns, f"{prefix}'cash' column missing"

    # Capital = sum of parts (skip first row — initial allocation)
    if "options capital" in bal.columns and "stocks capital" in bal.columns:
        reconstructed = bal["cash"] + bal["stocks capital"] + bal["options capital"]
        assert np.allclose(
            tc.values[1:], reconstructed.values[1:],
            rtol=1e-4, atol=1.0,
        ), f"{prefix}total capital != cash + stocks + options"

    # Trade log
    if min_trades > 0:
        assert len(eng.trade_log) >= min_trades, (
            f"{prefix}expected >= {min_trades} trades, got {len(eng.trade_log)}"
        )

    # Entry quantities positive
    if not eng.trade_log.empty:
        qtys = eng.trade_log["totals"]["qty"].values
        assert all(q > 0 for q in qtys), f"{prefix}found non-positive qty"

    # No negative stock quantities (sell strategies can cause negative via margin)
    if not allow_negative_capital:
        for col in bal.columns:
            if col.endswith(" qty"):
                vals = pd.to_numeric(bal[col], errors="coerce").dropna()
                assert (vals >= -0.01).all(), (
                    f"{prefix}negative qty in '{col}': min={vals.min()}"
                )
