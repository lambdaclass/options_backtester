from __future__ import annotations

import warnings

import pandas as pd
import pytest

from options_backtester.engine.algo_adapters import (
    BudgetPercent,
    EngineRunMonthly,
    EnginePipelineContext,
    EngineStepDecision,
    ExitOnThreshold,
    MaxGreekExposure,
    RangeFilter,
    SelectByDTE,
    SelectByDelta,
    IVRankFilter,
)
from options_backtester.engine.engine import BacktestEngine
from options_backtester.core.types import Greeks

from tests.engine.test_engine import _buy_strategy, _ivy_stocks, _options_data, _stocks_data


def _run_with_algos(algos):
    stocks = _ivy_stocks()
    stocks_data = _stocks_data()
    options_data = _options_data()
    schema = options_data.schema

    engine = BacktestEngine(
        {"stocks": 0.97, "options": 0.03, "cash": 0.0},
        algos=algos,
    )
    engine.stocks = stocks
    engine.stocks_data = stocks_data
    engine.options_data = options_data
    engine.options_strategy = _buy_strategy(schema)
    engine.run(rebalance_freq=1)
    return engine


def _dummy_ctx(**overrides) -> EnginePipelineContext:
    defaults = dict(
        date=pd.Timestamp("2024-01-02"),
        stocks=pd.DataFrame(),
        options=pd.DataFrame(),
        total_capital=100_000.0,
        current_cash=50_000.0,
        current_greeks=Greeks(delta=0.5, gamma=0.01, theta=-0.02, vega=0.1),
        options_allocation=3000.0,
    )
    defaults.update(overrides)
    return EnginePipelineContext(**defaults)


# ---------------------------------------------------------------------------
# EngineRunMonthly
# ---------------------------------------------------------------------------

def test_engine_algo_monthly_gate_logs_skip():
    engine = _run_with_algos([EngineRunMonthly()])
    logs = engine.events_dataframe()
    assert not logs.empty
    assert (logs["event"] == "algo_step").any()
    assert (logs["status"] == "skip_day").any()


def test_engine_run_monthly_reset():
    algo = EngineRunMonthly()
    ctx_jan = _dummy_ctx(date=pd.Timestamp("2024-01-02"))
    ctx_jan2 = _dummy_ctx(date=pd.Timestamp("2024-01-15"))
    assert algo(ctx_jan).status == "continue"
    assert algo(ctx_jan2).status == "skip_day"
    algo.reset()
    assert algo(ctx_jan).status == "continue"


# ---------------------------------------------------------------------------
# BudgetPercent
# ---------------------------------------------------------------------------

def test_budget_percent_zero_blocks_option_entries():
    engine = _run_with_algos([BudgetPercent(0.0)])
    # With 0% budget, options allocation is zero — no options should be bought
    assert engine.trade_log.empty or (engine.trade_log["totals"]["qty"] <= 0).all()


def test_budget_percent_sets_allocation():
    algo = BudgetPercent(0.05)
    ctx = _dummy_ctx(total_capital=200_000.0)
    algo(ctx)
    assert ctx.options_allocation == 10_000.0


def test_budget_percent_clamps_negative_capital():
    algo = BudgetPercent(0.05)
    ctx = _dummy_ctx(total_capital=-100.0)
    algo(ctx)
    assert ctx.options_allocation == 0.0


# ---------------------------------------------------------------------------
# RangeFilter (item 8 dedup)
# ---------------------------------------------------------------------------

def test_range_filter_appends_entry_filter():
    flt = RangeFilter(column="delta", min_val=-0.3, max_val=-0.1)
    ctx = _dummy_ctx()
    result = flt(ctx)
    assert result.status == "continue"
    assert len(ctx.entry_filters) == 1

    df = pd.DataFrame({"delta": [-0.5, -0.2, -0.1, 0.0, 0.3]})
    mask = ctx.entry_filters[0](df)
    assert mask.tolist() == [False, True, True, False, False]


def test_range_filter_missing_column_passes_all():
    flt = RangeFilter(column="nonexistent", min_val=0, max_val=1)
    ctx = _dummy_ctx()
    flt(ctx)
    df = pd.DataFrame({"other": [1, 2, 3]})
    mask = ctx.entry_filters[0](df)
    assert mask.all()


# ---------------------------------------------------------------------------
# SelectByDelta / SelectByDTE / IVRankFilter (backward-compat aliases)
# ---------------------------------------------------------------------------

def test_select_by_delta_returns_range_filter():
    flt = SelectByDelta(min_delta=-0.5, max_delta=-0.1)
    assert isinstance(flt, RangeFilter)
    assert flt.column == "delta"


def test_select_by_dte_returns_range_filter():
    flt = SelectByDTE(min_dte=30, max_dte=60)
    assert isinstance(flt, RangeFilter)
    assert flt.column == "dte"
    assert flt.min_val == 30.0
    assert flt.max_val == 60.0


def test_iv_rank_filter_returns_range_filter():
    flt = IVRankFilter(min_rank=0.3, max_rank=0.8, column="iv_rank")
    assert isinstance(flt, RangeFilter)
    assert flt.column == "iv_rank"


def test_select_by_dte_strict_filter_skips_candidates():
    engine = _run_with_algos([SelectByDTE(min_dte=0, max_dte=1)])
    events = engine.events_dataframe()
    assert isinstance(events, pd.DataFrame)
    assert ((events["event"] == "option_entry_no_candidates") | (events["event"] == "option_entry_filtered")).any()


# ---------------------------------------------------------------------------
# MaxGreekExposure
# ---------------------------------------------------------------------------

def test_max_greek_exposure_delta_blocks():
    algo = MaxGreekExposure(max_abs_delta=0.3)
    ctx = _dummy_ctx(current_greeks=Greeks(delta=0.5, gamma=0, theta=0, vega=0))
    result = algo(ctx)
    assert result.status == "skip_day"
    assert "delta" in result.message


def test_max_greek_exposure_vega_blocks():
    algo = MaxGreekExposure(max_abs_vega=0.05)
    ctx = _dummy_ctx(current_greeks=Greeks(delta=0, gamma=0, theta=0, vega=0.1))
    result = algo(ctx)
    assert result.status == "skip_day"
    assert "vega" in result.message


def test_max_greek_exposure_within_limits_continues():
    algo = MaxGreekExposure(max_abs_delta=1.0, max_abs_vega=1.0)
    ctx = _dummy_ctx(current_greeks=Greeks(delta=0.1, gamma=0, theta=0, vega=0.05))
    result = algo(ctx)
    assert result.status == "continue"


def test_max_greek_exposure_none_limits_pass():
    algo = MaxGreekExposure()
    ctx = _dummy_ctx(current_greeks=Greeks(delta=999, gamma=0, theta=0, vega=999))
    result = algo(ctx)
    assert result.status == "continue"


# ---------------------------------------------------------------------------
# ExitOnThreshold (item 17)
# ---------------------------------------------------------------------------

def test_exit_on_threshold_sets_override():
    algo = ExitOnThreshold(profit_pct=0.5, loss_pct=0.3)
    ctx = _dummy_ctx()
    algo(ctx)
    assert ctx.exit_threshold_override == (0.5, 0.3)


def test_exit_on_threshold_warns_on_all_inf():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ExitOnThreshold()
        assert len(w) == 1
        assert "no effect" in str(w[0].message).lower()


def test_exit_on_threshold_no_warn_when_finite():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ExitOnThreshold(profit_pct=0.5)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# Events dataframe structure (item 14 + item 18 flattened data)
# ---------------------------------------------------------------------------

def test_events_dataframe_has_flattened_columns():
    engine = _run_with_algos([])
    events = engine.events_dataframe()
    assert "date" in events.columns
    assert "event" in events.columns
    assert "status" in events.columns
    # "data" column should NOT exist (flattened into top-level)
    assert "data" not in events.columns


def test_events_dataframe_contains_cash_from_rebalance_start():
    engine = _run_with_algos([])
    events = engine.events_dataframe()
    rebal_starts = events[events["event"] == "rebalance_start"]
    if not rebal_starts.empty:
        assert "cash" in rebal_starts.columns
        assert pd.notna(rebal_starts["cash"].iloc[0])


def test_events_dataframe_empty_when_no_events():
    from options_backtester.engine.engine import BacktestEngine
    engine = BacktestEngine({"stocks": 0.97, "options": 0.03, "cash": 0.0})
    events = engine.events_dataframe()
    assert events.empty
    assert "date" in events.columns
    assert "event" in events.columns
    assert "status" in events.columns
