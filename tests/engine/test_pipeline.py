from __future__ import annotations

import pandas as pd
import numpy as np

from options_backtester.engine.pipeline import (
    AlgoPipelineBacktester,
    CapitalFlow,
    CloseDead,
    ClosePositionsAfterDates,
    CouponPayingPosition,
    HedgeRisks,
    LimitDeltas,
    LimitWeights,
    Margin,
    MaxDrawdownGuard,
    Not,
    Or,
    PipelineContext,
    RandomBenchmarkResult,
    Rebalance,
    RebalanceOverTime,
    ReplayTransactions,
    Require,
    RunAfterDate,
    RunAfterDays,
    RunDaily,
    RunEveryNPeriods,
    RunIfOutOfBounds,
    RunMonthly,
    RunOnce,
    RunOnDate,
    RunQuarterly,
    RunWeekly,
    RunYearly,
    ScaleWeights,
    SelectActive,
    SelectAll,
    SelectHasData,
    SelectMomentum,
    SelectN,
    SelectRandomly,
    SelectRegex,
    SelectThese,
    SelectWhere,
    StepDecision,
    TargetVol,
    WeighERC,
    WeighEqually,
    WeighInvVol,
    WeighMeanVar,
    WeighRandomly,
    WeighSpecified,
    WeighTarget,
    benchmark_random,
)


def _prices() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    return pd.DataFrame({"SPY": [100.0, 102.0, 101.0, 103.0]}, index=idx)


# ---------------------------------------------------------------------------
# RunMonthly
# ---------------------------------------------------------------------------

def test_pipeline_rebalances_on_month_start_only():
    bt = AlgoPipelineBacktester(
        prices=_prices(),
        initial_capital=1000.0,
        algos=[RunMonthly(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bal = bt.run()

    assert "SPY qty" in bal.columns
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 10
    assert bal.loc[pd.Timestamp("2024-01-03"), "SPY qty"] == 10
    assert bal.loc[pd.Timestamp("2024-02-01"), "SPY qty"] == 10

    logs = bt.logs_dataframe()
    jan3 = logs[logs["date"] == pd.Timestamp("2024-01-03")]
    assert (jan3["status"] == "skip_day").any()


def test_run_monthly_reset_allows_rerun():
    """After reset(), the algo should not skip the first month on a second run."""
    algos = [RunMonthly(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()]
    bt = AlgoPipelineBacktester(prices=_prices(), initial_capital=1000.0, algos=algos)
    bal1 = bt.run()
    bal2 = bt.run()  # second run should reset state
    assert bal1.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == bal2.loc[pd.Timestamp("2024-01-02"), "SPY qty"]


# ---------------------------------------------------------------------------
# MaxDrawdownGuard
# ---------------------------------------------------------------------------

def test_drawdown_guard_blocks_rebalance():
    prices = pd.DataFrame(
        {"SPY": [100.0, 60.0, 55.0]},
        index=pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"]),
    )
    bt = AlgoPipelineBacktester(
        prices=prices,
        initial_capital=1000.0,
        algos=[
            RunMonthly(),
            SelectThese(["SPY"]),
            WeighSpecified({"SPY": 1.0}),
            MaxDrawdownGuard(max_drawdown_pct=0.20),
            Rebalance(),
        ],
    )
    bal = bt.run()
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 10
    assert bal.loc[pd.Timestamp("2024-02-01"), "SPY qty"] == 10
    logs = bt.logs_dataframe()
    feb = logs[(logs["date"] == pd.Timestamp("2024-02-01")) & (logs["step"] == "MaxDrawdownGuard")]
    assert not feb.empty
    assert feb.iloc[0]["status"] == "skip_day"


def test_drawdown_guard_reset():
    guard = MaxDrawdownGuard(max_drawdown_pct=0.10)
    ctx = PipelineContext(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 100.0}),
        total_capital=1000.0,
        cash=1000.0,
        positions={},
    )
    guard(ctx)  # sets _peak = 1000
    assert guard._peak == 1000.0
    guard.reset()
    assert guard._peak == 0.0


# ---------------------------------------------------------------------------
# Stop status (item 13)
# ---------------------------------------------------------------------------

class _StopAlgo:
    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if ctx.date >= pd.Timestamp("2024-02-01"):
            return StepDecision(status="stop", message="halt")
        return StepDecision()


def test_stop_algo_halts_pipeline_early():
    bt = AlgoPipelineBacktester(
        prices=_prices(),
        initial_capital=1000.0,
        algos=[_StopAlgo(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bal = bt.run()
    # Should stop at 2024-02-01 — only 3 rows (Jan 2, Jan 3, Feb 1)
    assert len(bal) == 3
    logs = bt.logs_dataframe()
    stop_rows = logs[logs["status"] == "stop"]
    assert len(stop_rows) == 1
    assert stop_rows.iloc[0]["date"] == pd.Timestamp("2024-02-01")


# ---------------------------------------------------------------------------
# SelectThese
# ---------------------------------------------------------------------------

def test_select_these_filters_missing_symbols():
    prices = pd.DataFrame(
        {"SPY": [100.0, 102.0], "TLT": [50.0, np.nan]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    bt = AlgoPipelineBacktester(
        prices=prices,
        initial_capital=1000.0,
        algos=[SelectThese(["SPY", "TLT"]), WeighSpecified({"SPY": 0.5, "TLT": 0.5}), Rebalance()],
    )
    bal = bt.run()
    # On Jan 3, TLT is NaN, so only SPY should be selected with normalized weight = 1.0
    assert bal.loc[pd.Timestamp("2024-01-03"), "SPY qty"] > 0


def test_select_these_case_insensitive():
    algo = SelectThese(["spy", "Tlt"])
    assert algo.symbols == ["SPY", "TLT"]


# ---------------------------------------------------------------------------
# WeighSpecified
# ---------------------------------------------------------------------------

def test_weigh_specified_normalizes():
    algo = WeighSpecified({"SPY": 2.0, "TLT": 1.0})
    ctx = PipelineContext(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        total_capital=1000.0,
        cash=1000.0,
        positions={},
        selected_symbols=["SPY", "TLT"],
    )
    algo(ctx)
    assert abs(ctx.target_weights["SPY"] - 2.0 / 3.0) < 1e-12
    assert abs(ctx.target_weights["TLT"] - 1.0 / 3.0) < 1e-12


def test_weigh_specified_skips_on_empty_selected():
    algo = WeighSpecified({"SPY": 1.0})
    ctx = PipelineContext(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 100.0}),
        total_capital=1000.0,
        cash=1000.0,
        positions={},
        selected_symbols=[],
    )
    decision = algo(ctx)
    assert decision.status == "skip_day"


# ---------------------------------------------------------------------------
# Rebalance
# ---------------------------------------------------------------------------

def test_rebalance_computes_floor_qty():
    ctx = PipelineContext(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 333.0}),
        total_capital=1000.0,
        cash=1000.0,
        positions={},
        target_weights={"SPY": 1.0},
    )
    Rebalance()(ctx)
    # floor(1000 / 333) = 3
    assert ctx.positions["SPY"] == 3.0
    assert ctx.cash == 1000.0 - 3 * 333.0


def test_rebalance_skips_zero_price():
    ctx = PipelineContext(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 0.0, "TLT": 50.0}),
        total_capital=1000.0,
        cash=1000.0,
        positions={},
        target_weights={"SPY": 0.5, "TLT": 0.5},
    )
    Rebalance()(ctx)
    assert "SPY" not in ctx.positions
    assert ctx.positions["TLT"] == 10.0


# ---------------------------------------------------------------------------
# Balance output structure
# ---------------------------------------------------------------------------

def test_balance_has_expected_columns():
    bt = AlgoPipelineBacktester(
        prices=_prices(),
        initial_capital=1000.0,
        algos=[RunMonthly(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bal = bt.run()
    for col in ["cash", "stocks capital", "total capital", "% change", "accumulated return"]:
        assert col in bal.columns, f"Missing column: {col}"


def test_logs_dataframe_schema():
    bt = AlgoPipelineBacktester(
        prices=_prices(),
        initial_capital=1000.0,
        algos=[RunMonthly(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bt.run()
    logs = bt.logs_dataframe()
    assert list(logs.columns) == ["date", "step", "status", "message"]
    assert set(logs["status"].unique()) <= {"continue", "skip_day", "stop"}


def test_empty_run_returns_empty_balance():
    prices = pd.DataFrame({"SPY": pd.Series(dtype=float)})
    bt = AlgoPipelineBacktester(prices=prices, initial_capital=1000.0, algos=[])
    bal = bt.run()
    assert bal.empty


# ---------------------------------------------------------------------------
# Multi-symbol
# ---------------------------------------------------------------------------

def test_multi_symbol_rebalance():
    idx = pd.to_datetime(["2024-01-02", "2024-02-01"])
    prices = pd.DataFrame({"SPY": [100.0, 110.0], "TLT": [50.0, 48.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices,
        initial_capital=10_000.0,
        algos=[RunMonthly(), SelectThese(["SPY", "TLT"]), WeighSpecified({"SPY": 0.6, "TLT": 0.4}), Rebalance()],
    )
    bal = bt.run()
    assert "SPY qty" in bal.columns
    assert "TLT qty" in bal.columns
    # SPY target = 10000 * 0.6 = 6000, qty = floor(6000/100) = 60
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 60
    # TLT target = 10000 * 0.4 = 4000, qty = floor(4000/50) = 80
    assert bal.loc[pd.Timestamp("2024-01-02"), "TLT qty"] == 80


# ---------------------------------------------------------------------------
# Helper: longer price history for algos needing lookback
# ---------------------------------------------------------------------------

def _daily_prices(symbols=("SPY", "TLT"), days=60, seed=42) -> pd.DataFrame:
    """Generate synthetic daily prices for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    data = {}
    for s in symbols:
        base = 100.0 if s == "SPY" else 50.0
        rets = rng.normal(0.0005, 0.01, days)
        data[s] = base * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=idx)


def _weekly_prices() -> pd.DataFrame:
    """Prices spanning two full weeks (Mon-Fri), so RunWeekly triggers twice."""
    idx = pd.to_datetime([
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
    ])
    return pd.DataFrame({"SPY": np.linspace(100, 110, 10)}, index=idx)


def _ctx(prices=None, total_capital=1000.0, cash=1000.0, positions=None,
         selected_symbols=None, target_weights=None, price_history=None,
         date=None) -> PipelineContext:
    """Shortcut to build a PipelineContext."""
    if prices is None:
        prices = pd.Series({"SPY": 100.0})
    if date is None:
        date = pd.Timestamp("2024-01-02")
    return PipelineContext(
        date=date,
        prices=prices,
        total_capital=total_capital,
        cash=cash,
        positions=positions or {},
        selected_symbols=selected_symbols or [],
        target_weights=target_weights or {},
        price_history=price_history,
    )


# ===========================================================================
# SCHEDULING ALGOS
# ===========================================================================


# ---------------------------------------------------------------------------
# RunWeekly
# ---------------------------------------------------------------------------

def test_run_weekly_triggers_once_per_week():
    algo = RunWeekly()
    prices = _weekly_prices()
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[algo, SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bal = bt.run()
    logs = bt.logs_dataframe()
    rebalance_dates = logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")]["date"]
    # Should rebalance on the first day of each week
    assert len(rebalance_dates) == 2


def test_run_weekly_reset():
    algo = RunWeekly()
    ctx1 = _ctx(date=pd.Timestamp("2024-01-08"))
    algo(ctx1)
    assert algo._last_week is not None
    algo.reset()
    assert algo._last_week is None


def test_run_weekly_skips_same_week():
    algo = RunWeekly()
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-08")))  # Mon
    d2 = algo(_ctx(date=pd.Timestamp("2024-01-09")))  # Tue same week
    assert d1.status == "continue"
    assert d2.status == "skip_day"


# ---------------------------------------------------------------------------
# RunQuarterly
# ---------------------------------------------------------------------------

def test_run_quarterly_triggers_once_per_quarter():
    idx = pd.to_datetime([
        "2024-01-02", "2024-02-01", "2024-03-01",
        "2024-04-01", "2024-05-01", "2024-06-01",
        "2024-07-01",
    ])
    prices = pd.DataFrame({"SPY": [100] * 7}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[RunQuarterly(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bt.run()
    logs = bt.logs_dataframe()
    rebalance_dates = logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")]["date"]
    # Q1 (Jan), Q2 (Apr), Q3 (Jul) = 3 rebalances
    assert len(rebalance_dates) == 3


def test_run_quarterly_skips_same_quarter():
    algo = RunQuarterly()
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-02")))
    d2 = algo(_ctx(date=pd.Timestamp("2024-02-15")))
    assert d1.status == "continue"
    assert d2.status == "skip_day"


def test_run_quarterly_reset():
    algo = RunQuarterly()
    algo(_ctx(date=pd.Timestamp("2024-01-02")))
    algo.reset()
    assert algo._last_quarter is None


# ---------------------------------------------------------------------------
# RunYearly
# ---------------------------------------------------------------------------

def test_run_yearly_triggers_once_per_year():
    idx = pd.to_datetime(["2024-01-02", "2024-06-01", "2024-12-31", "2025-01-02", "2025-06-01"])
    prices = pd.DataFrame({"SPY": [100] * 5}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[RunYearly(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bt.run()
    logs = bt.logs_dataframe()
    rebalance_dates = logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")]["date"]
    # 2024 and 2025 = 2 rebalances
    assert len(rebalance_dates) == 2


def test_run_yearly_skips_same_year():
    algo = RunYearly()
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-02")))
    d2 = algo(_ctx(date=pd.Timestamp("2024-06-15")))
    assert d1.status == "continue"
    assert d2.status == "skip_day"


def test_run_yearly_reset():
    algo = RunYearly()
    algo(_ctx(date=pd.Timestamp("2024-01-02")))
    algo.reset()
    assert algo._last_year is None


# ---------------------------------------------------------------------------
# RunDaily
# ---------------------------------------------------------------------------

def test_run_daily_always_continues():
    algo = RunDaily()
    for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        assert algo(_ctx(date=pd.Timestamp(d))).status == "continue"


def test_run_daily_full_pipeline():
    prices = _prices()
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[RunDaily(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bt.run()
    logs = bt.logs_dataframe()
    rebalance_count = len(logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")])
    assert rebalance_count == len(prices)


# ---------------------------------------------------------------------------
# RunOnce
# ---------------------------------------------------------------------------

def test_run_once_only_first_date():
    algo = RunOnce()
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-02")))
    d2 = algo(_ctx(date=pd.Timestamp("2024-01-03")))
    d3 = algo(_ctx(date=pd.Timestamp("2024-02-01")))
    assert d1.status == "continue"
    assert d2.status == "skip_day"
    assert d3.status == "skip_day"


def test_run_once_reset():
    algo = RunOnce()
    algo(_ctx(date=pd.Timestamp("2024-01-02")))
    assert algo._ran is True
    algo.reset()
    assert algo._ran is False
    d = algo(_ctx(date=pd.Timestamp("2024-02-01")))
    assert d.status == "continue"


def test_run_once_full_pipeline():
    prices = _prices()
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[RunOnce(), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bt.run()
    logs = bt.logs_dataframe()
    rebalance_count = len(logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")])
    assert rebalance_count == 1


# ---------------------------------------------------------------------------
# RunOnDate
# ---------------------------------------------------------------------------

def test_run_on_date_specific_dates():
    algo = RunOnDate(["2024-01-02", "2024-02-01"])
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-02")))
    d2 = algo(_ctx(date=pd.Timestamp("2024-01-03")))
    d3 = algo(_ctx(date=pd.Timestamp("2024-02-01")))
    assert d1.status == "continue"
    assert d2.status == "skip_day"
    assert d3.status == "continue"


def test_run_on_date_accepts_timestamps():
    algo = RunOnDate([pd.Timestamp("2024-03-15")])
    d = algo(_ctx(date=pd.Timestamp("2024-03-15")))
    assert d.status == "continue"


def test_run_on_date_full_pipeline():
    prices = _prices()
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[
            RunOnDate(["2024-02-01"]),
            SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance(),
        ],
    )
    bt.run()
    logs = bt.logs_dataframe()
    rebalance_count = len(logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")])
    assert rebalance_count == 1


# ---------------------------------------------------------------------------
# RunAfterDate
# ---------------------------------------------------------------------------

def test_run_after_date_skips_before():
    algo = RunAfterDate("2024-02-01")
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-15")))
    d2 = algo(_ctx(date=pd.Timestamp("2024-02-01")))
    d3 = algo(_ctx(date=pd.Timestamp("2024-03-01")))
    assert d1.status == "skip_day"
    assert d2.status == "continue"
    assert d3.status == "continue"


def test_run_after_date_full_pipeline():
    prices = _prices()
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[
            RunAfterDate("2024-02-01"),
            SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance(),
        ],
    )
    bal = bt.run()
    # First two dates (Jan 2, Jan 3) are skipped, Feb 1 and Feb 2 rebalance
    logs = bt.logs_dataframe()
    rebalance_count = len(logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")])
    assert rebalance_count == 2


# ---------------------------------------------------------------------------
# RunEveryNPeriods
# ---------------------------------------------------------------------------

def test_run_every_n_periods():
    algo = RunEveryNPeriods(3)
    results = []
    for i in range(9):
        d = algo(_ctx(date=pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)))
        results.append(d.status)
    # Period 1: continue (first), 2: skip, 3: skip, 4: continue, 5: skip, 6: skip, 7: continue, ...
    assert results == [
        "continue", "skip_day", "skip_day",
        "continue", "skip_day", "skip_day",
        "continue", "skip_day", "skip_day",
    ]


def test_run_every_n_periods_reset():
    algo = RunEveryNPeriods(5)
    for _ in range(3):
        algo(_ctx())
    algo.reset()
    assert algo._count == 0
    d = algo(_ctx())
    assert d.status == "continue"


# ---------------------------------------------------------------------------
# Or combinator
# ---------------------------------------------------------------------------

def test_or_passes_if_any_child_passes():
    algo = Or(RunMonthly(), RunWeekly())
    # First call: both children haven't seen any date, so both pass → Or passes
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-08")))
    assert d1.status == "continue"


def test_or_skips_if_all_children_skip():
    monthly = RunMonthly()
    weekly = RunWeekly()
    algo = Or(monthly, weekly)
    # First call: RunMonthly passes → Or short-circuits, RunWeekly never called
    algo(_ctx(date=pd.Timestamp("2024-01-08")))
    # Second call in same month: RunMonthly skips, RunWeekly sees first date → passes
    algo(_ctx(date=pd.Timestamp("2024-01-09")))
    # Third call: same month AND same week → both skip → Or skips
    d3 = algo(_ctx(date=pd.Timestamp("2024-01-10")))
    assert d3.status == "skip_day"


def test_or_passes_when_one_passes():
    monthly = RunMonthly()
    weekly = RunWeekly()
    algo = Or(monthly, weekly)
    algo(_ctx(date=pd.Timestamp("2024-01-08")))
    # New week but same month → weekly passes → Or passes
    d = algo(_ctx(date=pd.Timestamp("2024-01-15")))
    assert d.status == "continue"


def test_or_reset():
    monthly = RunMonthly()
    weekly = RunWeekly()
    algo = Or(monthly, weekly)
    algo(_ctx(date=pd.Timestamp("2024-01-08")))
    algo.reset()
    assert monthly._last_month is None
    assert weekly._last_week is None


# ---------------------------------------------------------------------------
# Not combinator
# ---------------------------------------------------------------------------

def test_not_inverts_skip_to_continue():
    monthly = RunMonthly()
    algo = Not(monthly)
    # First call: RunMonthly returns continue → Not inverts to skip_day
    d1 = algo(_ctx(date=pd.Timestamp("2024-01-02")))
    assert d1.status == "skip_day"


def test_not_inverts_continue_to_skip():
    monthly = RunMonthly()
    algo = Not(monthly)
    algo(_ctx(date=pd.Timestamp("2024-01-02")))
    # Same month → RunMonthly skips → Not inverts to continue
    d2 = algo(_ctx(date=pd.Timestamp("2024-01-03")))
    assert d2.status == "continue"


def test_not_reset():
    monthly = RunMonthly()
    algo = Not(monthly)
    algo(_ctx(date=pd.Timestamp("2024-01-02")))
    algo.reset()
    assert monthly._last_month is None


# ===========================================================================
# SELECTION ALGOS
# ===========================================================================


# ---------------------------------------------------------------------------
# SelectAll
# ---------------------------------------------------------------------------

def test_select_all_picks_valid_prices():
    ctx = _ctx(prices=pd.Series({"SPY": 100.0, "TLT": 50.0, "BAD": np.nan}))
    d = SelectAll()(ctx)
    assert d.status == "continue"
    assert set(ctx.selected_symbols) == {"SPY", "TLT"}


def test_select_all_skips_zero_price():
    ctx = _ctx(prices=pd.Series({"SPY": 0.0}))
    d = SelectAll()(ctx)
    assert d.status == "skip_day"


def test_select_all_skips_all_nan():
    ctx = _ctx(prices=pd.Series({"SPY": np.nan, "TLT": np.nan}))
    d = SelectAll()(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# SelectHasData
# ---------------------------------------------------------------------------

def test_select_has_data_filters_by_history_length():
    prices = _daily_prices(days=10)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    algo = SelectHasData(min_days=10)
    d = algo(ctx)
    assert d.status == "continue"
    assert set(ctx.selected_symbols) == {"SPY", "TLT"}


def test_select_has_data_removes_short_history():
    prices = _daily_prices(days=5)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    algo = SelectHasData(min_days=10)
    d = algo(ctx)
    assert d.status == "skip_day"


def test_select_has_data_no_history():
    ctx = _ctx(selected_symbols=["SPY"])
    algo = SelectHasData(min_days=1)
    d = algo(ctx)
    assert d.status == "skip_day"


def test_select_has_data_uses_all_symbols_if_none_selected():
    prices = _daily_prices(days=5)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=[],  # empty
        price_history=prices,
    )
    algo = SelectHasData(min_days=3)
    d = algo(ctx)
    assert d.status == "continue"
    assert set(ctx.selected_symbols) == {"SPY", "TLT"}


# ---------------------------------------------------------------------------
# SelectMomentum
# ---------------------------------------------------------------------------

def test_select_momentum_picks_top_n():
    # SPY goes up, TLT goes down
    idx = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.DataFrame({
        "SPY": np.linspace(100, 120, 20),  # +20%
        "TLT": np.linspace(100, 90, 20),   # -10%
        "GLD": np.linspace(100, 105, 20),  # +5%
    }, index=idx)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT", "GLD"],
        price_history=prices,
    )
    algo = SelectMomentum(n=2, lookback=20)
    d = algo(ctx)
    assert d.status == "continue"
    assert ctx.selected_symbols == ["SPY", "GLD"]


def test_select_momentum_ascending():
    idx = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.DataFrame({
        "SPY": np.linspace(100, 120, 20),
        "TLT": np.linspace(100, 90, 20),
    }, index=idx)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    algo = SelectMomentum(n=1, lookback=20, sort_descending=False)
    algo(ctx)
    assert ctx.selected_symbols == ["TLT"]


def test_select_momentum_no_history():
    ctx = _ctx(selected_symbols=["SPY"])
    d = SelectMomentum(n=1)(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# SelectN
# ---------------------------------------------------------------------------

def test_select_n_truncates():
    ctx = _ctx(selected_symbols=["SPY", "TLT", "GLD", "QQQ"])
    d = SelectN(2)(ctx)
    assert d.status == "continue"
    assert ctx.selected_symbols == ["SPY", "TLT"]


def test_select_n_empty():
    ctx = _ctx(selected_symbols=[])
    d = SelectN(5)(ctx)
    assert d.status == "skip_day"


def test_select_n_fewer_than_n():
    ctx = _ctx(selected_symbols=["SPY"])
    d = SelectN(5)(ctx)
    assert d.status == "continue"
    assert ctx.selected_symbols == ["SPY"]


# ---------------------------------------------------------------------------
# SelectWhere
# ---------------------------------------------------------------------------

def test_select_where_custom_filter():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0, "GLD": 200.0}),
        selected_symbols=["SPY", "TLT", "GLD"],
    )
    # Only keep symbols with price > 80
    algo = SelectWhere(lambda s, c: float(c.prices[s]) > 80)
    d = algo(ctx)
    assert d.status == "continue"
    assert set(ctx.selected_symbols) == {"SPY", "GLD"}


def test_select_where_all_filtered():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0}),
        selected_symbols=["SPY"],
    )
    algo = SelectWhere(lambda s, c: False)
    d = algo(ctx)
    assert d.status == "skip_day"


def test_select_where_falls_back_to_prices_index():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        selected_symbols=[],  # empty
    )
    algo = SelectWhere(lambda s, c: s == "TLT")
    d = algo(ctx)
    assert d.status == "continue"
    assert ctx.selected_symbols == ["TLT"]


# ===========================================================================
# WEIGHTING ALGOS
# ===========================================================================


# ---------------------------------------------------------------------------
# WeighEqually
# ---------------------------------------------------------------------------

def test_weigh_equally_two_symbols():
    ctx = _ctx(selected_symbols=["SPY", "TLT"])
    d = WeighEqually()(ctx)
    assert d.status == "continue"
    assert abs(ctx.target_weights["SPY"] - 0.5) < 1e-12
    assert abs(ctx.target_weights["TLT"] - 0.5) < 1e-12


def test_weigh_equally_single_symbol():
    ctx = _ctx(selected_symbols=["SPY"])
    WeighEqually()(ctx)
    assert abs(ctx.target_weights["SPY"] - 1.0) < 1e-12


def test_weigh_equally_empty():
    ctx = _ctx(selected_symbols=[])
    d = WeighEqually()(ctx)
    assert d.status == "skip_day"


def test_weigh_equally_three_symbols():
    ctx = _ctx(selected_symbols=["SPY", "TLT", "GLD"])
    WeighEqually()(ctx)
    for s in ["SPY", "TLT", "GLD"]:
        assert abs(ctx.target_weights[s] - 1.0 / 3) < 1e-12


# ---------------------------------------------------------------------------
# WeighInvVol
# ---------------------------------------------------------------------------

def test_weigh_inv_vol_basic():
    prices = _daily_prices(days=30)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    d = WeighInvVol(lookback=30)(ctx)
    assert d.status == "continue"
    assert abs(sum(ctx.target_weights.values()) - 1.0) < 1e-10
    assert all(w > 0 for w in ctx.target_weights.values())


def test_weigh_inv_vol_lower_vol_gets_higher_weight():
    # Create data where TLT has much lower vol than SPY
    idx = pd.bdate_range("2024-01-02", periods=30)
    rng = np.random.RandomState(99)
    spy = 100 * np.cumprod(1 + rng.normal(0, 0.03, 30))  # high vol
    tlt = 50 * np.cumprod(1 + rng.normal(0, 0.005, 30))   # low vol
    prices = pd.DataFrame({"SPY": spy, "TLT": tlt}, index=idx)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    WeighInvVol(lookback=30)(ctx)
    # Lower vol (TLT) should get higher weight
    assert ctx.target_weights["TLT"] > ctx.target_weights["SPY"]


def test_weigh_inv_vol_no_history():
    ctx = _ctx(selected_symbols=["SPY"])
    d = WeighInvVol()(ctx)
    assert d.status == "skip_day"


def test_weigh_inv_vol_no_selected():
    ctx = _ctx(selected_symbols=[])
    d = WeighInvVol()(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# WeighMeanVar
# ---------------------------------------------------------------------------

def test_weigh_mean_var_basic():
    prices = _daily_prices(days=30)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    d = WeighMeanVar(lookback=30)(ctx)
    assert d.status == "continue"
    assert abs(sum(ctx.target_weights.values()) - 1.0) < 1e-10
    assert all(w >= 0 for w in ctx.target_weights.values())


def test_weigh_mean_var_single_asset():
    prices = _daily_prices(symbols=("SPY",), days=30)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY"],
        price_history=prices,
    )
    d = WeighMeanVar(lookback=30)(ctx)
    assert d.status == "continue"
    assert abs(ctx.target_weights["SPY"] - 1.0) < 1e-10


def test_weigh_mean_var_no_history():
    ctx = _ctx(selected_symbols=["SPY"])
    d = WeighMeanVar()(ctx)
    assert d.status == "skip_day"


def test_weigh_mean_var_insufficient_data():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=idx)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY"],
        price_history=prices,
    )
    d = WeighMeanVar(lookback=252)(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# WeighERC
# ---------------------------------------------------------------------------

def test_weigh_erc_basic():
    prices = _daily_prices(days=30)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        price_history=prices,
    )
    d = WeighERC(lookback=30)(ctx)
    assert d.status == "continue"
    assert abs(sum(ctx.target_weights.values()) - 1.0) < 1e-10
    assert all(w > 0 for w in ctx.target_weights.values())


def test_weigh_erc_no_history():
    ctx = _ctx(selected_symbols=["SPY"])
    d = WeighERC()(ctx)
    assert d.status == "skip_day"


def test_weigh_erc_single_asset():
    prices = _daily_prices(symbols=("SPY",), days=30)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY"],
        price_history=prices,
    )
    d = WeighERC(lookback=30)(ctx)
    assert d.status == "continue"
    assert abs(ctx.target_weights["SPY"] - 1.0) < 1e-10


def test_weigh_erc_weights_sum_to_one():
    prices = _daily_prices(symbols=("SPY", "TLT", "GLD"), days=60, seed=123)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT", "GLD"],
        price_history=prices,
    )
    WeighERC(lookback=60)(ctx)
    assert abs(sum(ctx.target_weights.values()) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# TargetVol
# ---------------------------------------------------------------------------

def test_target_vol_scales_weights():
    prices = _daily_prices(days=60)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        target_weights={"SPY": 0.6, "TLT": 0.4},
        price_history=prices,
    )
    d = TargetVol(target=0.05, lookback=60)(ctx)
    assert d.status == "continue"
    # Weights should be scaled down (realized vol likely > 5%)
    total_w = sum(ctx.target_weights.values())
    assert total_w <= 1.0 + 1e-10


def test_target_vol_no_weights():
    ctx = _ctx(target_weights={})
    d = TargetVol(target=0.10)(ctx)
    assert d.status == "skip_day"


def test_target_vol_no_history():
    ctx = _ctx(target_weights={"SPY": 1.0})
    d = TargetVol(target=0.10)(ctx)
    assert d.status == "skip_day"


def test_target_vol_never_levers():
    """TargetVol should never scale weights above 1.0."""
    # Create very low vol data
    idx = pd.bdate_range("2024-01-02", periods=60)
    prices = pd.DataFrame({
        "SPY": np.linspace(100, 100.5, 60),  # nearly flat → near-zero vol
    }, index=idx)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        target_weights={"SPY": 1.0},
        price_history=prices,
    )
    TargetVol(target=0.50, lookback=60)(ctx)
    # Scale should be capped at 1.0
    assert ctx.target_weights["SPY"] <= 1.0 + 1e-10


# ===========================================================================
# WEIGHT LIMITS
# ===========================================================================


def test_limit_weights_caps():
    ctx = _ctx(target_weights={"SPY": 0.80, "TLT": 0.20})
    LimitWeights(limit=0.50)(ctx)
    assert ctx.target_weights["SPY"] <= 0.50 + 1e-10
    # Total should still be close to 1.0
    assert abs(sum(ctx.target_weights.values()) - 1.0) < 1e-8


def test_limit_weights_no_change_under_limit():
    ctx = _ctx(target_weights={"SPY": 0.40, "TLT": 0.60})
    LimitWeights(limit=0.70)(ctx)
    assert abs(ctx.target_weights["SPY"] - 0.40) < 1e-10
    assert abs(ctx.target_weights["TLT"] - 0.60) < 1e-10


def test_limit_weights_empty():
    ctx = _ctx(target_weights={})
    d = LimitWeights(limit=0.25)(ctx)
    assert d.status == "continue"


def test_limit_weights_redistributes():
    ctx = _ctx(target_weights={"A": 0.70, "B": 0.20, "C": 0.10})
    LimitWeights(limit=0.40)(ctx)
    assert ctx.target_weights["A"] <= 0.40 + 1e-10
    # B and C should get the excess redistributed
    assert ctx.target_weights["B"] > 0.20
    assert ctx.target_weights["C"] > 0.10


# ===========================================================================
# CAPITAL FLOWS
# ===========================================================================


def test_capital_flow_dict():
    flow = CapitalFlow({"2024-02-01": 500.0})
    ctx = _ctx(date=pd.Timestamp("2024-02-01"), cash=1000.0, total_capital=1000.0)
    flow(ctx)
    assert ctx.cash == 1500.0
    assert ctx.total_capital == 1500.0


def test_capital_flow_no_flow_date():
    flow = CapitalFlow({"2024-02-01": 500.0})
    ctx = _ctx(date=pd.Timestamp("2024-01-15"), cash=1000.0, total_capital=1000.0)
    flow(ctx)
    assert ctx.cash == 1000.0


def test_capital_flow_withdrawal():
    flow = CapitalFlow({"2024-02-01": -200.0})
    ctx = _ctx(date=pd.Timestamp("2024-02-01"), cash=1000.0, total_capital=1000.0)
    flow(ctx)
    assert ctx.cash == 800.0
    assert ctx.total_capital == 800.0


def test_capital_flow_callable():
    # Add 100 on every Monday
    def monday_flow(d: pd.Timestamp) -> float:
        return 100.0 if d.weekday() == 0 else 0.0

    flow = CapitalFlow(monday_flow)
    ctx_mon = _ctx(date=pd.Timestamp("2024-01-08"), cash=1000.0, total_capital=1000.0)
    flow(ctx_mon)
    assert ctx_mon.cash == 1100.0

    ctx_tue = _ctx(date=pd.Timestamp("2024-01-09"), cash=1000.0, total_capital=1000.0)
    flow(ctx_tue)
    assert ctx_tue.cash == 1000.0


def test_capital_flow_in_pipeline():
    idx = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"])
    prices = pd.DataFrame({"SPY": [100.0, 100.0, 100.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[
            RunMonthly(),
            CapitalFlow({"2024-02-01": 500.0}),
            SelectThese(["SPY"]),
            WeighSpecified({"SPY": 1.0}),
            Rebalance(),
        ],
    )
    bal = bt.run()
    # On Feb 1, capital should include the 500 addition
    assert bal.loc[pd.Timestamp("2024-02-01"), "total capital"] > 1000.0


# ===========================================================================
# REBALANCE OVER TIME
# ===========================================================================


def test_rebalance_over_time_gradual():
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame({"SPY": [100.0] * 10}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[
            RunDaily(),
            SelectThese(["SPY"]),
            WeighSpecified({"SPY": 1.0}),
            RebalanceOverTime(n=5),
        ],
    )
    bal = bt.run()
    # With n=5, the first 5 days should gradually increase position
    qtys = [bal.iloc[i].get("SPY qty", 0) for i in range(5)]
    # Each day should get closer to full position
    assert qtys[-1] >= qtys[0]


def test_rebalance_over_time_reset():
    algo = RebalanceOverTime(n=3)
    algo._target = {"SPY": 1.0}
    algo._remaining = 2
    algo.reset()
    assert algo._target == {}
    assert algo._remaining == 0


def test_rebalance_over_time_no_target():
    algo = RebalanceOverTime(n=3)
    ctx = _ctx()
    d = algo(ctx)
    assert d.status == "skip_day"


# ===========================================================================
# INTEGRATION: Full pipeline with new algos
# ===========================================================================


def test_pipeline_select_all_weigh_equally():
    idx = pd.to_datetime(["2024-01-02", "2024-02-01"])
    prices = pd.DataFrame({"SPY": [100.0, 100.0], "TLT": [50.0, 50.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[RunMonthly(), SelectAll(), WeighEqually(), Rebalance()],
    )
    bal = bt.run()
    # 50% each: SPY = floor(5000/100) = 50, TLT = floor(5000/50) = 100
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 50
    assert bal.loc[pd.Timestamp("2024-01-02"), "TLT qty"] == 100


def test_pipeline_momentum_selection():
    idx = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame({
        "SPY": np.linspace(100, 130, 30),  # +30%
        "TLT": np.linspace(100, 95, 30),   # -5%
        "GLD": np.linspace(100, 110, 30),  # +10%
    }, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            RunMonthly(),
            SelectAll(),
            SelectMomentum(n=2, lookback=30),
            WeighEqually(),
            Rebalance(),
        ],
    )
    bal = bt.run()
    # Should pick SPY and GLD (top 2 momentum), not TLT
    assert "SPY qty" in bal.columns
    assert "GLD qty" in bal.columns
    # TLT should not have been bought (or have 0 qty)
    if "TLT qty" in bal.columns:
        assert bal["TLT qty"].fillna(0).sum() == 0


def test_pipeline_limit_weights_integration():
    idx = pd.to_datetime(["2024-01-02"])
    prices = pd.DataFrame({"SPY": [100.0], "TLT": [50.0], "GLD": [200.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            SelectThese(["SPY", "TLT", "GLD"]),
            WeighSpecified({"SPY": 0.8, "TLT": 0.1, "GLD": 0.1}),
            LimitWeights(limit=0.40),
            Rebalance(),
        ],
    )
    bal = bt.run()
    spy_val = bal.iloc[0]["SPY qty"] * 100.0
    total = bal.iloc[0]["total capital"]
    # SPY weight should be ≤ 40%
    assert spy_val / total <= 0.45  # small tolerance for floor rounding


def test_pipeline_run_on_date_with_capital_flow():
    idx = pd.to_datetime(["2024-01-02", "2024-01-15", "2024-02-01", "2024-02-15"])
    prices = pd.DataFrame({"SPY": [100.0] * 4}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[
            RunOnDate(["2024-01-02", "2024-02-01"]),
            CapitalFlow({"2024-02-01": 500.0}),
            SelectThese(["SPY"]),
            WeighSpecified({"SPY": 1.0}),
            Rebalance(),
        ],
    )
    bal = bt.run()
    # Jan 2: 1000/100 = 10 shares
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 10
    # Feb 1: 1000 + 500 = 1500, 1500/100 = 15 shares
    assert bal.loc[pd.Timestamp("2024-02-01"), "SPY qty"] == 15


def test_pipeline_inv_vol_with_limit_weights():
    prices = _daily_prices(symbols=("SPY", "TLT", "GLD"), days=60, seed=77)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=100_000.0,
        algos=[
            RunMonthly(),
            SelectAll(),
            WeighInvVol(lookback=60),
            LimitWeights(limit=0.50),
            Rebalance(),
        ],
    )
    bal = bt.run()
    assert not bal.empty
    # All weights should respect the 50% limit (check via position values)
    row = bal.iloc[-1]
    total = row["total capital"]
    for sym in ["SPY", "TLT", "GLD"]:
        qty_col = f"{sym} qty"
        if qty_col in row.index and row[qty_col] > 0:
            price = prices[sym].iloc[-1]
            weight = row[qty_col] * price / total
            assert weight <= 0.55  # tolerance for floor rounding


# ===========================================================================
# NEW ALGOS (round 2)
# ===========================================================================


# ---------------------------------------------------------------------------
# RunAfterDays
# ---------------------------------------------------------------------------

def test_run_after_days_skips_warmup():
    algo = RunAfterDays(3)
    results = []
    for i in range(6):
        d = algo(_ctx(date=pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)))
        results.append(d.status)
    assert results == ["skip_day", "skip_day", "skip_day", "continue", "continue", "continue"]


def test_run_after_days_reset():
    algo = RunAfterDays(2)
    algo(_ctx())
    algo(_ctx())
    algo.reset()
    assert algo._count == 0
    d = algo(_ctx())
    assert d.status == "skip_day"  # back to warmup


def test_run_after_days_in_pipeline():
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame({"SPY": [100.0] * 10}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[RunAfterDays(5), SelectThese(["SPY"]), WeighSpecified({"SPY": 1.0}), Rebalance()],
    )
    bal = bt.run()
    # First 5 days skipped, rebalance on days 6-10
    logs = bt.logs_dataframe()
    rebalance_count = len(logs[(logs["step"] == "Rebalance") & (logs["status"] == "continue")])
    assert rebalance_count == 5


# ---------------------------------------------------------------------------
# RunIfOutOfBounds
# ---------------------------------------------------------------------------

def test_run_if_out_of_bounds_skips_when_in_bounds():
    algo = RunIfOutOfBounds(tolerance=0.10)
    algo.update_target({"SPY": 0.60, "TLT": 0.40})
    # Positions match target closely
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        total_capital=10000.0,
        positions={"SPY": 60.0, "TLT": 80.0},  # SPY=60%, TLT=40%
    )
    d = algo(ctx)
    assert d.status == "skip_day"


def test_run_if_out_of_bounds_triggers_when_drifted():
    algo = RunIfOutOfBounds(tolerance=0.05)
    algo.update_target({"SPY": 0.50, "TLT": 0.50})
    # SPY drifted to 70%, TLT to 30%
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        total_capital=10000.0,
        positions={"SPY": 70.0, "TLT": 60.0},  # SPY=70%, TLT=30%
    )
    d = algo(ctx)
    assert d.status == "continue"


def test_run_if_out_of_bounds_no_prior_target():
    algo = RunIfOutOfBounds(tolerance=0.05)
    d = algo(_ctx())
    assert d.status == "skip_day"


def test_run_if_out_of_bounds_reset():
    algo = RunIfOutOfBounds(tolerance=0.05)
    algo.update_target({"SPY": 1.0})
    algo.reset()
    assert algo._last_target == {}


# ---------------------------------------------------------------------------
# LimitDeltas
# ---------------------------------------------------------------------------

def test_limit_deltas_clips_large_change():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        total_capital=10000.0,
        positions={"SPY": 50.0, "TLT": 100.0},  # SPY=50%, TLT=50%
        target_weights={"SPY": 0.80, "TLT": 0.20},  # want to move 30%
    )
    LimitDeltas(limit=0.10)(ctx)
    # SPY delta capped: 0.50 + 0.10 = 0.60 max
    assert ctx.target_weights["SPY"] <= 0.65  # after renorm


def test_limit_deltas_no_change_needed():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0}),
        total_capital=10000.0,
        positions={"SPY": 98.0},  # ~98%
        target_weights={"SPY": 1.0},
    )
    LimitDeltas(limit=0.10)(ctx)
    # Small delta, should pass through mostly unchanged
    assert ctx.target_weights["SPY"] > 0.9


def test_limit_deltas_empty():
    ctx = _ctx(target_weights={})
    d = LimitDeltas(limit=0.10)(ctx)
    assert d.status == "continue"


# ---------------------------------------------------------------------------
# ScaleWeights
# ---------------------------------------------------------------------------

def test_scale_weights_half():
    ctx = _ctx(target_weights={"SPY": 0.60, "TLT": 0.40})
    ScaleWeights(scale=0.5)(ctx)
    assert abs(ctx.target_weights["SPY"] - 0.30) < 1e-10
    assert abs(ctx.target_weights["TLT"] - 0.20) < 1e-10


def test_scale_weights_double():
    ctx = _ctx(target_weights={"SPY": 0.30, "TLT": 0.20})
    ScaleWeights(scale=2.0)(ctx)
    assert abs(ctx.target_weights["SPY"] - 0.60) < 1e-10
    assert abs(ctx.target_weights["TLT"] - 0.40) < 1e-10


def test_scale_weights_empty():
    ctx = _ctx(target_weights={})
    d = ScaleWeights(scale=0.5)(ctx)
    assert d.status == "continue"


# ---------------------------------------------------------------------------
# SelectRandomly
# ---------------------------------------------------------------------------

def test_select_randomly_picks_n():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0, "GLD": 200.0, "QQQ": 300.0}),
        selected_symbols=["SPY", "TLT", "GLD", "QQQ"],
    )
    algo = SelectRandomly(n=2, seed=42)
    d = algo(ctx)
    assert d.status == "continue"
    assert len(ctx.selected_symbols) == 2
    assert all(s in ["SPY", "TLT", "GLD", "QQQ"] for s in ctx.selected_symbols)


def test_select_randomly_deterministic():
    ctx1 = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0, "GLD": 200.0}),
        selected_symbols=["SPY", "TLT", "GLD"],
    )
    ctx2 = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0, "GLD": 200.0}),
        selected_symbols=["SPY", "TLT", "GLD"],
    )
    algo1 = SelectRandomly(n=2, seed=42)
    algo2 = SelectRandomly(n=2, seed=42)
    algo1(ctx1)
    algo2(ctx2)
    assert ctx1.selected_symbols == ctx2.selected_symbols


def test_select_randomly_n_exceeds_candidates():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0}),
        selected_symbols=["SPY"],
    )
    algo = SelectRandomly(n=5, seed=1)
    d = algo(ctx)
    assert d.status == "continue"
    assert ctx.selected_symbols == ["SPY"]


def test_select_randomly_no_candidates():
    ctx = _ctx(
        prices=pd.Series({"SPY": np.nan}),
        selected_symbols=[],
    )
    algo = SelectRandomly(n=2, seed=1)
    d = algo(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# SelectActive
# ---------------------------------------------------------------------------

def test_select_active_filters_dead():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 0.0, "GLD": np.nan}),
        selected_symbols=["SPY", "TLT", "GLD"],
    )
    d = SelectActive()(ctx)
    assert d.status == "continue"
    assert ctx.selected_symbols == ["SPY"]


def test_select_active_all_dead():
    ctx = _ctx(
        prices=pd.Series({"SPY": 0.0, "TLT": np.nan}),
        selected_symbols=["SPY", "TLT"],
    )
    d = SelectActive()(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# WeighRandomly
# ---------------------------------------------------------------------------

def test_weigh_randomly_sums_to_one():
    ctx = _ctx(selected_symbols=["SPY", "TLT", "GLD"])
    WeighRandomly(seed=42)(ctx)
    assert abs(sum(ctx.target_weights.values()) - 1.0) < 1e-10
    assert all(w > 0 for w in ctx.target_weights.values())


def test_weigh_randomly_deterministic():
    ctx1 = _ctx(selected_symbols=["SPY", "TLT"])
    ctx2 = _ctx(selected_symbols=["SPY", "TLT"])
    WeighRandomly(seed=99)(ctx1)
    WeighRandomly(seed=99)(ctx2)
    assert abs(ctx1.target_weights["SPY"] - ctx2.target_weights["SPY"]) < 1e-10


def test_weigh_randomly_empty():
    ctx = _ctx(selected_symbols=[])
    d = WeighRandomly(seed=1)(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# WeighTarget
# ---------------------------------------------------------------------------

def test_weigh_target_basic():
    weights_df = pd.DataFrame(
        {"SPY": [0.60, 0.70], "TLT": [0.40, 0.30]},
        index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
    )
    algo = WeighTarget(weights_df)
    ctx = _ctx(
        date=pd.Timestamp("2024-01-15"),
        selected_symbols=["SPY", "TLT"],
    )
    d = algo(ctx)
    assert d.status == "continue"
    # Should pick Jan 1 row (most recent before Jan 15)
    assert abs(ctx.target_weights["SPY"] - 0.60) < 1e-10
    assert abs(ctx.target_weights["TLT"] - 0.40) < 1e-10


def test_weigh_target_uses_latest_row():
    weights_df = pd.DataFrame(
        {"SPY": [0.50, 0.80]},
        index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
    )
    algo = WeighTarget(weights_df)
    ctx = _ctx(
        date=pd.Timestamp("2024-03-01"),
        selected_symbols=["SPY"],
    )
    algo(ctx)
    assert abs(ctx.target_weights["SPY"] - 1.0) < 1e-10  # normalized from 0.80


def test_weigh_target_no_data_before_date():
    weights_df = pd.DataFrame(
        {"SPY": [1.0]},
        index=pd.to_datetime(["2024-06-01"]),
    )
    algo = WeighTarget(weights_df)
    ctx = _ctx(
        date=pd.Timestamp("2024-01-01"),
        selected_symbols=["SPY"],
    )
    d = algo(ctx)
    assert d.status == "skip_day"


def test_weigh_target_empty_selected():
    weights_df = pd.DataFrame(
        {"SPY": [1.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    ctx = _ctx(date=pd.Timestamp("2024-01-15"), selected_symbols=[])
    d = WeighTarget(weights_df)(ctx)
    assert d.status == "skip_day"


# ---------------------------------------------------------------------------
# CloseDead
# ---------------------------------------------------------------------------

def test_close_dead_removes_zero_price():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 0.0}),
        positions={"SPY": 10.0, "TLT": 20.0},
    )
    CloseDead()(ctx)
    assert "SPY" in ctx.positions
    assert "TLT" not in ctx.positions


def test_close_dead_removes_nan_price():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": np.nan}),
        positions={"SPY": 10.0, "TLT": 20.0},
    )
    CloseDead()(ctx)
    assert "TLT" not in ctx.positions


def test_close_dead_no_dead():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        positions={"SPY": 10.0, "TLT": 20.0},
    )
    d = CloseDead()(ctx)
    assert d.status == "continue"
    assert len(ctx.positions) == 2


def test_close_dead_missing_price():
    ctx = _ctx(
        prices=pd.Series({"SPY": 100.0}),
        positions={"SPY": 10.0, "XYZ": 5.0},  # XYZ not in prices
    )
    CloseDead()(ctx)
    assert "XYZ" not in ctx.positions


# ---------------------------------------------------------------------------
# ClosePositionsAfterDates
# ---------------------------------------------------------------------------

def test_close_positions_after_dates():
    algo = ClosePositionsAfterDates({"TLT": "2024-02-01"})
    ctx = _ctx(
        date=pd.Timestamp("2024-02-15"),
        positions={"SPY": 10.0, "TLT": 20.0},
    )
    d = algo(ctx)
    assert "TLT" not in ctx.positions
    assert "SPY" in ctx.positions
    assert "closed after date" in d.message


def test_close_positions_before_date():
    algo = ClosePositionsAfterDates({"TLT": "2024-06-01"})
    ctx = _ctx(
        date=pd.Timestamp("2024-02-15"),
        positions={"SPY": 10.0, "TLT": 20.0},
    )
    algo(ctx)
    assert "TLT" in ctx.positions  # not yet


def test_close_positions_on_exact_date():
    algo = ClosePositionsAfterDates({"SPY": "2024-02-01"})
    ctx = _ctx(
        date=pd.Timestamp("2024-02-01"),
        positions={"SPY": 10.0},
    )
    algo(ctx)
    assert "SPY" not in ctx.positions


# ---------------------------------------------------------------------------
# Require
# ---------------------------------------------------------------------------

def test_require_passes_when_inner_passes():
    inner = RunDaily()  # always passes
    algo = Require(inner)
    d = algo(_ctx())
    assert d.status == "continue"


def test_require_blocks_when_inner_skips():
    inner = RunOnce()
    inner._ran = True  # already ran → will skip
    algo = Require(inner)
    d = algo(_ctx())
    assert d.status == "skip_day"


def test_require_reset():
    inner = RunOnce()
    inner._ran = True
    algo = Require(inner)
    algo.reset()
    assert inner._ran is False


# ---------------------------------------------------------------------------
# benchmark_random
# ---------------------------------------------------------------------------

def test_benchmark_random_basic():
    prices = _daily_prices(symbols=("SPY", "TLT"), days=30)
    strategy_algos = [
        RunMonthly(),
        SelectThese(["SPY"]),
        WeighSpecified({"SPY": 1.0}),
        Rebalance(),
    ]
    result = benchmark_random(
        prices=prices,
        strategy_algos=strategy_algos,
        n_random=10,
        initial_capital=10_000.0,
        seed=42,
    )
    assert isinstance(result, RandomBenchmarkResult)
    assert len(result.random_returns) == 10
    assert 0 <= result.percentile <= 100
    assert result.mean_random != 0.0 or all(r == 0 for r in result.random_returns)


def test_benchmark_random_deterministic():
    prices = _daily_prices(days=30)
    algos = [RunMonthly(), SelectAll(), WeighEqually(), Rebalance()]
    r1 = benchmark_random(prices, algos, n_random=5, seed=42)
    r2 = benchmark_random(prices, algos, n_random=5, seed=42)
    assert r1.random_returns == r2.random_returns
    assert r1.percentile == r2.percentile


def test_benchmark_random_result_properties():
    result = RandomBenchmarkResult(
        strategy_return=0.10,
        random_returns=[0.05, 0.08, 0.12, 0.03],
        percentile=50.0,
    )
    assert abs(result.mean_random - np.mean([0.05, 0.08, 0.12, 0.03])) < 1e-10
    assert abs(result.std_random - np.std([0.05, 0.08, 0.12, 0.03])) < 1e-10


# ===========================================================================
# INTEGRATION: Round 2 algos in full pipelines
# ===========================================================================


def test_pipeline_or_run_if_out_of_bounds():
    """Or(RunQuarterly(), RunIfOutOfBounds(0.05)) pattern."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"SPY": [100.0] * 5}, index=idx)
    oob = RunIfOutOfBounds(tolerance=0.05)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[
            Or(RunQuarterly(), oob),
            SelectThese(["SPY"]),
            WeighSpecified({"SPY": 1.0}),
            Rebalance(),
        ],
    )
    bal = bt.run()
    assert not bal.empty


def test_pipeline_close_dead_then_rebalance():
    idx = pd.to_datetime(["2024-01-02", "2024-02-01"])
    prices = pd.DataFrame({"SPY": [100.0, 100.0], "TLT": [50.0, 0.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            RunMonthly(), CloseDead(),
            SelectActive(), WeighEqually(), Rebalance(),
        ],
    )
    bal = bt.run()
    # On Feb 1, TLT is dead → should not hold any TLT
    if "TLT qty" in bal.columns:
        assert bal.loc[pd.Timestamp("2024-02-01"), "TLT qty"] == 0 or \
               pd.isna(bal.loc[pd.Timestamp("2024-02-01"), "TLT qty"])


def test_pipeline_scale_weights_deleverage():
    idx = pd.to_datetime(["2024-01-02"])
    prices = pd.DataFrame({"SPY": [100.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            SelectThese(["SPY"]),
            WeighSpecified({"SPY": 1.0}),
            ScaleWeights(scale=0.5),
            Rebalance(),
        ],
    )
    bal = bt.run()
    # 50% of 10000 = 5000, floor(5000/100) = 50 shares
    assert bal.iloc[0]["SPY qty"] == 50


def test_pipeline_select_randomly_weigh_randomly():
    prices = _daily_prices(symbols=("SPY", "TLT", "GLD"), days=30)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            RunMonthly(),
            SelectRandomly(n=2, seed=42),
            WeighRandomly(seed=42),
            Rebalance(),
        ],
    )
    bal = bt.run()
    assert not bal.empty


def test_pipeline_weigh_target_from_df():
    weights_df = pd.DataFrame(
        {"SPY": [0.60, 0.40], "TLT": [0.40, 0.60]},
        index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
    )
    idx = pd.to_datetime(["2024-01-15", "2024-02-15"])
    prices = pd.DataFrame({"SPY": [100.0, 100.0], "TLT": [50.0, 50.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            SelectThese(["SPY", "TLT"]),
            WeighTarget(weights_df),
            Rebalance(),
        ],
    )
    bal = bt.run()
    # Jan 15 → uses Jan 1 weights (60/40)
    spy_val = bal.iloc[0]["SPY qty"] * 100.0
    total = bal.iloc[0]["total capital"]
    assert spy_val / total > 0.55  # roughly 60%


# ---------------------------------------------------------------------------
# SelectRegex
# ---------------------------------------------------------------------------

def test_select_regex_matches():
    ctx = _ctx()
    ctx.prices = pd.Series({"SPY": 100, "SPXL": 50, "QQQ": 200, "IWM": 80})
    algo = SelectRegex(r"^SP")
    algo(ctx)
    assert sorted(ctx.selected_symbols) == ["SPXL", "SPY"]


def test_select_regex_no_match_skips():
    ctx = _ctx()
    algo = SelectRegex(r"^ZZZZZ")
    decision = algo(ctx)
    assert decision.status == "skip_day"


def test_select_regex_case_insensitive():
    ctx = _ctx()
    ctx.prices = pd.Series({"spy": 100, "SPY": 100, "QQQ": 200})
    algo = SelectRegex(r"(?i)spy")
    algo(ctx)
    assert set(ctx.selected_symbols) == {"spy", "SPY"}


def test_select_regex_in_pipeline():
    prices = pd.DataFrame(
        {"SPY": [100, 102], "SPXL": [50, 51], "QQQ": [200, 202]},
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    bt = AlgoPipelineBacktester(
        prices=prices,
        initial_capital=1000.0,
        algos=[RunDaily(), SelectRegex(r"^SP"), WeighEqually(), Rebalance()],
    )
    bal = bt.run()
    # Should only hold SPY and SPXL, not QQQ
    assert bal.iloc[-1].get("QQQ qty", 0) == 0
    assert bal.iloc[-1]["SPY qty"] > 0
    assert bal.iloc[-1]["SPXL qty"] > 0


# ===========================================================================
# HEDGE RISKS
# ===========================================================================


def test_hedge_risks_adjusts_weights():
    prices = _daily_prices(symbols=("SPY", "TLT"), days=30)
    ctx = _ctx(
        prices=prices.iloc[-1],
        date=prices.index[-1],
        selected_symbols=["SPY", "TLT"],
        target_weights={"SPY": 0.80},
        price_history=prices,
    )
    algo = HedgeRisks(target_delta=0.0, hedge_symbols=["TLT"])
    d = algo(ctx)
    assert d.status == "continue"
    # TLT should now have a hedge weight assigned
    assert "TLT" in ctx.target_weights


def test_hedge_risks_no_target_weights():
    ctx = _ctx(target_weights={})
    d = HedgeRisks()(ctx)
    assert d.status == "skip_day"


def test_hedge_risks_no_history():
    ctx = _ctx(
        target_weights={"SPY": 1.0},
        selected_symbols=["TLT"],
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
    )
    d = HedgeRisks()(ctx)
    assert d.status == "skip_day"


def test_hedge_risks_in_pipeline():
    prices = _daily_prices(symbols=("SPY", "TLT"), days=30)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[
            RunMonthly(),
            SelectThese(["SPY", "TLT"]),
            WeighSpecified({"SPY": 0.80, "TLT": 0.20}),
            HedgeRisks(target_delta=0.0, hedge_symbols=["TLT"]),
            Rebalance(),
        ],
    )
    bal = bt.run()
    assert not bal.empty


# ===========================================================================
# MARGIN
# ===========================================================================


def test_margin_scales_weights():
    ctx = _ctx(
        target_weights={"SPY": 0.50},
        total_capital=10000.0,
        cash=5000.0,
    )
    algo = Margin(leverage=2.0)
    d = algo(ctx)
    assert d.status == "continue"
    assert abs(ctx.target_weights["SPY"] - 1.0) < 1e-10


def test_margin_charges_interest():
    ctx = _ctx(
        total_capital=10000.0,
        cash=3000.0,  # invested=7000, borrowed=max(0, 7000-3000)=4000
        positions={"SPY": 70.0},
        prices=pd.Series({"SPY": 100.0}),
    )
    algo = Margin(leverage=1.0, interest_rate=0.05)
    original_capital = ctx.total_capital
    algo(ctx)
    # Should have charged interest: 0.05/252 * 4000 ≈ 0.79
    assert ctx.total_capital < original_capital


def test_margin_reset():
    algo = Margin(leverage=2.0)
    algo._borrowed = 5000.0
    algo.reset()
    assert algo._borrowed == 0.0


def test_margin_call_stops():
    # equity = cash + stock_value = -400 + 500 = 100
    # exposure = stock_value = 500
    # equity/exposure = 100/500 = 0.20 < 0.25 → margin call
    ctx = _ctx(
        total_capital=100.0,
        cash=-400.0,
        positions={"SPY": 5.0},
        prices=pd.Series({"SPY": 100.0}),
    )
    algo = Margin(leverage=2.0, maintenance_pct=0.25)
    d = algo(ctx)
    assert d.status == "stop"


# ===========================================================================
# COUPON PAYING POSITION
# ===========================================================================


def test_coupon_pays_on_schedule():
    algo = CouponPayingPosition(coupon_amount=500.0, frequency="monthly")
    ctx1 = _ctx(date=pd.Timestamp("2024-01-15"), cash=10000.0, total_capital=10000.0)
    d1 = algo(ctx1)
    assert ctx1.cash == 10500.0
    assert "coupon paid" in d1.message

    # Same month → no second coupon
    ctx2 = _ctx(date=pd.Timestamp("2024-01-20"), cash=10000.0, total_capital=10000.0)
    d2 = algo(ctx2)
    assert ctx2.cash == 10000.0


def test_coupon_semi_annual_spacing():
    algo = CouponPayingPosition(coupon_amount=250.0, frequency="semi-annual")
    ctx1 = _ctx(date=pd.Timestamp("2024-01-15"), cash=10000.0, total_capital=10000.0)
    algo(ctx1)
    assert ctx1.cash == 10250.0  # first coupon

    # 3 months later → too early
    ctx2 = _ctx(date=pd.Timestamp("2024-04-15"), cash=10000.0, total_capital=10000.0)
    algo(ctx2)
    assert ctx2.cash == 10000.0

    # 6 months later → pays
    ctx3 = _ctx(date=pd.Timestamp("2024-07-15"), cash=10000.0, total_capital=10000.0)
    algo(ctx3)
    assert ctx3.cash == 10250.0


def test_coupon_stops_at_maturity():
    algo = CouponPayingPosition(
        coupon_amount=100.0, frequency="monthly", maturity_date="2024-03-01",
    )
    ctx = _ctx(date=pd.Timestamp("2024-03-15"), cash=5000.0, total_capital=5000.0)
    d = algo(ctx)
    assert d.status == "stop"
    assert ctx.cash == 5100.0  # final coupon paid


def test_coupon_before_start_date():
    algo = CouponPayingPosition(
        coupon_amount=100.0, frequency="monthly", start_date="2024-06-01",
    )
    ctx = _ctx(date=pd.Timestamp("2024-01-15"), cash=5000.0, total_capital=5000.0)
    algo(ctx)
    assert ctx.cash == 5000.0  # no payment before start


def test_coupon_invalid_frequency():
    import pytest
    with pytest.raises(ValueError, match="frequency must be one of"):
        CouponPayingPosition(coupon_amount=100.0, frequency="bi-weekly")


def test_coupon_reset():
    algo = CouponPayingPosition(coupon_amount=100.0, frequency="monthly")
    algo._last_coupon_month = (2024, 5)
    algo.reset()
    assert algo._last_coupon_month is None


# ===========================================================================
# REPLAY TRANSACTIONS
# ===========================================================================


def test_replay_buys_on_matching_date():
    blotter = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"],
        "symbol": ["SPY", "TLT"],
        "quantity": [10, 20],
    })
    algo = ReplayTransactions(blotter)
    ctx = _ctx(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 100.0, "TLT": 50.0}),
        cash=5000.0,
        positions={},
    )
    d = algo(ctx)
    assert d.status == "continue"
    assert ctx.positions["SPY"] == 10
    assert ctx.positions["TLT"] == 20
    # cash = 5000 - 10*100 - 20*50 = 5000 - 1000 - 1000 = 3000
    assert abs(ctx.cash - 3000.0) < 1e-10


def test_replay_sells():
    blotter = pd.DataFrame({
        "date": ["2024-01-02"],
        "symbol": ["SPY"],
        "quantity": [-5],
    })
    algo = ReplayTransactions(blotter)
    ctx = _ctx(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 100.0}),
        cash=0.0,
        positions={"SPY": 10.0},
    )
    algo(ctx)
    assert ctx.positions["SPY"] == 5.0
    assert abs(ctx.cash - 500.0) < 1e-10  # received 5*100


def test_replay_no_trades_on_date():
    blotter = pd.DataFrame({
        "date": ["2024-02-01"],
        "symbol": ["SPY"],
        "quantity": [10],
    })
    algo = ReplayTransactions(blotter)
    ctx = _ctx(date=pd.Timestamp("2024-01-15"), cash=5000.0, positions={})
    d = algo(ctx)
    assert d.status == "continue"
    assert ctx.positions == {}
    assert ctx.cash == 5000.0


def test_replay_closes_position_to_zero():
    blotter = pd.DataFrame({
        "date": ["2024-01-02"],
        "symbol": ["SPY"],
        "quantity": [-10],
    })
    algo = ReplayTransactions(blotter)
    ctx = _ctx(
        date=pd.Timestamp("2024-01-02"),
        prices=pd.Series({"SPY": 100.0}),
        cash=0.0,
        positions={"SPY": 10.0},
    )
    algo(ctx)
    assert "SPY" not in ctx.positions  # fully closed


def test_replay_missing_columns_raises():
    import pytest
    bad_blotter = pd.DataFrame({"date": ["2024-01-02"], "symbol": ["SPY"]})
    with pytest.raises(ValueError, match="missing columns"):
        ReplayTransactions(bad_blotter)


def test_replay_in_pipeline():
    blotter = pd.DataFrame({
        "date": ["2024-01-02"],
        "symbol": ["SPY"],
        "quantity": [5],
    })
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"SPY": [100.0, 105.0]}, index=idx)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=1000.0,
        algos=[RunDaily(), ReplayTransactions(blotter)],
    )
    bal = bt.run()
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 5


# ===========================================================================
# set_date_range on AlgoPipelineBacktester
# ===========================================================================


def test_set_date_range_returns_stats():
    prices = _daily_prices(days=60)
    bt = AlgoPipelineBacktester(
        prices=prices, initial_capital=10_000.0,
        algos=[RunMonthly(), SelectAll(), WeighEqually(), Rebalance()],
    )
    bt.run()
    stats = bt.set_date_range(start="2024-02-01")
    assert stats.total_return != 0.0 or stats.total_trades == 0
    assert hasattr(stats, "sharpe_ratio")
