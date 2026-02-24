from __future__ import annotations

import pandas as pd
import numpy as np

from options_backtester.engine.pipeline import (
    AlgoPipelineBacktester,
    MaxDrawdownGuard,
    PipelineContext,
    Rebalance,
    RunMonthly,
    SelectThese,
    StepDecision,
    WeighSpecified,
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
