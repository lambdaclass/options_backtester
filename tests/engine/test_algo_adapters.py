from __future__ import annotations

import pandas as pd

from options_backtester.engine.algo_adapters import (
    BudgetPercent,
    EngineRunMonthly,
    SelectByDTE,
)
from options_backtester.engine.engine import BacktestEngine

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


def test_engine_algo_monthly_gate_logs_skip():
    engine = _run_with_algos([EngineRunMonthly()])
    logs = engine.events_dataframe()
    assert not logs.empty
    assert (logs["event"] == "algo_step").any()
    assert (logs["status"] == "skip_day").any()


def test_budget_percent_zero_blocks_option_entries():
    engine = _run_with_algos([BudgetPercent(0.0)])
    if engine.trade_log.empty:
        assert True
        return
    assert (engine.trade_log["totals"]["qty"] > 0).sum() == 0


def test_select_by_dte_strict_filter_skips_candidates():
    engine = _run_with_algos([SelectByDTE(min_dte=0, max_dte=1)])
    events = engine.events_dataframe()
    assert isinstance(events, pd.DataFrame)
    assert ((events["event"] == "option_entry_no_candidates") | (events["event"] == "option_entry_filtered")).any()
