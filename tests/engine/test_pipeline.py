from __future__ import annotations

import pandas as pd

from options_backtester.engine.pipeline import (
    AlgoPipelineBacktester,
    MaxDrawdownGuard,
    Rebalance,
    RunMonthly,
    SelectThese,
    WeighSpecified,
)


def _prices() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    return pd.DataFrame({"SPY": [100.0, 102.0, 101.0, 103.0]}, index=idx)


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
    # First month buys 10 shares at 100.
    assert bal.loc[pd.Timestamp("2024-01-02"), "SPY qty"] == 10
    # At Feb drawdown > 20%, guard blocks rebalance; qty remains unchanged.
    assert bal.loc[pd.Timestamp("2024-02-01"), "SPY qty"] == 10
    logs = bt.logs_dataframe()
    feb = logs[(logs["date"] == pd.Timestamp("2024-02-01")) & (logs["step"] == "MaxDrawdownGuard")]
    assert not feb.empty
    assert feb.iloc[0]["status"] == "skip_day"
