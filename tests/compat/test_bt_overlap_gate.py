from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_with_bt import normalize_weights, run_bt, run_options_backtester


@pytest.mark.bench
def test_bt_overlap_gate_stock_only():
    stocks_file = Path("data/processed/stocks.csv")
    if not stocks_file.exists():
        pytest.skip("stocks.csv is not available")

    bt_available = True
    try:
        import bt  # noqa: F401
    except Exception:
        bt_available = False
    if not bt_available:
        pytest.skip("bt is not installed")

    symbols = ["SPY"]
    weights = normalize_weights(symbols, None)
    ob = run_options_backtester(
        stocks_file=str(stocks_file),
        symbols=symbols,
        weights=weights,
        initial_capital=1_000_000.0,
        rebalance_months=1,
        runs=1,
    )
    bt_res = run_bt(
        stocks_file=str(stocks_file),
        symbols=symbols,
        weights=weights,
        initial_capital=1_000_000.0,
        runs=1,
    )
    assert bt_res is not None
    common = ob.equity.index.intersection(bt_res.equity.index)
    assert len(common) > 200
    ob_n = ob.equity.loc[common] / ob.equity.loc[common].iloc[0]
    bt_n = bt_res.equity.loc[common] / bt_res.equity.loc[common].iloc[0]
    delta = (ob_n - bt_n).abs().max()
    assert float(delta) < 0.10
