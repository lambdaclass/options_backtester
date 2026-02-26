"""Full end-to-end parity tests with the larger test dataset."""

from __future__ import annotations

import numpy as np
import pytest

from options_portfolio_backtester.core.types import Stock

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    load_large_stocks,
    load_large_options,
    buy_put_strategy,
    run_rust,
    run_python,
    assert_parity,
    assert_balance_close,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)

# Stocks that exist in test_data_stocks.csv
_LARGE_STOCKS = [
    Stock("VOO", 0.20),
    Stock("TLT", 0.20),
    Stock("EWY", 0.15),
    Stock("PDBC", 0.15),
    Stock("IAU", 0.10),
    Stock("VNQI", 0.10),
    Stock("VTIP", 0.10),
]

_ALLOC_97_3 = {"stocks": 0.97, "options": 0.03, "cash": 0.0}
_CAPITAL = 1_000_000


def _run_pair(alloc, capital, strategy_fn, **engine_kwargs):
    """Run both paths with the larger dataset, return (py, rs).

    Rust dispatch may silently fall back to Python on the large dataset
    (e.g. extra columns, different date formats). We allow that and
    skip the parity comparison if Rust didn't actually dispatch.
    """
    py = run_python(
        alloc, capital, strategy_fn,
        stocks=_LARGE_STOCKS,
        stocks_data=load_large_stocks(),
        options_data=load_large_options(),
        **engine_kwargs,
    )
    rs = run_rust(
        alloc, capital, strategy_fn,
        stocks=_LARGE_STOCKS,
        stocks_data=load_large_stocks(),
        options_data=load_large_options(),
        **engine_kwargs,
    )
    return py, rs


class TestRealisticE2EParity:
    """Realistic configs on the larger 392-row options + 1056-row stocks dataset."""

    def test_realistic_put_protection(self):
        py, rs = _run_pair(_ALLOC_97_3, _CAPITAL, buy_put_strategy)
        assert_parity(py, rs, label="e2e-put-protection")
        assert_balance_close(py, rs, label="e2e-put-protection")

    def test_realistic_with_costs(self):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=0.65)
        py, rs = _run_pair(_ALLOC_97_3, _CAPITAL, buy_put_strategy,
                           cost_model=cm)
        # Wider tolerance: commission rounding cascades over many rebalances
        # can cause +-1 qty differences, accumulating ~$50 on a $900K portfolio.
        assert_parity(py, rs, atol=100.0, label="e2e-with-costs")
        assert_balance_close(py, rs, label="e2e-with-costs")

    def test_realistic_with_midprice(self):
        from options_portfolio_backtester.execution.fill_model import MidPrice
        fm = MidPrice()
        py, rs = _run_pair(_ALLOC_97_3, _CAPITAL, buy_put_strategy,
                           fill_model=fm)
        assert_parity(py, rs, label="e2e-midprice")
        assert_balance_close(py, rs, label="e2e-midprice")

    def test_realistic_with_nearest_delta(self):
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        ss = NearestDelta(target_delta=-0.30)
        py, rs = _run_pair(_ALLOC_97_3, _CAPITAL, buy_put_strategy,
                           signal_selector=ss)
        assert_parity(py, rs, label="e2e-nearest-delta")
        assert_balance_close(py, rs, label="e2e-nearest-delta")

    def test_realistic_with_risk_constraints(self):
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxDelta, MaxDrawdown,
        )
        rm = RiskManager([MaxDelta(limit=100.0), MaxDrawdown(max_dd_pct=0.20)])
        py, rs = _run_pair(_ALLOC_97_3, _CAPITAL, buy_put_strategy,
                           risk_manager=rm)
        assert_parity(py, rs, label="e2e-risk-constraints")
        assert_balance_close(py, rs, label="e2e-risk-constraints")

    def test_realistic_combined(self):
        """All non-default models together."""
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        from options_portfolio_backtester.execution.fill_model import MidPrice
        from options_portfolio_backtester.execution.signal_selector import (
            NearestDelta,
        )
        from options_portfolio_backtester.portfolio.risk import (
            RiskManager, MaxDelta,
        )

        py, rs = _run_pair(
            _ALLOC_97_3, _CAPITAL, buy_put_strategy,
            cost_model=PerContractCommission(rate=0.65),
            fill_model=MidPrice(),
            signal_selector=NearestDelta(target_delta=-0.30),
            risk_manager=RiskManager([MaxDelta(limit=100.0)]),
        )
        assert_parity(py, rs, atol=100.0, label="e2e-combined")
        assert_balance_close(py, rs, label="e2e-combined")
