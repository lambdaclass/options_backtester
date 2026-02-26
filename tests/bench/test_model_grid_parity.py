"""Exhaustive model combination grid — deterministic parametrized tests.

27-combo model grid (cost × fill × signal) and 108-combo direction/type grid.
Every combination is tested for Rust-vs-Python parity.
"""

from __future__ import annotations

import pytest

from tests.bench._parity_helpers import (
    RUST_AVAILABLE,
    DEFAULT_ALLOC,
    DEFAULT_CAPITAL,
    buy_put_strategy,
    buy_call_strategy,
    sell_put_strategy,
    sell_call_strategy,
    buy_put_spread_strategy,
    sell_call_spread_strategy,
    strangle_strategy,
    straddle_strategy,
    run_rust,
    run_python,
    assert_parity,
)

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not installed"
)


# ── Model factories ───────────────────────────────────────────────────

def _cost_model(name):
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


def _fill_model(name):
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


def _signal_selector(name):
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


# ── Strategy map ──────────────────────────────────────────────────────

_STRATEGY_MAP = {
    ("buy", "put"): buy_put_strategy,
    ("buy", "call"): buy_call_strategy,
    ("sell", "put"): sell_put_strategy,
    ("sell", "call"): sell_call_strategy,
}


# ── 27-combo model grid ──────────────────────────────────────────────

@pytest.mark.parametrize("cost_model", ["NoCosts", "PerContract", "Tiered"])
@pytest.mark.parametrize("fill_model", ["MarketAtBidAsk", "MidPrice", "VolumeAware"])
@pytest.mark.parametrize("signal_selector", ["FirstMatch", "NearestDelta", "MaxOpenInterest"])
class TestModelCombination:
    """27 deterministic model combinations for buy-put strategy."""

    def test_model_combination(self, cost_model, fill_model, signal_selector):
        cm = _cost_model(cost_model)
        fm = _fill_model(fill_model)
        ss = _signal_selector(signal_selector)

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                        cost_model=cm, fill_model=fm, signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, buy_put_strategy,
                      cost_model=cm, fill_model=fm, signal_selector=ss)
        assert_parity(py, rs,
                      label=f"grid({cost_model},{fill_model},{signal_selector})")


# ── 108-combo direction × type × model grid ──────────────────────────

@pytest.mark.parametrize("cost_model", ["NoCosts", "PerContract", "Tiered"])
@pytest.mark.parametrize("fill_model", ["MarketAtBidAsk", "MidPrice", "VolumeAware"])
@pytest.mark.parametrize("signal_selector", ["FirstMatch", "NearestDelta", "MaxOpenInterest"])
@pytest.mark.parametrize("direction", ["buy", "sell"])
@pytest.mark.parametrize("option_type", ["put", "call"])
class TestModelDirectionTypeGrid:
    """108 deterministic combos: 27 models × 4 direction/type combos."""

    def test_model_direction_type_grid(self, cost_model, fill_model,
                                       signal_selector, direction, option_type):
        cm = _cost_model(cost_model)
        fm = _fill_model(fill_model)
        ss = _signal_selector(signal_selector)
        strategy_fn = _STRATEGY_MAP[(direction, option_type)]

        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                        cost_model=cm, fill_model=fm, signal_selector=ss)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                      cost_model=cm, fill_model=fm, signal_selector=ss)
        assert_parity(
            py, rs,
            label=f"grid({direction}_{option_type},{cost_model},{fill_model},{signal_selector})",
        )


# ── Strategy-type grid (typical strategies with default models) ──────

@pytest.mark.parametrize("strategy_name,strategy_fn", [
    ("buy_put", buy_put_strategy),
    ("buy_call", buy_call_strategy),
    ("sell_put", sell_put_strategy),
    ("sell_call", sell_call_strategy),
    ("buy_put_spread", buy_put_spread_strategy),
    ("sell_call_spread", sell_call_spread_strategy),
    ("strangle", strangle_strategy),
    ("straddle", straddle_strategy),
])
class TestTypicalStrategies:
    """Test typical option strategies for parity."""

    def test_strategy_parity(self, strategy_name, strategy_fn):
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn)
        assert_parity(py, rs, label=f"strategy({strategy_name})")

    def test_strategy_with_costs(self, strategy_name, strategy_fn):
        from options_portfolio_backtester.execution.cost_model import (
            PerContractCommission,
        )
        cm = PerContractCommission(rate=0.65)
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                        cost_model=cm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                      cost_model=cm)
        assert_parity(py, rs,
                      label=f"strategy({strategy_name},PerContract)")

    def test_strategy_with_midprice(self, strategy_name, strategy_fn):
        from options_portfolio_backtester.execution.fill_model import MidPrice
        fm = MidPrice()
        py = run_python(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                        fill_model=fm)
        rs = run_rust(DEFAULT_ALLOC, DEFAULT_CAPITAL, strategy_fn,
                      fill_model=fm)
        assert_parity(py, rs,
                      label=f"strategy({strategy_name},MidPrice)")
