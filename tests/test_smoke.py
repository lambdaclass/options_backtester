"""Smoke tests — verify all public imports work."""


def test_top_level_imports():
    """All public symbols importable from options_portfolio_backtester."""
    from options_portfolio_backtester import (
        # Engine
        BacktestEngine,
        # Analytics
        BacktestStats,
        Direction,
    )
    # Quick sanity: verify some aren't None
    assert Direction is not None
    assert BacktestEngine is not None
    assert BacktestStats is not None


def test_strategy_presets_import():
    """Strategy presets importable."""
    from options_portfolio_backtester.strategy.presets import (
        butterfly,
        cash_secured_put,
        collar,
        covered_call,
        iron_condor,
        strangle,
    )
    assert callable(strangle)
    assert callable(iron_condor)
    assert callable(covered_call)
    assert callable(cash_secured_put)
    assert callable(collar)
    assert callable(butterfly)
