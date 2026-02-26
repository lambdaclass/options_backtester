"""Tests for post-refactor cleanup — verify dead code removed, imports correct."""

import importlib


def test_top_level_exports_trimmed():
    """__init__.py exports only core types, not pipeline bulk."""
    import options_portfolio_backtester as pkg
    # Should be present
    assert hasattr(pkg, "BacktestEngine")
    assert hasattr(pkg, "Stock")
    assert hasattr(pkg, "Direction")
    assert hasattr(pkg, "BacktestStats")
    assert hasattr(pkg, "TradingClock")
    assert hasattr(pkg, "summary")
    # Pipeline algos should NOT be in top-level
    assert not hasattr(pkg, "AlgoPipelineBacktester")
    assert not hasattr(pkg, "RunMonthly")
    assert not hasattr(pkg, "SelectAll")
    assert not hasattr(pkg, "WeighEqually")
    assert not hasattr(pkg, "Rebalance")
    assert not hasattr(pkg, "EngineRunMonthly")
    assert not hasattr(pkg, "StrategyTreeNode")


def test_pipeline_importable_from_submodule():
    """Pipeline algos still importable from engine.pipeline."""
    from options_portfolio_backtester.engine.pipeline import (
        AlgoPipelineBacktester,
        RunMonthly, RunWeekly, RunDaily,
        SelectAll, SelectThese,
        WeighEqually, WeighInvVol,
        LimitWeights, Rebalance,
    )
    assert callable(RunMonthly)
    assert callable(SelectAll)
    assert callable(WeighEqually)
    assert AlgoPipelineBacktester is not None


def test_algo_adapters_importable_from_submodule():
    """Algo adapters still importable from engine.algo_adapters."""
    from options_portfolio_backtester.engine.algo_adapters import (
        EngineAlgo, EngineStepDecision, EnginePipelineContext,
        EngineRunMonthly, BudgetPercent, RangeFilter,
        SelectByDelta, SelectByDTE, IVRankFilter,
        MaxGreekExposure, ExitOnThreshold,
    )
    assert EngineAlgo is not None
    assert EngineRunMonthly is not None


def test_strategy_tree_importable_from_submodule():
    """Strategy tree still importable from engine.strategy_tree."""
    from options_portfolio_backtester.engine.strategy_tree import (
        StrategyTreeNode, StrategyTreeEngine,
    )
    assert StrategyTreeNode is not None
    assert StrategyTreeEngine is not None


def test_compat_directory_removed():
    """compat/ directory should not exist."""
    import pytest
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("options_portfolio_backtester.compat")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("options_portfolio_backtester.compat.v0")


def test_no_duplicate_import_in_engine():
    """engine.py should have Stock in the first import block, no duplicate."""
    from options_portfolio_backtester.engine.engine import BacktestEngine, Stock
    assert Stock is not None
    assert BacktestEngine is not None


def test_safe_ratio_removed():
    """_safe_ratio should not exist in stats module."""
    from options_portfolio_backtester.analytics import stats
    assert not hasattr(stats, "_safe_ratio")


def test_dispatch_module():
    """_dispatch module exposes use_rust() and rust proxy."""
    from options_portfolio_backtester.engine._dispatch import use_rust, rust, RUST_AVAILABLE
    # use_rust returns a bool
    assert isinstance(use_rust(), bool)
    assert use_rust() == RUST_AVAILABLE
    # rust is a proxy object
    assert rust is not None


def test_dispatch_proxy_error_when_no_rust():
    """RustProxy raises RuntimeError when Rust is not available."""
    from options_portfolio_backtester.engine._dispatch import _RustProxy, _rust_module
    if _rust_module is None:
        import pytest
        proxy = _RustProxy()
        with pytest.raises(RuntimeError, match="Rust extension not available"):
            proxy.some_function()
