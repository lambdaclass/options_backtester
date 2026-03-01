"""Convexity scanner: cross-asset tail protection scoring and allocation."""

from options_portfolio_backtester.convexity.allocator import (
    allocate_equal_weight,
    allocate_inverse_vol,
    pick_cheapest,
)
from options_portfolio_backtester.convexity.backtest import (
    BacktestResult,
    run_backtest,
    run_unhedged,
)
from options_portfolio_backtester.convexity.config import (
    BacktestConfig,
    InstrumentConfig,
    default_config,
)
from options_portfolio_backtester.convexity.scoring import compute_convexity_scores

__all__ = [
    "InstrumentConfig",
    "BacktestConfig",
    "default_config",
    "compute_convexity_scores",
    "BacktestResult",
    "run_backtest",
    "run_unhedged",
    "pick_cheapest",
    "allocate_equal_weight",
    "allocate_inverse_vol",
]
