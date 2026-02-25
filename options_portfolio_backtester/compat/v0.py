"""Backward-compatible Backtest class wrapping the new BacktestEngine.

Usage:
    from options_portfolio_backtester.compat.v0 import Backtest, Stock

This provides the exact same API as the original backtester.backtester.Backtest,
backed by BacktestEngine internals.
"""

from __future__ import annotations

from options_portfolio_backtester.engine.engine import BacktestEngine
from backtester.enums import Stock  # noqa: F401


class Backtest(BacktestEngine):
    """Drop-in replacement for backtester.Backtest using the new engine."""
    pass
