"""Dispatch layer for Rust acceleration.

The Rust extension (_ob_rust) is always available. The `use_rust()` gate
checks whether the full Rust backtest loop can be used (requires polars).
The `rust` proxy provides attribute access to the _ob_rust module.

Usage in engine.py:
    from options_portfolio_backtester.engine._dispatch import rust

    result = rust.run_backtest_py(...)
    result = rust.update_balance(...)
"""

from __future__ import annotations

from options_portfolio_backtester import _ob_rust


def use_rust() -> bool:
    """Return True — Rust extension is always available."""
    return True


class _RustProxy:
    """Proxy to the Rust module for attribute access."""

    def __getattr__(self, name: str):
        return getattr(_ob_rust, name)


rust = _RustProxy()
