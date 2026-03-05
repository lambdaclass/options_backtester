"""Dispatch layer for Rust backend.

The Rust extension (_ob_rust) is a hard dependency — all execution paths
require it.  The `rust` proxy provides attribute access to the module.

Usage in engine.py:
    from options_portfolio_backtester.engine._dispatch import rust

    result = rust.run_backtest_py(...)
    result = rust.update_balance(...)
"""

from __future__ import annotations

from options_portfolio_backtester import _ob_rust

RUST_AVAILABLE = True


def use_rust() -> bool:
    """Return True — Rust extension is always required."""
    return True


class _RustProxy:
    """Proxy to the Rust module for attribute access."""

    def __getattr__(self, name: str):
        return getattr(_ob_rust, name)


rust = _RustProxy()
