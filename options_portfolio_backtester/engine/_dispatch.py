"""Dispatch layer for optional Rust acceleration.

When the Rust extension (_ob_rust) is available, hot-path functions are
dispatched to compiled Rust code via PyO3. When unavailable, falls back
transparently to the pure-Python implementation. Zero API change for users.

Usage in engine.py:
    from options_portfolio_backtester.engine._dispatch import use_rust, rust

    if use_rust():
        result = rust.update_balance(...)
    else:
        result = _python_update_balance(...)
"""

from __future__ import annotations

RUST_AVAILABLE: bool = False
_rust_module = None

try:
    from options_portfolio_backtester import _ob_rust

    _rust_module = _ob_rust
    RUST_AVAILABLE = True
except ImportError:
    pass


def use_rust() -> bool:
    """Check if Rust acceleration is available."""
    return RUST_AVAILABLE


class _RustProxy:
    """Lazy proxy to the Rust module — avoids ImportError at attribute access."""

    def __getattr__(self, name: str):
        if _rust_module is None:
            raise RuntimeError(
                "Rust extension not available. Install with: "
                "maturin develop --manifest-path rust/ob_python/Cargo.toml --release"
            )
        return getattr(_rust_module, name)


rust = _RustProxy()
