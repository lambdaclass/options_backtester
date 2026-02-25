"""Filter DSL — preserved from original codebase.

Only addition: Filter.to_dict() for future Rust compilation.
"""

# Re-export the original schema module verbatim.
# The original implementation in backtester.datahandler.schema is the
# canonical source; we re-export here so new code can import from
# options_portfolio_backtester.data.schema.

from backtester.datahandler.schema import Schema, Field, Filter  # noqa: F401

__all__ = ["Schema", "Field", "Filter"]
