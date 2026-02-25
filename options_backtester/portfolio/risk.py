"""Risk management — constraints checked before entering positions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from options_backtester.core.types import Greeks


class RiskConstraint(ABC):
    """A single risk constraint."""

    @abstractmethod
    def check(self, current_greeks: Greeks, proposed_greeks: Greeks,
              portfolio_value: float, peak_value: float) -> bool:
        """Return True if the trade is allowed, False if it violates the constraint."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the constraint."""
        ...


class MaxDelta(RiskConstraint):
    """Reject trades that would push portfolio delta beyond a limit."""

    def __init__(self, limit: float = 100.0) -> None:
        self.limit = limit

    def check(self, current_greeks: Greeks, proposed_greeks: Greeks,
              portfolio_value: float, peak_value: float) -> bool:
        new_delta = current_greeks.delta + proposed_greeks.delta
        return abs(new_delta) <= self.limit

    def describe(self) -> str:
        return f"MaxDelta(limit={self.limit})"

    def to_rust_config(self) -> dict:
        return {"type": "MaxDelta", "limit": self.limit}


class MaxVega(RiskConstraint):
    """Reject trades that would push portfolio vega beyond a limit."""

    def __init__(self, limit: float = 50.0) -> None:
        self.limit = limit

    def check(self, current_greeks: Greeks, proposed_greeks: Greeks,
              portfolio_value: float, peak_value: float) -> bool:
        new_vega = current_greeks.vega + proposed_greeks.vega
        return abs(new_vega) <= self.limit

    def describe(self) -> str:
        return f"MaxVega(limit={self.limit})"

    def to_rust_config(self) -> dict:
        return {"type": "MaxVega", "limit": self.limit}


class MaxDrawdown(RiskConstraint):
    """Reject new entries if portfolio drawdown exceeds a threshold."""

    def __init__(self, max_dd_pct: float = 0.20) -> None:
        self.max_dd_pct = max_dd_pct

    def check(self, current_greeks: Greeks, proposed_greeks: Greeks,
              portfolio_value: float, peak_value: float) -> bool:
        if peak_value <= 0:
            return True
        dd = (peak_value - portfolio_value) / peak_value
        return dd < self.max_dd_pct

    def describe(self) -> str:
        return f"MaxDrawdown(max_dd_pct={self.max_dd_pct})"

    def to_rust_config(self) -> dict:
        return {"type": "MaxDrawdown", "max_dd_pct": self.max_dd_pct}


class RiskManager:
    """Evaluates a set of risk constraints before allowing a trade."""

    def __init__(self, constraints: list[RiskConstraint] | None = None) -> None:
        self.constraints = constraints or []

    def add_constraint(self, constraint: RiskConstraint) -> None:
        self.constraints.append(constraint)

    def is_allowed(self, current_greeks: Greeks, proposed_greeks: Greeks,
                   portfolio_value: float, peak_value: float) -> tuple[bool, str]:
        """Check all constraints. Returns (allowed, reason)."""
        for c in self.constraints:
            if not c.check(current_greeks, proposed_greeks,
                          portfolio_value, peak_value):
                return False, c.describe()
        return True, ""
