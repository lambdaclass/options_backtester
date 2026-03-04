"""Risk management — constraints checked before entering positions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from options_portfolio_backtester.core.types import Greeks
from options_portfolio_backtester.execution._rust_bridge import rust_risk_check


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


def _greeks_list(g: Greeks) -> list[float]:
    return [g.delta, g.gamma, g.theta, g.vega]


class MaxDelta(RiskConstraint):
    """Reject trades that would push portfolio delta beyond a limit."""

    def __init__(self, limit: float = 100.0) -> None:
        self.limit = limit

    def check(self, current_greeks: Greeks, proposed_greeks: Greeks,
              portfolio_value: float, peak_value: float) -> bool:
        return rust_risk_check(
            "MaxDelta", self.limit,
            _greeks_list(current_greeks), _greeks_list(proposed_greeks),
            portfolio_value, peak_value,
        )

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
        return rust_risk_check(
            "MaxVega", self.limit,
            _greeks_list(current_greeks), _greeks_list(proposed_greeks),
            portfolio_value, peak_value,
        )

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
        return rust_risk_check(
            "MaxDrawdown", self.max_dd_pct,
            _greeks_list(current_greeks), _greeks_list(proposed_greeks),
            portfolio_value, peak_value,
        )

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
