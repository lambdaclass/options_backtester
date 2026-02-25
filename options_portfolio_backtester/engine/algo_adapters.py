"""Algo adapter layer to drive BacktestEngine with bt-style pipeline blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd

from options_portfolio_backtester.core.types import Greeks


StepStatus = Literal["continue", "skip_day", "stop"]


@dataclass(frozen=True)
class EngineStepDecision:
    """Decision emitted by one engine-algo step."""

    status: StepStatus = "continue"
    message: str = ""


@dataclass
class EnginePipelineContext:
    """Mutable run context shared by all engine algo steps for one rebalance date."""

    date: pd.Timestamp
    stocks: pd.DataFrame
    options: pd.DataFrame
    total_capital: float
    current_cash: float
    current_greeks: Greeks
    options_allocation: float
    entry_filters: list = field(default_factory=list)
    exit_threshold_override: tuple[float, float] | None = None


class EngineAlgo(Protocol):
    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        ...


class EngineRunMonthly:
    """Allow rebalances only on first rebalance day per month."""

    def __init__(self) -> None:
        self._last_month: tuple[int, int] | None = None

    def reset(self) -> None:
        self._last_month = None

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        key = (ctx.date.year, ctx.date.month)
        if self._last_month == key:
            return EngineStepDecision(status="skip_day", message="not month-start")
        self._last_month = key
        return EngineStepDecision()


class BudgetPercent:
    """Set options allocation budget as percent of current total capital."""

    def __init__(self, pct: float) -> None:
        self.pct = float(pct)

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        ctx.options_allocation = max(0.0, float(ctx.total_capital) * self.pct)
        return EngineStepDecision()


class RangeFilter:
    """Keep contracts where *column* falls within [min_val, max_val].

    Generic building block — use directly or via the convenience aliases
    ``SelectByDelta``, ``SelectByDTE``, ``IVRankFilter``.
    """

    def __init__(self, column: str, min_val: float, max_val: float) -> None:
        self.column = column
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        lo, hi, col = self.min_val, self.max_val, self.column

        def _flt(df: pd.DataFrame) -> pd.Series:
            if col not in df.columns:
                return pd.Series(True, index=df.index)
            v = df[col]
            return (v >= lo) & (v <= hi)

        ctx.entry_filters.append(_flt)
        return EngineStepDecision()


def SelectByDelta(min_delta: float = -1.0, max_delta: float = 1.0, column: str = "delta") -> RangeFilter:
    """Keep contracts with delta within [min_delta, max_delta]."""
    return RangeFilter(column=column, min_val=min_delta, max_val=max_delta)


def SelectByDTE(min_dte: int = 0, max_dte: int = 10_000, column: str = "dte") -> RangeFilter:
    """Keep contracts with DTE within [min_dte, max_dte]."""
    return RangeFilter(column=column, min_val=float(min_dte), max_val=float(max_dte))


def IVRankFilter(min_rank: float = 0.0, max_rank: float = 1.0, column: str = "iv_rank") -> RangeFilter:
    """Keep contracts with IV rank within [min_rank, max_rank]."""
    return RangeFilter(column=column, min_val=min_rank, max_val=max_rank)


class MaxGreekExposure:
    """Skip new entries when current absolute greek exposure exceeds limits."""

    def __init__(
        self,
        max_abs_delta: float | None = None,
        max_abs_vega: float | None = None,
    ) -> None:
        self.max_abs_delta = float(max_abs_delta) if max_abs_delta is not None else None
        self.max_abs_vega = float(max_abs_vega) if max_abs_vega is not None else None

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        if self.max_abs_delta is not None and abs(float(ctx.current_greeks.delta)) > self.max_abs_delta:
            return EngineStepDecision(
                status="skip_day",
                message=f"|delta|>{self.max_abs_delta}",
            )
        if self.max_abs_vega is not None and abs(float(ctx.current_greeks.vega)) > self.max_abs_vega:
            return EngineStepDecision(
                status="skip_day",
                message=f"|vega|>{self.max_abs_vega}",
            )
        return EngineStepDecision()


class ExitOnThreshold:
    """Override strategy exit profit/loss thresholds for this run.

    At least one of *profit_pct* or *loss_pct* must be finite, otherwise the
    algo is a no-op and likely a caller mistake.
    """

    def __init__(self, profit_pct: float = float("inf"), loss_pct: float = float("inf")) -> None:
        self.profit_pct = float(profit_pct)
        self.loss_pct = float(loss_pct)
        if math.isinf(self.profit_pct) and math.isinf(self.loss_pct):
            import warnings
            warnings.warn(
                "ExitOnThreshold created with both thresholds infinite — "
                "exit overrides will have no effect",
                stacklevel=2,
            )

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        ctx.exit_threshold_override = (self.profit_pct, self.loss_pct)
        return EngineStepDecision()
