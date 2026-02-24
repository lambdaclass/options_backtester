"""Algo adapter layer to drive BacktestEngine with bt-style pipeline blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd

from options_backtester.core.types import Greeks


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


class SelectByDelta:
    """Keep contracts with delta within [min_delta, max_delta]."""

    def __init__(self, min_delta: float = -1.0, max_delta: float = 1.0, column: str = "delta") -> None:
        self.min_delta = float(min_delta)
        self.max_delta = float(max_delta)
        self.column = column

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        def _flt(df: pd.DataFrame) -> pd.Series:
            if self.column not in df.columns:
                return pd.Series(True, index=df.index)
            v = df[self.column]
            return (v >= self.min_delta) & (v <= self.max_delta)

        ctx.entry_filters.append(_flt)
        return EngineStepDecision()


class SelectByDTE:
    """Keep contracts with DTE within [min_dte, max_dte]."""

    def __init__(self, min_dte: int = 0, max_dte: int = 10_000, column: str = "dte") -> None:
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)
        self.column = column

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        def _flt(df: pd.DataFrame) -> pd.Series:
            if self.column not in df.columns:
                return pd.Series(True, index=df.index)
            v = df[self.column]
            return (v >= self.min_dte) & (v <= self.max_dte)

        ctx.entry_filters.append(_flt)
        return EngineStepDecision()


class IVRankFilter:
    """Keep contracts with IV rank within [min_rank, max_rank]."""

    def __init__(self, min_rank: float = 0.0, max_rank: float = 1.0, column: str = "iv_rank") -> None:
        self.min_rank = float(min_rank)
        self.max_rank = float(max_rank)
        self.column = column

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        def _flt(df: pd.DataFrame) -> pd.Series:
            if self.column not in df.columns:
                return pd.Series(True, index=df.index)
            v = df[self.column]
            return (v >= self.min_rank) & (v <= self.max_rank)

        ctx.entry_filters.append(_flt)
        return EngineStepDecision()


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
    """Override strategy exit profit/loss thresholds for this run."""

    def __init__(self, profit_pct: float = float("inf"), loss_pct: float = float("inf")) -> None:
        self.profit_pct = float(profit_pct)
        self.loss_pct = float(loss_pct)

    def __call__(self, ctx: EnginePipelineContext) -> EngineStepDecision:
        ctx.exit_threshold_override = (self.profit_pct, self.loss_pct)
        return EngineStepDecision()
