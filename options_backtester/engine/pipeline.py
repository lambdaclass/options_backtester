"""Composable algo pipeline for stock portfolio workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd


StepStatus = Literal["continue", "skip_day", "stop"]


@dataclass(frozen=True)
class StepDecision:
    """Outcome returned by a pipeline step."""

    status: StepStatus = "continue"
    message: str = ""


@dataclass
class PipelineContext:
    """Mutable state shared across pipeline steps for one date."""

    date: pd.Timestamp
    prices: pd.Series
    total_capital: float
    cash: float
    positions: dict[str, float]
    selected_symbols: list[str] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineLogRow:
    date: pd.Timestamp
    step: str
    status: StepStatus
    message: str


class Algo(Protocol):
    """Protocol for a pipeline step."""

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        ...


class RunMonthly:
    """Gate pipeline execution to month starts."""

    def __init__(self) -> None:
        self._last_month: tuple[int, int] | None = None

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        key = (ctx.date.year, ctx.date.month)
        if self._last_month == key:
            return StepDecision(status="skip_day", message="not month-start")
        self._last_month = key
        return StepDecision()


class SelectThese:
    """Select a fixed list of symbols if priced on current date."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = [s.upper() for s in symbols]

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        available = [s for s in self.symbols if s in ctx.prices.index and pd.notna(ctx.prices[s])]
        ctx.selected_symbols = available
        if not available:
            return StepDecision(status="skip_day", message="no selected symbols with valid prices")
        return StepDecision()


class WeighSpecified:
    """Set fixed target weights, normalized over selected symbols."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = {k.upper(): float(v) for k, v in weights.items()}

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        raw = {s: self.weights.get(s, 0.0) for s in ctx.selected_symbols}
        total = float(sum(raw.values()))
        if total <= 0:
            return StepDecision(status="skip_day", message="target weights sum to zero")
        ctx.target_weights = {s: w / total for s, w in raw.items()}
        return StepDecision()


class MaxDrawdownGuard:
    """Block new rebalances while drawdown exceeds threshold."""

    def __init__(self, max_drawdown_pct: float) -> None:
        self.max_drawdown_pct = float(max_drawdown_pct)
        self._peak = 0.0

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        self._peak = max(self._peak, float(ctx.total_capital))
        if self._peak <= 0:
            return StepDecision()
        dd = (self._peak - float(ctx.total_capital)) / self._peak
        if dd > self.max_drawdown_pct:
            return StepDecision(status="skip_day", message=f"drawdown {dd:.2%} > {self.max_drawdown_pct:.2%}")
        return StepDecision()


class Rebalance:
    """Rebalance positions to target weights at current prices."""

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.target_weights:
            return StepDecision(status="skip_day", message="no target weights")
        new_positions: dict[str, float] = {}
        spent = 0.0
        for sym, w in ctx.target_weights.items():
            price = float(ctx.prices[sym])
            if price <= 0:
                continue
            target_value = float(ctx.total_capital) * w
            qty = float(np.floor(target_value / price))
            new_positions[sym] = qty
            spent += qty * price

        ctx.positions.clear()
        ctx.positions.update(new_positions)
        ctx.cash = float(ctx.total_capital - spent)
        return StepDecision()


class AlgoPipelineBacktester:
    """Simple stock backtester driven by composable pipeline algos."""

    def __init__(
        self,
        prices: pd.DataFrame,
        algos: list[Algo],
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self.prices = prices.sort_index()
        self.algos = algos
        self.initial_capital = float(initial_capital)
        self.logs: list[PipelineLogRow] = []

    def run(self) -> pd.DataFrame:
        cash = float(self.initial_capital)
        positions: dict[str, float] = {}
        rows: list[dict[str, float | pd.Timestamp]] = []

        for date, price_row in self.prices.iterrows():
            stocks_cap = float(sum(float(qty) * float(price_row.get(sym, np.nan))
                                   for sym, qty in positions.items()
                                   if sym in price_row.index and pd.notna(price_row[sym])))
            total_cap = cash + stocks_cap
            ctx = PipelineContext(
                date=pd.Timestamp(date),
                prices=price_row,
                total_capital=total_cap,
                cash=cash,
                positions=dict(positions),
            )

            stop_all = False
            for algo in self.algos:
                decision = algo(ctx)
                self.logs.append(
                    PipelineLogRow(
                        date=pd.Timestamp(date),
                        step=algo.__class__.__name__,
                        status=decision.status,
                        message=decision.message,
                    )
                )
                if decision.status == "skip_day":
                    break
                if decision.status == "stop":
                    stop_all = True
                    break

            cash = float(ctx.cash)
            positions = dict(ctx.positions)
            stocks_cap = float(sum(float(qty) * float(price_row.get(sym, np.nan))
                                   for sym, qty in positions.items()
                                   if sym in price_row.index and pd.notna(price_row[sym])))
            total_cap = cash + stocks_cap
            row: dict[str, float | pd.Timestamp] = {
                "date": pd.Timestamp(date),
                "cash": cash,
                "stocks capital": stocks_cap,
                "total capital": total_cap,
            }
            for sym, qty in positions.items():
                row[f"{sym} qty"] = float(qty)
            rows.append(row)

            if stop_all:
                break

        balance = pd.DataFrame(rows).set_index("date")
        if not balance.empty:
            balance["% change"] = balance["total capital"].pct_change()
            balance["accumulated return"] = (1.0 + balance["% change"]).cumprod()
        self.balance = balance
        return balance

    def logs_dataframe(self) -> pd.DataFrame:
        if not self.logs:
            return pd.DataFrame(columns=["date", "step", "status", "message"])
        return pd.DataFrame([{
            "date": r.date,
            "step": r.step,
            "status": r.status,
            "message": r.message,
        } for r in self.logs])
