"""Composable algo pipeline for stock portfolio workflows.

Provides bt-compatible scheduling, selection, weighting, and rebalancing algos.
"""

from __future__ import annotations

import re as _re
import random as _random
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Sequence

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
    # Price history up to current date (set by AlgoPipelineBacktester).
    price_history: pd.DataFrame | None = None


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


# ---------------------------------------------------------------------------
# Scheduling algos
# ---------------------------------------------------------------------------

class RunMonthly:
    """Gate pipeline execution to month starts."""

    def __init__(self) -> None:
        self._last_month: tuple[int, int] | None = None

    def reset(self) -> None:
        self._last_month = None

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        key = (ctx.date.year, ctx.date.month)
        if self._last_month == key:
            return StepDecision(status="skip_day", message="not month-start")
        self._last_month = key
        return StepDecision()


class RunWeekly:
    """Gate pipeline execution to week starts (Monday)."""

    def __init__(self) -> None:
        self._last_week: tuple[int, int] | None = None

    def reset(self) -> None:
        self._last_week = None

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        key = (ctx.date.isocalendar()[0], ctx.date.isocalendar()[1])
        if self._last_week == key:
            return StepDecision(status="skip_day", message="not week-start")
        self._last_week = key
        return StepDecision()


class RunQuarterly:
    """Gate pipeline execution to quarter starts."""

    def __init__(self) -> None:
        self._last_quarter: tuple[int, int] | None = None

    def reset(self) -> None:
        self._last_quarter = None

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        key = (ctx.date.year, (ctx.date.month - 1) // 3)
        if self._last_quarter == key:
            return StepDecision(status="skip_day", message="not quarter-start")
        self._last_quarter = key
        return StepDecision()


class RunYearly:
    """Gate pipeline execution to year starts."""

    def __init__(self) -> None:
        self._last_year: int | None = None

    def reset(self) -> None:
        self._last_year = None

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if self._last_year == ctx.date.year:
            return StepDecision(status="skip_day", message="not year-start")
        self._last_year = ctx.date.year
        return StepDecision()


class RunDaily:
    """Allow pipeline execution on every date (no gating)."""

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        return StepDecision()


class RunOnce:
    """Execute pipeline only on the first date, skip all subsequent dates."""

    def __init__(self) -> None:
        self._ran = False

    def reset(self) -> None:
        self._ran = False

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if self._ran:
            return StepDecision(status="skip_day", message="already ran")
        self._ran = True
        return StepDecision()


class RunOnDate:
    """Execute pipeline only on specific dates."""

    def __init__(self, dates: Sequence[str | pd.Timestamp]) -> None:
        self._dates = {pd.Timestamp(d).normalize() for d in dates}

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if ctx.date.normalize() not in self._dates:
            return StepDecision(status="skip_day", message="not a target date")
        return StepDecision()


class RunAfterDate:
    """Execute pipeline only after a specific date (inclusive)."""

    def __init__(self, date: str | pd.Timestamp) -> None:
        self._date = pd.Timestamp(date).normalize()

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if ctx.date.normalize() < self._date:
            return StepDecision(status="skip_day", message="before start date")
        return StepDecision()


class RunEveryNPeriods:
    """Execute pipeline every N trading days."""

    def __init__(self, n: int) -> None:
        self._n = int(n)
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        self._count += 1
        if self._count % self._n != 1 and self._count != 1:
            return StepDecision(status="skip_day", message=f"period {self._count}, not every {self._n}")
        return StepDecision()


class RunAfterDays:
    """Warmup gate: skip the first *n* trading days."""

    def __init__(self, n: int) -> None:
        self._n = int(n)
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        self._count += 1
        if self._count <= self._n:
            return StepDecision(status="skip_day", message=f"warmup day {self._count}/{self._n}")
        return StepDecision()


class RunIfOutOfBounds:
    """Trigger rebalance when any position drifts beyond *tolerance* from target.

    Typically used with ``Or``: ``Or(RunQuarterly(), RunIfOutOfBounds(0.05))``.
    Requires ``target_weights`` to have been set by a prior weighting algo
    on the *previous* rebalance (stored internally).
    """

    def __init__(self, tolerance: float = 0.05) -> None:
        self._tolerance = float(tolerance)
        self._last_target: dict[str, float] = {}

    def reset(self) -> None:
        self._last_target = {}

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not self._last_target:
            # No previous target — let downstream algos set it, then remember
            return StepDecision(status="skip_day", message="no prior target weights")

        total = float(ctx.total_capital)
        if total <= 0:
            return StepDecision(status="skip_day", message="no capital")

        for sym, target_w in self._last_target.items():
            qty = ctx.positions.get(sym, 0.0)
            if sym in ctx.prices.index and pd.notna(ctx.prices[sym]):
                actual_w = float(qty) * float(ctx.prices[sym]) / total
            else:
                actual_w = 0.0
            if abs(actual_w - target_w) > self._tolerance:
                return StepDecision()  # out of bounds → allow rebalance

        return StepDecision(status="skip_day", message="all weights within bounds")

    def update_target(self, weights: dict[str, float]) -> None:
        """Call after a successful rebalance to remember the new target."""
        self._last_target = dict(weights)


class Or:
    """Logical OR combinator: pass if any child algo passes."""

    def __init__(self, *algos: Algo) -> None:
        self._algos = algos

    def reset(self) -> None:
        for algo in self._algos:
            if hasattr(algo, "reset"):
                algo.reset()

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        for algo in self._algos:
            decision = algo(ctx)
            if decision.status == "continue":
                return StepDecision()
        return StepDecision(status="skip_day", message="all sub-algos skipped")


class Not:
    """Logical NOT combinator: invert the child algo's decision."""

    def __init__(self, algo: Algo) -> None:
        self._algo = algo

    def reset(self) -> None:
        if hasattr(self._algo, "reset"):
            self._algo.reset()

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        decision = self._algo(ctx)
        if decision.status == "skip_day":
            return StepDecision()
        return StepDecision(status="skip_day", message="inverted")


# ---------------------------------------------------------------------------
# Selection algos
# ---------------------------------------------------------------------------

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


class SelectAll:
    """Select all symbols with valid prices on current date."""

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        available = [s for s in ctx.prices.index if pd.notna(ctx.prices[s]) and float(ctx.prices[s]) > 0]
        ctx.selected_symbols = sorted(available)
        if not available:
            return StepDecision(status="skip_day", message="no symbols with valid prices")
        return StepDecision()


class SelectHasData:
    """Select symbols that have at least *min_days* of price history."""

    def __init__(self, min_days: int = 1) -> None:
        self._min_days = int(min_days)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if ctx.price_history is None or ctx.price_history.empty:
            return StepDecision(status="skip_day", message="no price history")
        keep = []
        for s in ctx.selected_symbols or list(ctx.prices.index):
            if s in ctx.price_history.columns:
                valid = ctx.price_history[s].dropna()
                if len(valid) >= self._min_days:
                    keep.append(s)
        ctx.selected_symbols = keep
        if not keep:
            return StepDecision(status="skip_day", message=f"no symbols with {self._min_days}+ days")
        return StepDecision()


class SelectMomentum:
    """Select top *n* symbols by trailing momentum (total return over *lookback* days)."""

    def __init__(self, n: int, lookback: int = 252, sort_descending: bool = True) -> None:
        self._n = int(n)
        self._lookback = int(lookback)
        self._sort_desc = sort_descending

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if ctx.price_history is None or ctx.price_history.empty:
            return StepDecision(status="skip_day", message="no price history for momentum")
        candidates = ctx.selected_symbols or [
            s for s in ctx.prices.index if pd.notna(ctx.prices[s])
        ]
        scores: dict[str, float] = {}
        for s in candidates:
            if s not in ctx.price_history.columns:
                continue
            series = ctx.price_history[s].dropna()
            if len(series) < 2:
                continue
            window = series.iloc[-self._lookback:]
            if len(window) < 2 or float(window.iloc[0]) <= 0:
                continue
            scores[s] = float(window.iloc[-1] / window.iloc[0] - 1)
        ranked = sorted(scores, key=scores.get, reverse=self._sort_desc)  # type: ignore[arg-type]
        ctx.selected_symbols = ranked[: self._n]
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no symbols with enough momentum data")
        return StepDecision()


class SelectN:
    """Keep the first *n* symbols from current selection (stable order)."""

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        ctx.selected_symbols = ctx.selected_symbols[: self._n]
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no symbols after SelectN")
        return StepDecision()


class SelectRandomly:
    """Select *n* symbols at random from the current selection."""

    def __init__(self, n: int, seed: int | None = None) -> None:
        self._n = int(n)
        self._rng = _random.Random(seed)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        candidates = ctx.selected_symbols or [
            s for s in ctx.prices.index if pd.notna(ctx.prices[s])
        ]
        if not candidates:
            return StepDecision(status="skip_day", message="no candidates for random selection")
        k = min(self._n, len(candidates))
        ctx.selected_symbols = sorted(self._rng.sample(candidates, k))
        return StepDecision()


class SelectActive:
    """Filter out symbols whose price is zero or NaN (dead/expired)."""

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        candidates = ctx.selected_symbols or list(ctx.prices.index)
        active = [
            s for s in candidates
            if s in ctx.prices.index and pd.notna(ctx.prices[s]) and float(ctx.prices[s]) > 0
        ]
        ctx.selected_symbols = active
        if not active:
            return StepDecision(status="skip_day", message="no active symbols")
        return StepDecision()


class SelectRegex:
    """Select symbols whose name matches a regex pattern."""

    def __init__(self, pattern: str) -> None:
        self._pattern = _re.compile(pattern)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        candidates = ctx.selected_symbols or list(ctx.prices.index)
        matched = [s for s in candidates if self._pattern.search(s)]
        ctx.selected_symbols = matched
        if not matched:
            return StepDecision(status="skip_day", message=f"no symbols match {self._pattern.pattern!r}")
        return StepDecision()


class SelectWhere:
    """Select symbols where a user-defined function returns True."""

    def __init__(self, fn: Callable[[str, PipelineContext], bool]) -> None:
        self._fn = fn

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        candidates = ctx.selected_symbols or [
            s for s in ctx.prices.index if pd.notna(ctx.prices[s])
        ]
        ctx.selected_symbols = [s for s in candidates if self._fn(s, ctx)]
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no symbols passed filter")
        return StepDecision()


# ---------------------------------------------------------------------------
# Weighting algos
# ---------------------------------------------------------------------------

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


class WeighEqually:
    """Equal-weight all selected symbols."""

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        w = 1.0 / len(ctx.selected_symbols)
        ctx.target_weights = {s: w for s in ctx.selected_symbols}
        return StepDecision()


class WeighRandomly:
    """Assign random weights to selected symbols (normalized to sum to 1).

    Useful for constructing random benchmark strategies.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.RandomState(seed)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        raw = self._rng.dirichlet(np.ones(len(ctx.selected_symbols)))
        ctx.target_weights = {s: float(w) for s, w in zip(ctx.selected_symbols, raw)}
        return StepDecision()


class WeighTarget:
    """Read target weights from a pre-computed DataFrame indexed by date.

    *weights_df* should have dates as index and symbol names as columns.
    On each date, looks up the closest prior row.
    """

    def __init__(self, weights_df: pd.DataFrame) -> None:
        self._weights = weights_df.sort_index()

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        # Find the most recent row <= current date
        mask = self._weights.index <= ctx.date
        if not mask.any():
            return StepDecision(status="skip_day", message="no weight data for this date")
        row = self._weights.loc[mask].iloc[-1]
        weights = {}
        for s in ctx.selected_symbols:
            if s in row.index and pd.notna(row[s]):
                weights[s] = float(row[s])
        if not weights:
            return StepDecision(status="skip_day", message="no matching weights")
        total = sum(weights.values())
        if total <= 0:
            return StepDecision(status="skip_day", message="weights sum to zero")
        ctx.target_weights = {s: w / total for s, w in weights.items()}
        return StepDecision()


class WeighInvVol:
    """Inverse-volatility weighting (risk parity lite).

    Weight_i = (1/vol_i) / sum(1/vol_j).
    Uses trailing *lookback*-day returns standard deviation.
    """

    def __init__(self, lookback: int = 252) -> None:
        self._lookback = int(lookback)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        if ctx.price_history is None or ctx.price_history.empty:
            return StepDecision(status="skip_day", message="no price history for inv-vol")
        inv_vols: dict[str, float] = {}
        for s in ctx.selected_symbols:
            if s not in ctx.price_history.columns:
                continue
            series = ctx.price_history[s].dropna()
            window = series.iloc[-self._lookback:]
            if len(window) < 3:
                continue
            rets = window.pct_change().dropna()
            vol = float(rets.std())
            if vol > 0:
                inv_vols[s] = 1.0 / vol
        if not inv_vols:
            return StepDecision(status="skip_day", message="no valid vol data")
        total = sum(inv_vols.values())
        ctx.target_weights = {s: v / total for s, v in inv_vols.items()}
        return StepDecision()


class WeighMeanVar:
    """Mean-variance optimization (max Sharpe ratio portfolio).

    Uses trailing *lookback*-day returns. Falls back to equal weight
    if optimization fails (singular covariance, etc.).
    """

    def __init__(self, lookback: int = 252, risk_free_rate: float = 0.0) -> None:
        self._lookback = int(lookback)
        self._rf = float(risk_free_rate)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        if ctx.price_history is None or ctx.price_history.empty:
            return StepDecision(status="skip_day", message="no price history for mean-var")
        syms = [s for s in ctx.selected_symbols if s in ctx.price_history.columns]
        if len(syms) < 1:
            return StepDecision(status="skip_day", message="no price history columns match")
        prices = ctx.price_history[syms].dropna()
        if len(prices) < 3:
            return StepDecision(status="skip_day", message="insufficient data for mean-var")
        rets = prices.iloc[-self._lookback:].pct_change().dropna()
        if len(rets) < 3:
            return StepDecision(status="skip_day", message="insufficient returns for mean-var")
        mu = rets.mean().values
        cov = rets.cov().values
        n = len(syms)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Singular covariance — fall back to equal weight
            w = 1.0 / n
            ctx.target_weights = {s: w for s in syms}
            return StepDecision()
        excess = mu - self._rf / 252
        raw_w = cov_inv @ excess
        # Normalize to sum to 1, allow short positions only if naturally arising
        total = float(np.sum(np.abs(raw_w)))
        if total <= 0:
            w = 1.0 / n
            ctx.target_weights = {s: w for s in syms}
            return StepDecision()
        # Long-only: clip negatives, renormalize
        clipped = np.maximum(raw_w, 0.0)
        clip_sum = float(np.sum(clipped))
        if clip_sum <= 0:
            w = 1.0 / n
            ctx.target_weights = {s: w for s in syms}
            return StepDecision()
        weights = clipped / clip_sum
        ctx.target_weights = {s: float(weights[i]) for i, s in enumerate(syms)}
        return StepDecision()


class WeighERC:
    """Equal Risk Contribution weighting.

    Each asset contributes equally to portfolio risk.
    Uses iterative bisection approximation.
    """

    def __init__(self, lookback: int = 252, max_iter: int = 100) -> None:
        self._lookback = int(lookback)
        self._max_iter = int(max_iter)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.selected_symbols:
            return StepDecision(status="skip_day", message="no selected symbols")
        if ctx.price_history is None or ctx.price_history.empty:
            return StepDecision(status="skip_day", message="no price history for ERC")
        syms = [s for s in ctx.selected_symbols if s in ctx.price_history.columns]
        if len(syms) < 1:
            return StepDecision(status="skip_day", message="no matching columns")
        prices = ctx.price_history[syms].dropna()
        rets = prices.iloc[-self._lookback:].pct_change().dropna()
        if len(rets) < 3:
            return StepDecision(status="skip_day", message="insufficient data for ERC")
        cov = rets.cov().values
        n = len(syms)
        # Start with equal weights
        w = np.ones(n) / n
        for _ in range(self._max_iter):
            sigma = np.sqrt(float(w @ cov @ w))
            if sigma <= 0:
                break
            mrc = (cov @ w) / sigma  # marginal risk contribution
            rc = w * mrc  # risk contribution
            target_rc = sigma / n
            # Adjust: increase weight of under-contributing, decrease over-contributing
            adj = target_rc / np.maximum(rc, 1e-12)
            w = w * adj
            w = np.maximum(w, 0.0)
            w_sum = float(np.sum(w))
            if w_sum > 0:
                w = w / w_sum
        ctx.target_weights = {s: float(w[i]) for i, s in enumerate(syms)}
        return StepDecision()


class TargetVol:
    """Scale weights to target a specific annualized portfolio volatility.

    Scales the existing target_weights by (target_vol / realized_vol).
    Excess weight goes to cash.
    """

    def __init__(self, target: float = 0.10, lookback: int = 252) -> None:
        self._target = float(target)
        self._lookback = int(lookback)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.target_weights:
            return StepDecision(status="skip_day", message="no target weights to scale")
        if ctx.price_history is None or ctx.price_history.empty:
            return StepDecision(status="skip_day", message="no price history for vol scaling")
        syms = list(ctx.target_weights.keys())
        available = [s for s in syms if s in ctx.price_history.columns]
        if not available:
            return StepDecision(status="skip_day", message="no price data for vol scaling")
        prices = ctx.price_history[available].dropna()
        rets = prices.iloc[-self._lookback:].pct_change().dropna()
        if len(rets) < 3:
            return StepDecision()  # not enough data, pass through unchanged
        weights_arr = np.array([ctx.target_weights.get(s, 0.0) for s in available])
        port_rets = rets.values @ weights_arr
        realized_vol = float(np.std(port_rets) * np.sqrt(252))
        if realized_vol <= 0:
            return StepDecision()
        scale = min(self._target / realized_vol, 1.0)  # never lever above 1.0
        ctx.target_weights = {s: w * scale for s, w in ctx.target_weights.items()}
        return StepDecision()


# ---------------------------------------------------------------------------
# Weight limits
# ---------------------------------------------------------------------------

class LimitWeights:
    """Cap individual position weights and renormalize."""

    def __init__(self, limit: float = 0.25) -> None:
        self._limit = float(limit)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.target_weights:
            return StepDecision()
        # Iteratively clip and renormalize (may need multiple passes)
        weights = dict(ctx.target_weights)
        for _ in range(10):
            over = {s: w for s, w in weights.items() if w > self._limit}
            if not over:
                break
            under = {s: w for s, w in weights.items() if w <= self._limit}
            for s in over:
                weights[s] = self._limit
            under_sum = sum(under.values())
            over_excess = sum(w - self._limit for w in over.values())
            if under_sum > 0:
                scale = 1.0 + over_excess / under_sum
                for s in under:
                    weights[s] = weights[s] * scale
        ctx.target_weights = weights
        return StepDecision()


class LimitDeltas:
    """Cap how much any single weight can change between rebalances.

    On each call, computes the current portfolio weights from positions and
    clips ``target_weights`` so no weight moves more than *limit* from its
    current value.  Excess is redistributed proportionally.
    """

    def __init__(self, limit: float = 0.10) -> None:
        self._limit = float(limit)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.target_weights:
            return StepDecision()
        total = float(ctx.total_capital)
        if total <= 0:
            return StepDecision()

        # Compute current weights from positions
        current: dict[str, float] = {}
        for sym in ctx.target_weights:
            qty = ctx.positions.get(sym, 0.0)
            if sym in ctx.prices.index and pd.notna(ctx.prices[sym]):
                current[sym] = float(qty) * float(ctx.prices[sym]) / total
            else:
                current[sym] = 0.0

        # Clip deltas
        clipped: dict[str, float] = {}
        for sym, target_w in ctx.target_weights.items():
            cur_w = current.get(sym, 0.0)
            delta = target_w - cur_w
            clamped = max(-self._limit, min(self._limit, delta))
            clipped[sym] = cur_w + clamped

        # Renormalize to sum to original target sum
        orig_sum = sum(ctx.target_weights.values())
        clip_sum = sum(clipped.values())
        if clip_sum > 0 and orig_sum > 0:
            scale = orig_sum / clip_sum
            clipped = {s: w * scale for s, w in clipped.items()}

        ctx.target_weights = clipped
        return StepDecision()


class ScaleWeights:
    """Multiply all target weights by a scalar.

    Useful for leverage (scale > 1) or de-leverage (scale < 1).
    Excess weight goes to cash.
    """

    def __init__(self, scale: float) -> None:
        self._scale = float(scale)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        if not ctx.target_weights:
            return StepDecision()
        ctx.target_weights = {s: w * self._scale for s, w in ctx.target_weights.items()}
        return StepDecision()


# ---------------------------------------------------------------------------
# Capital flows
# ---------------------------------------------------------------------------

class CapitalFlow:
    """Model periodic capital additions (+) or withdrawals (-).

    *flows* is a dict mapping dates to amounts, or a callable
    ``(date: pd.Timestamp) -> float`` returning the flow amount.
    """

    def __init__(self, flows: dict[str | pd.Timestamp, float] | Callable[[pd.Timestamp], float]) -> None:
        if callable(flows):
            self._fn = flows
        else:
            mapping = {pd.Timestamp(k).normalize(): float(v) for k, v in flows.items()}
            self._fn = lambda d: mapping.get(d.normalize(), 0.0)

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        amount = self._fn(ctx.date)
        if amount != 0.0:
            ctx.cash = float(ctx.cash + amount)
            ctx.total_capital = float(ctx.total_capital + amount)
        return StepDecision()


# ---------------------------------------------------------------------------
# Risk guards
# ---------------------------------------------------------------------------

class MaxDrawdownGuard:
    """Block new rebalances while drawdown exceeds threshold."""

    def __init__(self, max_drawdown_pct: float) -> None:
        self.max_drawdown_pct = float(max_drawdown_pct)
        self._peak = 0.0

    def reset(self) -> None:
        self._peak = 0.0

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        self._peak = max(self._peak, float(ctx.total_capital))
        if self._peak <= 0:
            return StepDecision()
        dd = (self._peak - float(ctx.total_capital)) / self._peak
        if dd > self.max_drawdown_pct:
            return StepDecision(status="skip_day", message=f"drawdown {dd:.2%} > {self.max_drawdown_pct:.2%}")
        return StepDecision()


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

class CloseDead:
    """Close positions where price has dropped to zero or is NaN.

    Removes dead positions and frees up the capital (at zero value).
    """

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        dead = []
        for sym, qty in ctx.positions.items():
            if sym not in ctx.prices.index or pd.isna(ctx.prices[sym]) or float(ctx.prices[sym]) <= 0:
                dead.append(sym)
        for sym in dead:
            del ctx.positions[sym]
        if dead:
            return StepDecision(message=f"closed dead: {', '.join(dead)}")
        return StepDecision()


class ClosePositionsAfterDates:
    """Close specific positions on or after given dates.

    *schedule* maps symbol names to the date after which they should be closed.
    """

    def __init__(self, schedule: dict[str, str | pd.Timestamp]) -> None:
        self._schedule = {s.upper(): pd.Timestamp(d).normalize() for s, d in schedule.items()}

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        closed = []
        for sym, close_date in self._schedule.items():
            if ctx.date.normalize() >= close_date and sym in ctx.positions:
                del ctx.positions[sym]
                closed.append(sym)
        if closed:
            return StepDecision(message=f"closed after date: {', '.join(closed)}")
        return StepDecision()


class Require:
    """Guard: only continue if the wrapped algo returns 'continue'.

    Unlike normal pipeline flow, ``Require`` runs the inner algo but does
    NOT break the pipeline on skip — it only checks whether the algo *would*
    have passed. Use it to conditionally gate downstream steps.
    """

    def __init__(self, algo: Algo) -> None:
        self._algo = algo

    def reset(self) -> None:
        if hasattr(self._algo, "reset"):
            self._algo.reset()

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        decision = self._algo(ctx)
        if decision.status != "continue":
            return StepDecision(status="skip_day", message=f"requirement not met: {decision.message}")
        return StepDecision()


# ---------------------------------------------------------------------------
# Rebalancing algos
# ---------------------------------------------------------------------------

class Rebalance:
    """Rebalance positions to target weights at current prices.

    Performs a full liquidate-and-rebuy on each rebalance date.
    """

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


class RebalanceOverTime:
    """Spread rebalancing over *n* periods to reduce market impact.

    On each trigger, moves 1/n of the way from current to target weights.
    Must be preceded by a scheduling algo and a weighting algo.
    """

    def __init__(self, n: int = 5) -> None:
        self._n = int(n)
        self._target: dict[str, float] = {}
        self._remaining = 0

    def reset(self) -> None:
        self._target = {}
        self._remaining = 0

    def __call__(self, ctx: PipelineContext) -> StepDecision:
        # If new target weights are set, start a new gradual rebalance
        if ctx.target_weights and ctx.target_weights != self._target:
            self._target = dict(ctx.target_weights)
            self._remaining = self._n

        if self._remaining <= 0 or not self._target:
            return StepDecision(status="skip_day", message="no gradual rebalance in progress")

        # Compute current weights from positions
        total = float(ctx.total_capital)
        if total <= 0:
            return StepDecision(status="skip_day", message="no capital")

        current_weights: dict[str, float] = {}
        all_syms = set(self._target.keys()) | set(ctx.positions.keys())
        for sym in all_syms:
            qty = ctx.positions.get(sym, 0.0)
            if sym in ctx.prices.index and pd.notna(ctx.prices[sym]):
                current_weights[sym] = float(qty) * float(ctx.prices[sym]) / total
            else:
                current_weights[sym] = 0.0

        # Move fraction of the way toward target
        frac = 1.0 / self._remaining
        blended: dict[str, float] = {}
        for sym in all_syms:
            cur = current_weights.get(sym, 0.0)
            tgt = self._target.get(sym, 0.0)
            blended[sym] = cur + frac * (tgt - cur)

        # Apply blended weights
        new_positions: dict[str, float] = {}
        spent = 0.0
        for sym, w in blended.items():
            if sym not in ctx.prices.index or pd.isna(ctx.prices[sym]):
                continue
            price = float(ctx.prices[sym])
            if price <= 0:
                continue
            target_value = total * w
            qty = float(np.floor(target_value / price))
            if qty > 0:
                new_positions[sym] = qty
                spent += qty * price

        ctx.positions.clear()
        ctx.positions.update(new_positions)
        ctx.cash = float(total - spent)
        self._remaining -= 1
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
        self.logs = []
        for algo in self.algos:
            if hasattr(algo, "reset"):
                algo.reset()
        cash = float(self.initial_capital)
        positions: dict[str, float] = {}
        rows: list[dict[str, float | pd.Timestamp]] = []

        all_dates = list(self.prices.index)
        for i, (date, price_row) in enumerate(self.prices.iterrows()):
            stocks_cap = float(sum(float(qty) * float(price_row[sym])
                                   for sym, qty in positions.items()
                                   if sym in price_row.index and pd.notna(price_row[sym])))
            total_cap = cash + stocks_cap
            # Price history up to current date (for algos that need lookback)
            history = self.prices.iloc[:i + 1] if i > 0 else self.prices.iloc[:1]
            ctx = PipelineContext(
                date=pd.Timestamp(date),
                prices=price_row,
                total_capital=total_cap,
                cash=cash,
                positions=dict(positions),
                price_history=history,
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
            stocks_cap = float(sum(float(qty) * float(price_row[sym])
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

        if not rows:
            balance = pd.DataFrame()
            self.balance = balance
            return balance
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


# ---------------------------------------------------------------------------
# Random benchmarking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RandomBenchmarkResult:
    """Result of ``benchmark_random``: your strategy vs random portfolios."""

    strategy_return: float
    random_returns: list[float]
    percentile: float  # what % of random runs your strategy beat

    @property
    def mean_random(self) -> float:
        return float(np.mean(self.random_returns))

    @property
    def std_random(self) -> float:
        return float(np.std(self.random_returns))


def benchmark_random(
    prices: pd.DataFrame,
    strategy_algos: list[Algo],
    n_random: int = 100,
    initial_capital: float = 1_000_000.0,
    seed: int = 42,
) -> RandomBenchmarkResult:
    """Compare a strategy against *n_random* random-weight portfolios.

    Runs the given strategy once, then runs *n_random* simulations with
    ``SelectAll → WeighRandomly → Rebalance`` on the same price data.
    Returns a ``RandomBenchmarkResult`` with the strategy's total return,
    the distribution of random returns, and the percentile rank.
    """
    # Run the target strategy
    bt = AlgoPipelineBacktester(prices=prices, algos=strategy_algos, initial_capital=initial_capital)
    bal = bt.run()
    if bal.empty:
        strat_ret = 0.0
    else:
        strat_ret = float(bal["total capital"].iloc[-1] / bal["total capital"].iloc[0] - 1)

    # Run random strategies
    random_rets: list[float] = []
    for i in range(n_random):
        random_algos: list[Algo] = [
            RunMonthly(),
            SelectAll(),
            WeighRandomly(seed=seed + i),
            Rebalance(),
        ]
        rbt = AlgoPipelineBacktester(prices=prices, algos=random_algos, initial_capital=initial_capital)
        rbal = rbt.run()
        if rbal.empty:
            random_rets.append(0.0)
        else:
            random_rets.append(float(rbal["total capital"].iloc[-1] / rbal["total capital"].iloc[0] - 1))

    beaten = sum(1 for r in random_rets if strat_ret > r)
    pct = beaten / max(len(random_rets), 1) * 100

    return RandomBenchmarkResult(
        strategy_return=strat_ret,
        random_returns=random_rets,
        percentile=pct,
    )
