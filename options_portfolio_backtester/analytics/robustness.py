"""Score a strategy over entry calendars, not on one lucky schedule.

An options overlay that enters every N months is exposed to *which* days it
enters and exits: a single crash bottom landing on an exit date can make a
configuration look like a winner. In the SPX tail-hedge study, the same
strategy ranged from -0.84 to +0.72pp/yr vs buy & hold depending only on
whether entries fell in odd or even months, and the "best" exit DTE won
because its Dec-2008 sale fell on the 2008-11-20 closing low.

The fix is cheap: run the same configuration on every phase of the schedule
(each starting month of the cycle x a few trading-day offsets) and report the
mean and the worst calendar, not the best one::

    cals = entry_calendars(stocks_df["date"], every_n_months=2)

    def run(dates):
        engine = make_engine()           # your configured BacktestEngine
        engine.run(rebalance_dates=dates)
        return engine

    per_calendar = run_over_calendars(run, cals)
    print(calendar_summary(per_calendar))
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


def entry_calendars(
    trading_days: Iterable[Any],
    every_n_months: int = 2,
    day_offsets: tuple[int, ...] = (0, 9),
) -> dict[str, list[pd.Timestamp]]:
    """Every phase of an every-N-months entry schedule.

    For each starting month of the cycle (``every_n_months`` of them) and
    each trading-day offset within the month, the dates are the
    ``offset``-th trading day of every N-th month. Every calendar also enters
    on the first trading day, so all start invested on the same day as a
    buy & hold benchmark. Labels look like ``"m1/d1"`` (cycle month 1,
    first trading day) or ``"m2/d10"``.
    """
    if every_n_months < 1:
        raise ValueError("every_n_months must be >= 1")
    days = pd.Series(pd.DatetimeIndex(sorted(set(pd.to_datetime(list(trading_days))))))
    if days.empty:
        raise ValueError("no trading days")
    months = days.dt.year * 12 + days.dt.month
    out: dict[str, list[pd.Timestamp]] = {}
    for phase in range(every_n_months):
        keep = (months - months.iloc[0]) % every_n_months == phase
        groups = days[keep].groupby(months[keep])
        for off in day_offsets:
            dates = [g.iloc[min(off, len(g) - 1)] for _, g in groups]
            if not dates or dates[0] != days.iloc[0]:
                dates = [days.iloc[0]] + dates
            out[f"m{phase + 1}/d{off + 1}"] = dates
    return out


def _curve_metrics(total: pd.Series) -> dict[str, float]:
    years = (total.index[-1] - total.index[0]).days / 365.25
    cagr = ((total.iloc[-1] / total.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else math.nan
    peak = total.cummax()
    daily = total.pct_change().dropna()
    vol = daily.std() * math.sqrt(252) * 100
    return {
        "cagr": cagr,
        "max_drawdown": ((total - peak) / peak).min() * 100,
        "volatility": vol,
        "sharpe": cagr / vol if vol > 0 else math.nan,
        "final": float(total.iloc[-1]),
    }


def run_over_calendars(
    run: Callable[[list[pd.Timestamp]], Any],
    calendars: dict[str, list[pd.Timestamp]],
) -> pd.DataFrame:
    """Call ``run(dates)`` once per calendar and score each result.

    ``run`` returns either a finished :class:`BacktestEngine` (its
    ``balance["total capital"]`` is scored and its ``option_fill_rate``
    reported) or a total-capital ``pd.Series`` indexed by date. Returns one
    row per calendar: ``cagr`` and ``max_drawdown`` in percent, annualized
    ``volatility`` in percent, ``sharpe`` (CAGR / vol, no risk-free rate),
    ``final`` capital and ``fill_rate``.
    """
    rows = []
    for label, dates in calendars.items():
        result = run(dates)
        if isinstance(result, pd.Series):
            total, fill = result, math.nan
        else:
            total = result.balance["total capital"]
            fill = getattr(result, "option_fill_rate", math.nan)
        rows.append({"calendar": label, **_curve_metrics(total), "fill_rate": fill})
    return pd.DataFrame(rows).set_index("calendar")


def calendar_summary(per_calendar: pd.DataFrame, column: str = "cagr") -> pd.Series:
    """Mean, worst, best and spread of ``column`` across calendars. The mean
    is the honest point estimate; a spread wider than the edge you are
    measuring means the edge is calendar luck."""
    values = per_calendar[column]
    return pd.Series({
        "mean": values.mean(),
        "worst": values.min(),
        "best": values.max(),
        "spread": values.max() - values.min(),
        "calendars": float(len(values)),
    }, name=column)
