"""BacktestResults — a dataclass bundle of a backtest's outputs and metadata.

The motivation: after ``BacktestEngine.run()`` finishes the engine instance
holds ``balance``, ``trade_log``, etc., but downstream callers commonly need
the *configuration* and *engine version* alongside those frames so they can
audit reproductions or compare runs across engine releases. Without that
metadata, saved results are ambiguous — "was this from before or after the
budget-mode fix?" — and the answer requires off-band notes.

``BacktestResults`` packages everything together. Use ``engine.get_results()``
after ``engine.run()`` to capture it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from importlib.metadata import version
    _PACKAGE_VERSION = version("options_portfolio_backtester")
except Exception:  # pragma: no cover
    _PACKAGE_VERSION = "unknown"


@dataclass
class BacktestResults:
    """Bundle of a backtest run's outputs and the configuration that produced
    them. Designed to be saved to disk (pickle / parquet split) and re-loaded
    later without losing context about which engine version produced it."""

    balance: pd.DataFrame
    trade_log: Optional[pd.DataFrame]
    config: dict[str, Any]
    engine_version: str = field(default=_PACKAGE_VERSION)
    data_hash: Optional[str] = None

    # ---------------------------------------------------------------- properties
    @property
    def returns(self) -> pd.Series:
        """Daily returns of ``total capital`` as a clean ``pd.Series``.

        DatetimeIndex, NaN-free, named ``"strategy"`` — the shape that
        return-analytics libraries (quantstats, empyrical) expect, so a full
        generic tearsheet is two lines:

            >>> import quantstats as qs
            >>> qs.reports.html(results.returns, benchmark_results.returns)
        """
        return returns_from_balance(self.balance, name="strategy")

    @property
    def annual_return(self) -> float:
        """Compound annual growth rate of ``total capital`` (percentage)."""
        bal = self.balance["total capital"]
        years = (bal.index[-1] - bal.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        return ((bal.iloc[-1] / bal.iloc[0]) ** (1 / years) - 1) * 100

    @property
    def max_drawdown(self) -> float:
        """Worst peak-to-trough drawdown over the run (negative percentage)."""
        bal = self.balance["total capital"]
        cummax = bal.cummax()
        return ((bal - cummax) / cummax).min() * 100

    @property
    def _periods_per_year(self) -> float:
        """Sampling frequency of the balance rows (≈252 for daily rows)."""
        idx = self.balance.index
        years = (idx[-1] - idx[0]).days / 365.25
        return (len(idx) - 1) / years if years > 0 else 252.0

    @property
    def annualized_volatility(self) -> float:
        """Annualized standard deviation of period returns (percentage)."""
        rets = self.balance["total capital"].pct_change().dropna()
        return rets.std() * math.sqrt(self._periods_per_year) * 100

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio at a 0% risk-free rate: annualized mean period return
        over annualized volatility -- the same definition as
        :class:`BacktestStats` (not CAGR / volatility)."""
        rets = self.balance["total capital"].pct_change().dropna()
        sd = rets.std()
        return float(rets.mean() / sd * math.sqrt(self._periods_per_year)) if sd > 0 else 0.0

    @property
    def round_trips(self) -> int:
        """Completed round-trip trades (an entry and its exit count once)."""
        from options_portfolio_backtester.analytics.trade_log import TradeLog
        if self.trade_log is None or self.trade_log.empty:
            return 0
        if not isinstance(self.trade_log.columns, pd.MultiIndex):
            return len(self.trade_log)  # not an engine trade log: rows as given
        return len(TradeLog.from_legacy_trade_log(self.trade_log))

    def summary(self) -> dict[str, Any]:
        """Headline metrics in a dict, handy for logging and assertions."""
        return {
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "volatility": self.annualized_volatility,
            "sharpe": self.sharpe_ratio,
            "trades": self.round_trips,
            "engine_version": self.engine_version,
            "data_hash": self.data_hash,
        }


def returns_from_balance(balance: pd.DataFrame, name: str = "strategy") -> pd.Series:
    """Daily returns series from a balance frame's ``total capital`` column.

    Returns an empty float series (with the given name) when the balance is
    empty or lacks the column, so callers can branch on ``.empty`` instead of
    catching KeyError.
    """
    if balance is None or balance.empty or "total capital" not in balance.columns:
        return pd.Series(dtype=float, name=name)
    total = balance["total capital"].dropna()
    rets = total.pct_change().dropna()
    rets.name = name
    return rets


def hash_data_file(path: str | Path, chunk_size: int = 65536) -> str:
    """SHA-256 of a file on disk, useful for ``BacktestResults.data_hash``.

    Two backtests with the same engine version, the same config, and the
    same ``data_hash`` should produce identical results — pinning all
    three is the cleanest definition of "reproducible run" for this
    framework.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
