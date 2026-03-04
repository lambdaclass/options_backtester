"""BacktestStats — comprehensive analytics matching and exceeding bt/ffn.

Provides:
- Trade stats: profit factor, win rate, largest win/loss
- Return stats: total, annualized, Sharpe, Sortino, Calmar
- Risk stats: max drawdown, drawdown duration, volatility, tail ratio
- Period stats: monthly/yearly Sharpe, Sortino, mean, vol, skew, kurtosis
- Extreme analysis: best/worst day, month, year
- Lookback returns: MTD, 3M, 6M, YTD, 1Y, 3Y, 5Y, 10Y
- Portfolio metrics: turnover, Herfindahl concentration index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from options_portfolio_backtester._ob_rust import compute_full_stats


@dataclass
class PeriodStats:
    """Stats for a specific return frequency (daily, monthly, yearly)."""
    mean: float = 0.0
    vol: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    best: float = 0.0
    worst: float = 0.0


@dataclass
class LookbackReturns:
    """Trailing period returns as of the end date."""
    mtd: float | None = None
    three_month: float | None = None
    six_month: float | None = None
    ytd: float | None = None
    one_year: float | None = None
    three_year: float | None = None
    five_year: float | None = None
    ten_year: float | None = None


@dataclass
class BacktestStats:
    """Comprehensive backtest statistics."""

    # Trade stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0

    # Return stats
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk stats
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    avg_drawdown_duration: int = 0
    volatility: float = 0.0
    tail_ratio: float = 0.0

    # Period stats
    daily: PeriodStats = field(default_factory=PeriodStats)
    monthly: PeriodStats = field(default_factory=PeriodStats)
    yearly: PeriodStats = field(default_factory=PeriodStats)

    # Lookback
    lookback: LookbackReturns = field(default_factory=LookbackReturns)

    # Portfolio metrics
    turnover: float = 0.0
    herfindahl: float = 0.0

    @classmethod
    def from_balance_range(
        cls,
        balance: pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        **kwargs,
    ) -> "BacktestStats":
        """Slice balance to [start, end] and recompute all stats."""
        if balance.empty:
            return cls()
        b = balance.copy()
        if start:
            b = b.loc[pd.Timestamp(start):]
        if end:
            b = b.loc[:pd.Timestamp(end)]
        if b.empty:
            return cls()
        b["% change"] = b["total capital"].pct_change()
        return cls.from_balance(b, **kwargs)

    @classmethod
    def from_balance(
        cls,
        balance: pd.DataFrame,
        trade_pnls: np.ndarray | None = None,
        risk_free_rate: float = 0.0,
    ) -> "BacktestStats":
        """Compute stats from a balance DataFrame and optional trade P&Ls."""
        if balance.empty:
            return cls()

        total_capital = balance["total capital"].values.astype(np.float64)
        timestamps_ns = balance.index.astype(np.int64).tolist()
        pnls = trade_pnls.astype(np.float64).tolist() if trade_pnls is not None else []

        # Build stock weight matrix
        stock_cols = [c for c in balance.columns if f"{c} qty" in balance.columns]
        if stock_cols:
            total = balance["total capital"].values
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = balance[stock_cols].values / total[:, None]
            weights = np.nan_to_num(weights, 0.0).astype(np.float64)
            flat_weights = weights.ravel().tolist()
            n_stocks = len(stock_cols)
        else:
            flat_weights = []
            n_stocks = 0

        d = compute_full_stats(
            total_capital.tolist(),
            timestamps_ns,
            pnls,
            flat_weights,
            n_stocks,
            risk_free_rate,
        )

        stats = cls()
        # Scalars
        for attr in (
            "total_trades", "wins", "losses", "win_pct", "profit_factor",
            "largest_win", "largest_loss", "avg_win", "avg_loss", "avg_trade",
            "total_return", "annualized_return", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "max_drawdown_duration",
            "avg_drawdown", "avg_drawdown_duration", "volatility", "tail_ratio",
            "turnover", "herfindahl",
        ):
            setattr(stats, attr, d[attr])

        # Period stats
        for period_name in ("daily", "monthly", "yearly"):
            pd_dict = d[period_name]
            setattr(stats, period_name, PeriodStats(
                mean=pd_dict["mean"], vol=pd_dict["vol"],
                sharpe=pd_dict["sharpe"], sortino=pd_dict["sortino"],
                skew=pd_dict["skew"], kurtosis=pd_dict["kurtosis"],
                best=pd_dict["best"], worst=pd_dict["worst"],
            ))

        # Lookback
        lb = d["lookback"]
        stats.lookback = LookbackReturns(
            mtd=lb["mtd"], three_month=lb["three_month"],
            six_month=lb["six_month"], ytd=lb["ytd"],
            one_year=lb["one_year"], three_year=lb["three_year"],
            five_year=lb["five_year"], ten_year=lb["ten_year"],
        )

        return stats

    def to_dataframe(self) -> pd.DataFrame:
        """Return stats as a styled DataFrame."""
        data = {
            "Total trades": self.total_trades,
            "Wins": self.wins,
            "Losses": self.losses,
            "Win %": self.win_pct,
            "Profit factor": self.profit_factor,
            "Largest win": self.largest_win,
            "Largest loss": self.largest_loss,
            "Avg win": self.avg_win,
            "Avg loss": self.avg_loss,
            "Avg trade": self.avg_trade,
            "Total return": self.total_return,
            "Annualized return": self.annualized_return,
            "Sharpe ratio": self.sharpe_ratio,
            "Sortino ratio": self.sortino_ratio,
            "Calmar ratio": self.calmar_ratio,
            "Max drawdown": self.max_drawdown,
            "Max DD duration (days)": self.max_drawdown_duration,
            "Avg drawdown": self.avg_drawdown,
            "Avg DD duration (days)": self.avg_drawdown_duration,
            "Volatility": self.volatility,
            "Tail ratio": self.tail_ratio,
            # Daily
            "Daily mean": self.daily.mean,
            "Daily vol": self.daily.vol,
            "Daily Sharpe": self.daily.sharpe,
            "Daily Sortino": self.daily.sortino,
            "Daily skew": self.daily.skew,
            "Daily kurtosis": self.daily.kurtosis,
            "Best day": self.daily.best,
            "Worst day": self.daily.worst,
            # Monthly
            "Monthly mean": self.monthly.mean,
            "Monthly vol": self.monthly.vol,
            "Monthly Sharpe": self.monthly.sharpe,
            "Monthly Sortino": self.monthly.sortino,
            "Monthly skew": self.monthly.skew,
            "Monthly kurtosis": self.monthly.kurtosis,
            "Best month": self.monthly.best,
            "Worst month": self.monthly.worst,
            # Yearly
            "Yearly mean": self.yearly.mean,
            "Yearly vol": self.yearly.vol,
            "Yearly Sharpe": self.yearly.sharpe,
            "Yearly Sortino": self.yearly.sortino,
            "Best year": self.yearly.best,
            "Worst year": self.yearly.worst,
            # Portfolio
            "Turnover": self.turnover,
            "Herfindahl index": self.herfindahl,
        }
        # Add lookback returns (skip None values)
        lb = self.lookback
        for label, val in [
            ("MTD", lb.mtd), ("3M return", lb.three_month),
            ("6M return", lb.six_month), ("YTD", lb.ytd),
            ("1Y return", lb.one_year), ("3Y return", lb.three_year),
            ("5Y return", lb.five_year), ("10Y return", lb.ten_year),
        ]:
            if val is not None:
                data[label] = val

        return pd.DataFrame(
            list(data.values()), index=list(data.keys()), columns=["Value"]
        )

    def summary(self) -> str:
        """Return a formatted text summary."""
        lines = [
            f"Total Return:      {self.total_return:>10.2%}",
            f"Annualized Return: {self.annualized_return:>10.2%}",
            f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"Sortino Ratio:     {self.sortino_ratio:>10.2f}",
            f"Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"Max DD Duration:   {self.max_drawdown_duration:>10d} days",
            f"Calmar Ratio:      {self.calmar_ratio:>10.2f}",
            f"Profit Factor:     {self.profit_factor:>10.2f}",
            f"Win Rate:          {self.win_pct:>10.1f}%",
            f"Total Trades:      {self.total_trades:>10d}",
        ]
        if self.monthly.sharpe != 0:
            lines.append(f"Monthly Sharpe:    {self.monthly.sharpe:>10.2f}")
        if self.monthly.best != 0:
            lines.append(f"Best Month:        {self.monthly.best:>10.2%}")
            lines.append(f"Worst Month:       {self.monthly.worst:>10.2%}")
        if self.turnover != 0:
            lines.append(f"Turnover:          {self.turnover:>10.2%}")
        return "\n".join(lines)

    def lookback_table(self) -> pd.DataFrame:
        """Lookback returns as a single-row DataFrame."""
        lb = self.lookback
        data = {}
        for label, val in [
            ("MTD", lb.mtd), ("3M", lb.three_month), ("6M", lb.six_month),
            ("YTD", lb.ytd), ("1Y", lb.one_year), ("3Y", lb.three_year),
            ("5Y", lb.five_year), ("10Y", lb.ten_year),
        ]:
            if val is not None:
                data[label] = val
        if not data:
            return pd.DataFrame()
        return pd.DataFrame([data])
