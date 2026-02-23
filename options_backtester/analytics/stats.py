"""BacktestStats — comprehensive analytics with fixed profit_factor.

Replaces the original stats.py with:
- Fixed profit_factor (dollar gross profit / gross loss, NOT win/loss count ratio)
- Sharpe, Sortino, Calmar ratios
- Max drawdown duration
- Tail ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestStats:
    """Comprehensive backtest statistics."""

    # Trade stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.0
    profit_factor: float = 0.0  # FIXED: dollar gross_profit / gross_loss
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
    max_drawdown_duration: int = 0  # in trading days
    volatility: float = 0.0
    tail_ratio: float = 0.0

    @classmethod
    def from_balance(
        cls,
        balance: pd.DataFrame,
        trade_pnls: np.ndarray | None = None,
        risk_free_rate: float = 0.0,
    ) -> "BacktestStats":
        """Compute stats from a balance DataFrame and optional trade P&Ls.

        Args:
            balance: DataFrame with 'total capital' and '% change' columns.
            trade_pnls: Array of per-trade net P&L values.
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
        """
        stats = cls()

        if balance.empty:
            return stats

        total_capital = balance["total capital"]
        returns = balance["% change"].dropna()

        # -- Return metrics --
        stats.total_return = (total_capital.iloc[-1] / total_capital.iloc[0]) - 1.0
        n_days = len(returns)
        if n_days > 0:
            stats.annualized_return = (
                (1 + stats.total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
            )
            stats.volatility = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

            # Sharpe
            daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            excess = returns - daily_rf
            if excess.std() > 0:
                stats.sharpe_ratio = float(
                    excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                )

            # Sortino (only downside deviation)
            downside = excess[excess < 0]
            if len(downside) > 0 and downside.std() > 0:
                stats.sortino_ratio = float(
                    excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                )

        # -- Drawdown --
        cummax = total_capital.cummax()
        drawdown = (total_capital - cummax) / cummax
        stats.max_drawdown = float(abs(drawdown.min()))

        # Max drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            groups = (~in_drawdown).cumsum()
            dd_lengths = in_drawdown.groupby(groups).sum()
            stats.max_drawdown_duration = int(dd_lengths.max())

        # Calmar
        if stats.max_drawdown > 0:
            stats.calmar_ratio = stats.annualized_return / stats.max_drawdown

        # Tail ratio (95th percentile / abs(5th percentile))
        if len(returns) > 20:
            p95 = np.percentile(returns, 95)
            p5 = abs(np.percentile(returns, 5))
            if p5 > 0:
                stats.tail_ratio = float(p95 / p5)

        # -- Trade stats --
        if trade_pnls is not None and len(trade_pnls) > 0:
            stats.total_trades = len(trade_pnls)
            winning = trade_pnls[trade_pnls > 0]
            losing = trade_pnls[trade_pnls <= 0]
            stats.wins = len(winning)
            stats.losses = len(losing)
            stats.win_pct = (stats.wins / stats.total_trades * 100) if stats.total_trades > 0 else 0

            # FIXED profit_factor: dollar ratio, not count ratio
            gross_profit = winning.sum() if len(winning) > 0 else 0
            gross_loss = abs(losing.sum()) if len(losing) > 0 else 0
            stats.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            stats.largest_win = float(winning.max()) if len(winning) > 0 else 0
            stats.largest_loss = float(losing.min()) if len(losing) > 0 else 0
            stats.avg_win = float(winning.mean()) if len(winning) > 0 else 0
            stats.avg_loss = float(losing.mean()) if len(losing) > 0 else 0
            stats.avg_trade = float(trade_pnls.mean())

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
            "Volatility": self.volatility,
            "Tail ratio": self.tail_ratio,
        }
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
        return "\n".join(lines)
