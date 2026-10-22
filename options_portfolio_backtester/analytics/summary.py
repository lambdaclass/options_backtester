"""Summary statistics for trade logs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from options_portfolio_backtester.analytics.trade_log import TradeLog


def summary(trade_log: pd.DataFrame, balance: pd.DataFrame,
            cost_model: Any = None) -> pd.io.formats.style.Styler:
    """A table of round-trip trade statistics and portfolio return.

    Trades are paired as in :meth:`TradeLog.from_legacy_trade_log` (so
    shorts and multi-leg positions are signed correctly); pass the run's
    ``cost_model`` to include commissions. Neither input is modified.
    """
    trades = TradeLog.from_legacy_trade_log(trade_log, cost_model=cost_model)
    pnls = trades.net_pnls
    total_trades = len(pnls)
    win_number = int(np.sum(pnls > 0))
    loss_number = total_trades - win_number
    gross_profit = float(pnls[pnls > 0].sum()) if total_trades else 0.0
    gross_loss = float(-pnls[pnls <= 0].sum()) if total_trades else 0.0

    total = balance["total capital"]
    data = [
        total_trades,
        win_number,
        loss_number,
        100.0 * win_number / total_trades if total_trades else 0.0,
        max(0.0, -float(pnls.min())) if total_trades else 0.0,
        gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
        float(pnls.mean()) if total_trades else 0.0,
        float(total.pct_change().mean() * 100),
        float((total.iloc[-1] / total.iloc[0] - 1) * 100),
    ]
    stats = [
        'Total trades', 'Number of wins', 'Number of losses', 'Win %', 'Largest loss', 'Profit factor',
        'Average profit', 'Average P&L %', 'Total P&L %'
    ]
    summary_df = pd.DataFrame(data, stats, ['Strategy'])

    formatters: dict[str, str] = {
        "Total trades": "{:.0f}",
        "Number of wins": "{:.0f}",
        "Number of losses": "{:.0f}",
        "Win %": "{:.2f}%",
        "Largest loss": "${:.2f}",
        "Profit factor": "{:.2f}",
        "Average profit": "${:.2f}",
        "Average P&L %": "{:.2f}%",
        "Total P&L %": "{:.2f}%"
    }

    styled = summary_df.style
    for row_label, fmt in formatters.items():
        styled = styled.format(fmt, subset=pd.IndexSlice[row_label, :])  # type: ignore[arg-type]
    return styled
