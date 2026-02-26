"""Legacy summary statistics function (moved from backtester.statistics.stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from options_portfolio_backtester.core.types import Order


def summary(trade_log: pd.DataFrame, balance: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Returns a table with summary statistics about the trade log"""

    initial_capital: float = balance['total capital'].iloc[0]
    trade_log.loc[:,
                  ('totals',
                   'capital')] = (-trade_log['totals']['cost'] * trade_log['totals']['qty']).cumsum() + initial_capital

    daily_returns: pd.Series = balance['% change'] * 100

    first_leg: str = trade_log.columns.levels[0][0]

    ## Not sure of a better way to to this, just doing `Order` or `@Order` inside
    ## the .eval(...) does not seem to work.
    order_bto = Order.BTO
    order_sto = Order.STO

    entry_mask: pd.Series = trade_log[first_leg].eval('(order == @order_bto) | (order == @order_sto)')
    entries: pd.DataFrame = trade_log.loc[entry_mask]
    exits: pd.DataFrame = trade_log.loc[~entry_mask]

    costs: np.ndarray = np.array([])
    for contract in entries[first_leg]['contract']:
        entry = entries.loc[entries[first_leg]['contract'] == contract]
        exit_ = exits.loc[exits[first_leg]['contract'] == contract]
        try:
            # Here we assume we are entering only once per contract (i.e both entry and exit_ have only one row)
            costs = np.append(costs, (entry['totals']['cost'] * entry['totals']['qty']).values[0] +
                              (exit_['totals']['cost'] * exit_['totals']['qty']).values[0])
        except IndexError:
            continue

    wins: np.ndarray = costs < 0
    losses: np.ndarray = costs >= 0
    total_trades: int = len(exits)
    win_number: int = int(np.sum(wins))
    loss_number: int = total_trades - win_number
    win_pct: float = (win_number / total_trades) * 100 if total_trades > 0 else 0
    profit_factor: float = np.sum(wins) / np.sum(losses) if np.sum(losses) > 0 else 0
    largest_loss: float = max(0, np.max(costs)) if len(costs) > 0 else 0
    avg_profit: float = np.mean(-costs) if len(costs) > 0 else 0
    avg_pl: float = np.mean(daily_returns)
    total_pl: float = (trade_log['totals']['capital'].iloc[-1] / initial_capital) * 100

    data = [total_trades, win_number, loss_number, win_pct, largest_loss, profit_factor, avg_profit, avg_pl, total_pl]
    stats = [
        'Total trades', 'Number of wins', 'Number of losses', 'Win %', 'Largest loss', 'Profit factor',
        'Average profit', 'Average P&L %', 'Total P&L %'
    ]
    strat = ['Strategy']
    summary_df = pd.DataFrame(data, stats, strat)

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
        styled = styled.format(fmt, subset=pd.IndexSlice[row_label, :])
    return styled
