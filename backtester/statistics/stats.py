import pandas as pd
import numpy as np

from ..enums import Order  # noqa: F401


def summary(trade_log, balance):
    """Returns a table with summary statistics about the trade log"""

    initial_capital = balance['total capital'].get(0)
    trade_log.loc[:,
                  ('totals',
                   'capital')] = (-trade_log['totals']['cost'] * trade_log['totals']['qty']).cumsum() + initial_capital

    daily_returns = balance['% change'] * 100

    first_leg = trade_log.columns.levels[0][0]

    entry_mask = trade_log[first_leg].eval('(order == @Order.BTO) | (order == @Order.STO)')
    entries = trade_log.loc[entry_mask]
    exits = trade_log.loc[~entry_mask]

    costs = np.array([])
    for contract in entries[first_leg]['contract']:
        entry = entries.loc[entries[first_leg]['contract'] == contract]
        exit_ = exits.loc[exits[first_leg]['contract'] == contract]
        try:
            # Here we assume we are entering only once per contract (i.e both entry and exit_ have only one row)
            costs = np.append(costs, (entry['totals']['cost'] * entry['totals']['qty']).values[0] +
                              (exit_['totals']['cost'] * exit_['totals']['qty']).values[0])
        except IndexError:
            continue

    wins = costs < 0
    losses = costs >= 0
    profit_factor = np.sum(wins) / np.sum(losses)
    total_trades = len(exits)
    win_number = np.sum(wins)
    loss_number = total_trades - win_number
    win_pct = (win_number / total_trades) * 100
    largest_loss = max(0, np.max(costs))
    avg_profit = np.mean(-costs)
    avg_pl = np.mean(daily_returns)
    total_pl = (trade_log['totals']['capital'].iloc[-1] / initial_capital) * 100

    data = [total_trades, win_number, loss_number, win_pct, largest_loss, profit_factor, avg_profit, avg_pl, total_pl]
    stats = [
        'Total trades', 'Number of wins', 'Number of losses', 'Win %', 'Largest loss', 'Profit factor',
        'Average profit', 'Average P&L %', 'Total P&L %'
    ]
    strat = ['Strategy']
    summary = pd.DataFrame(data, stats, strat)

    # Applies formatters to rows
    def format_row_wise(styler, formatters):
        for row, row_formatter in formatters.items():
            row_num = styler.index.get_loc(row)

            for col_num in range(len(styler.columns)):
                styler._display_funcs[(row_num, col_num)] = row_formatter

        return styler

    formatters = {
        "Total trades": lambda x: f"{x:.0f}",
        "Number of wins": lambda x: f"{x:.0f}",
        "Number of losses": lambda x: f"{x:.0f}",
        "Win %": lambda x: f"{x:.2f}%",
        "Largest loss": lambda x: f"${x:.2f}",
        "Profit factor": lambda x: f"{x:.2f}",
        "Average profit": lambda x: f"${x:.2f}",
        "Average P&L %": lambda x: f"{x:.2f}%",
        "Total P&L %": lambda x: f"{x:.2f}%"
    }

    return format_row_wise(summary.style, formatters)
