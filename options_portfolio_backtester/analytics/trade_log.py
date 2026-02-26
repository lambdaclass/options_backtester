"""Structured trade log — replaces MultiIndex trade log with per-trade P&L."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from options_portfolio_backtester.core.types import Order


@dataclass
class Trade:
    """A single round-trip trade (entry + exit)."""
    contract: str
    underlying: str
    option_type: str
    strike: float
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    quantity: int
    shares_per_contract: int
    entry_order: Order
    exit_order: Order
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    @property
    def gross_pnl(self) -> float:
        """P&L before commissions."""
        return (self.exit_price - self.entry_price) * self.quantity * self.shares_per_contract

    @property
    def net_pnl(self) -> float:
        """P&L after commissions."""
        return self.gross_pnl - self.entry_commission - self.exit_commission

    @property
    def return_pct(self) -> float:
        """Return as percentage of entry cost."""
        entry_cost = abs(self.entry_price * self.quantity * self.shares_per_contract)
        if entry_cost == 0:
            return 0.0
        return self.net_pnl / entry_cost


class TradeLog:
    """Structured collection of round-trip trades with analysis methods."""

    def __init__(self) -> None:
        self.trades: list[Trade] = []

    def add_trade(self, trade: Trade) -> None:
        self.trades.append(trade)

    @classmethod
    def from_legacy_trade_log(cls, trade_log: pd.DataFrame,
                              shares_per_contract: int = 100) -> "TradeLog":
        """Build a TradeLog from the legacy MultiIndex trade_log DataFrame."""
        tl = cls()
        if trade_log.empty:
            return tl

        first_leg: str = trade_log.columns.levels[0][0]

        order_bto = Order.BTO
        order_sto = Order.STO
        entry_mask = trade_log[first_leg].eval(
            "(order == @order_bto) | (order == @order_sto)"
        )
        entries = trade_log.loc[entry_mask]
        exits = trade_log.loc[~entry_mask]

        for contract in entries[first_leg]["contract"]:
            entry = entries.loc[entries[first_leg]["contract"] == contract]
            exit_ = exits.loc[exits[first_leg]["contract"] == contract]
            if entry.empty or exit_.empty:
                continue
            try:
                e_row = entry.iloc[0]
                x_row = exit_.iloc[0]
                trade = Trade(
                    contract=contract,
                    underlying=e_row[first_leg]["underlying"],
                    option_type=e_row[first_leg]["type"],
                    strike=e_row[first_leg]["strike"],
                    entry_date=e_row["totals"]["date"],
                    exit_date=x_row["totals"]["date"],
                    entry_price=abs(e_row[first_leg]["cost"]) / shares_per_contract,
                    exit_price=abs(x_row[first_leg]["cost"]) / shares_per_contract,
                    quantity=int(e_row["totals"]["qty"]),
                    shares_per_contract=shares_per_contract,
                    entry_order=e_row[first_leg]["order"],
                    exit_order=x_row[first_leg]["order"],
                )
                tl.add_trade(trade)
            except (IndexError, KeyError):
                continue

        return tl

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a flat DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "contract": t.contract,
                "underlying": t.underlying,
                "type": t.option_type,
                "strike": t.strike,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "return_pct": t.return_pct,
                "entry_commission": t.entry_commission,
                "exit_commission": t.exit_commission,
            })
        return pd.DataFrame(rows)

    @property
    def net_pnls(self) -> np.ndarray:
        return np.array([t.net_pnl for t in self.trades])

    @property
    def winners(self) -> list[Trade]:
        return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losers(self) -> list[Trade]:
        return [t for t in self.trades if t.net_pnl <= 0]

    def __len__(self) -> int:
        return len(self.trades)
