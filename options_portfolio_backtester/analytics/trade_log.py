"""Structured trade log — replaces MultiIndex trade log with per-trade P&L."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from options_portfolio_backtester.core.types import Order


@dataclass
class Trade:
    """A single round-trip trade (entry + exit).

    ``entry_price`` / ``exit_price`` are per-share prices of the (first) leg.
    P&L follows the entry order: a BTO trade gains when the price rises, an
    STO trade when it falls. For multi-leg positions set ``entry_cost`` /
    ``exit_cost``, the engine's signed per-unit cash flows summed over legs
    (positive = paid, in dollars per position unit); P&L is then
    ``-(entry_cost + exit_cost) * quantity``.
    """
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
    entry_cost: float | None = None
    exit_cost: float | None = None

    @property
    def is_short(self) -> bool:
        return str(getattr(self.entry_order, "value", self.entry_order)) == Order.STO.value

    @property
    def gross_pnl(self) -> float:
        """P&L before commissions."""
        if self.entry_cost is not None and self.exit_cost is not None:
            return -(self.entry_cost + self.exit_cost) * self.quantity
        sign = -1.0 if self.is_short else 1.0
        return sign * (self.exit_price - self.entry_price) * self.quantity * self.shares_per_contract

    @property
    def net_pnl(self) -> float:
        """P&L after commissions."""
        return self.gross_pnl - self.entry_commission - self.exit_commission

    @property
    def return_pct(self) -> float:
        """Net P&L as a fraction of the entry premium (paid or received)."""
        if self.entry_cost is not None:
            entry_cost = abs(self.entry_cost * self.quantity)
        else:
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
    def from_legacy_trade_log(cls, trade_log: pd.DataFrame, shares_per_contract: int = 100,
                              cost_model: Any = None) -> "TradeLog":
        """Round trips from the engine's MultiIndex trade log.

        A position is identified by its legs' contracts; each exit closes the
        oldest open entry of the same position (FIFO). P&L uses the signed
        position costs (``totals.cost``), so short and multi-leg positions
        are right. With ``cost_model`` (the run's), commissions are computed
        the way the engine charged them; otherwise they are 0.
        """
        tl = cls()
        if trade_log is None or trade_log.empty:
            return tl

        legs = [c for c in cast(pd.MultiIndex, trade_log.columns).levels[0] if c != "totals"]
        legs = [c for c in legs if c in trade_log.columns.get_level_values(0)]
        first = legs[0]
        orders = trade_log[first]["order"].map(lambda o: str(getattr(o, "value", o)))
        is_entry = orders.isin({Order.BTO.value, Order.STO.value})

        def commission(cost: float, qty: float) -> float:
            if cost_model is None:
                return 0.0
            return float(cost_model.option_cost(abs(cost * qty), qty, shares_per_contract))

        open_entries: dict[tuple, list[Any]] = {}
        for i in range(len(trade_log)):
            row = trade_log.iloc[i]
            key = tuple(row[leg]["contract"] for leg in legs)
            if is_entry.iloc[i]:
                open_entries.setdefault(key, []).append(row)
                continue
            queue = open_entries.get(key)
            if not queue:
                continue  # exit without a recorded entry
            e_row = queue.pop(0)
            qty = float(e_row["totals"]["qty"])
            entry_cost = float(e_row["totals"]["cost"])
            exit_cost = float(row["totals"]["cost"])
            tl.add_trade(Trade(
                contract=str(e_row[first]["contract"]) if len(legs) == 1
                else "/".join(str(c) for c in key),
                underlying=e_row[first]["underlying"],
                option_type=e_row[first]["type"],
                strike=e_row[first]["strike"],
                entry_date=e_row["totals"]["date"],
                exit_date=row["totals"]["date"],
                entry_price=abs(float(e_row[first]["cost"])) / shares_per_contract,
                exit_price=abs(float(row[first]["cost"])) / shares_per_contract,
                quantity=int(qty),
                shares_per_contract=shares_per_contract,
                entry_order=e_row[first]["order"],
                exit_order=row[first]["order"],
                entry_commission=commission(entry_cost, qty),
                exit_commission=commission(exit_cost, qty),
                entry_cost=entry_cost,
                exit_cost=exit_cost,
            ))
        return tl

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a flat DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            entry_order = getattr(t.entry_order, "value", t.entry_order)
            exit_order = getattr(t.exit_order, "value", t.exit_order)
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
                "shares_per_contract": t.shares_per_contract,
                "entry_order": entry_order,
                "exit_order": exit_order,
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
