"""Interest on cash and on borrowing, applied to a finished backtest.

The engine itself pays no interest on positive cash and charges none on
negative cash, so a levered run (stocks above 100%, financed by negative
cash) looks better than it is, and a cash-heavy one looks worse. This
module adjusts a balance curve after the fact: each day, the previous day's
cash earns ``rate`` (if positive) or pays ``rate + borrow_spread`` (if
negative), and the interest compounds into total capital.

It is a first-order adjustment: interest changes NAV but not the position
sizes the engine chose between rebalances. For small leverage or cash
buffers the difference is negligible. Margin calls are not modeled.
"""

from __future__ import annotations

import pandas as pd


def apply_financing(balance: pd.DataFrame, rate: pd.Series, borrow_spread: float = 0.0) -> pd.Series:
    """Total capital after crediting / charging interest on the cash balance.

    ``balance`` is an engine's ``balance`` frame (``"total capital"`` and
    ``"cash"`` columns, date index). ``rate`` is an annual rate as a decimal
    (e.g. the 3-month T-bill, 0.05 = 5%), indexed by date; it is
    forward-filled onto the balance dates. Interest accrues on calendar days
    (actual/365) from one balance row to the next.
    """
    total, cash = balance["total capital"], balance["cash"]
    r = rate.sort_index().reindex(total.index, method="ffill").bfill()
    r = r.where(cash.shift() >= 0, r + borrow_spread)
    dt = total.index.to_series().diff().dt.days.fillna(0) / 365.0
    interest = cash.shift().fillna(0) * r.shift().fillna(0) * dt
    return total * (1 + interest / total.shift().fillna(total)).cumprod()
