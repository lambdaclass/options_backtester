"""Charts — preserved Altair charts + matplotlib additions."""

from __future__ import annotations

import pandas as pd

# Re-export original chart functions
from backtester.statistics.charts import (  # noqa: F401
    returns_chart,
    returns_histogram,
    monthly_returns_heatmap,
)

__all__ = ["returns_chart", "returns_histogram", "monthly_returns_heatmap", "weights_chart"]


def weights_chart(balance: pd.DataFrame, figsize: tuple[float, float] = (12, 6)):
    """Stacked area chart of portfolio weights over time.

    Expects a balance DataFrame with ``{symbol} qty`` columns and a
    ``total capital`` column (as produced by ``AlgoPipelineBacktester``).

    Returns ``(fig, ax)`` from matplotlib.
    """
    import matplotlib.pyplot as plt

    qty_cols = [c for c in balance.columns if c.endswith(" qty")]
    if not qty_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Portfolio Weights (no positions found)")
        return fig, ax

    symbols = [c.replace(" qty", "") for c in qty_cols]
    total = balance["total capital"]

    # Compute weights: qty * price / total_capital
    # We don't have price columns directly, but stocks capital is available.
    # Reconstruct per-symbol value: qty * (total - cash) is aggregate,
    # so we estimate from qty shares of total stock value.
    weights = pd.DataFrame(index=balance.index)
    for sym, col in zip(symbols, qty_cols):
        weights[sym] = balance[col].fillna(0)

    # Normalize to weights (proportional share of total qty-weighted value)
    row_sums = weights.abs().sum(axis=1)
    row_sums = row_sums.replace(0, 1)  # avoid division by zero
    # If we have cash and total capital, use stock fraction
    if "cash" in balance.columns:
        stock_fraction = 1.0 - balance["cash"] / total.replace(0, 1)
        for sym in symbols:
            weights[sym] = (weights[sym] / row_sums) * stock_fraction
    else:
        weights = weights.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(weights.index, *[weights[s] for s in symbols], labels=symbols, alpha=0.8)
    ax.set_title("Portfolio Weights Over Time")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize="small")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, ax
