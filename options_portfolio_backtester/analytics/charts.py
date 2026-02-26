"""Charts — Altair charts + matplotlib additions."""

from __future__ import annotations

import altair as alt
import pandas as pd


def returns_chart(report: pd.DataFrame) -> alt.VConcatChart:
    # Time interval selector
    time_interval = alt.selection_interval(encodings=['x'])

    # Area plot
    areas = alt.Chart().mark_area(opacity=0.7).encode(x='index:T',
                                                      y=alt.Y('accumulated return:Q', axis=alt.Axis(format='%')))

    # Nearest point selector
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['index'])

    points = areas.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

    # Transparent date selector
    selectors = alt.Chart().mark_point().encode(
        x='index:T',
        opacity=alt.value(0),
    ).add_params(nearest)

    text = areas.mark_text(
        align='left', dx=5,
        dy=-5).encode(text=alt.condition(nearest, 'accumulated return:Q', alt.value(' '), format='.2%'))

    layered = alt.layer(selectors,
                        points,
                        text,
                        areas.encode(
                            alt.X('index:T', axis=alt.Axis(title='date'), scale=alt.Scale(domain=time_interval))),
                        width=700,
                        height=350,
                        title='Returns over time')

    lower = areas.properties(width=700, height=70).add_params(time_interval)

    return alt.vconcat(layered, lower, data=report.reset_index())


def returns_histogram(report: pd.DataFrame) -> alt.Chart:
    bar = alt.Chart(report).mark_bar().encode(x=alt.X('% change:Q',
                                                      bin=alt.BinParams(maxbins=100),
                                                      axis=alt.Axis(format='%')),
                                              y='count():Q')
    return bar


def monthly_returns_heatmap(report: pd.DataFrame) -> alt.Chart:
    resample = report.resample('ME')['total capital'].last()
    monthly_returns = resample.pct_change().reset_index()
    monthly_returns.loc[monthly_returns.index[0], 'total capital'] = resample.iloc[0] / report.iloc[0]['total capital'] - 1
    monthly_returns.columns = ['date', 'total capital']

    chart = alt.Chart(monthly_returns).mark_rect().encode(
        alt.X('year(date):O', title='Year'), alt.Y('month(date):O', title='Month'),
        alt.Color('mean(total capital)', title='Return', scale=alt.Scale(scheme='redyellowgreen')),
        alt.Tooltip('mean(total capital)', format='.2f')).properties(title='Monthly Returns')

    return chart


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


__all__ = ["returns_chart", "returns_histogram", "monthly_returns_heatmap", "weights_chart"]
