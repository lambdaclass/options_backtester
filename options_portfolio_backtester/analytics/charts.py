"""Charts — Altair charts + matplotlib additions."""

from __future__ import annotations

from typing import cast

import altair as alt
import numpy as np
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


def returns_histogram(report: pd.DataFrame) -> alt.LayerChart:
    # Pre-bin in numpy instead of letting vega-lite bin client-side: keeps
    # exact counts while staying under altair's 5000-row spec limit on
    # multi-decade daily backtests.
    rets = report['% change'].dropna()
    counts, edges = (np.histogram(rets, bins=100) if len(rets)
                     else (np.array([], dtype=int), np.array([0.0, 0.0])))
    data = pd.DataFrame({
        'bin_start': edges[:-1],
        'bin_end': edges[1:],
        'count': counts,
    })
    bars = alt.Chart(data).mark_bar(color='steelblue', opacity=0.8).encode(
        x=alt.X('bin_start:Q', bin='binned', axis=alt.Axis(format='%'),
                title='% change'),
        x2='bin_end:Q',
        # Symlog counts: on a linear axis the peak at zero crushes the tail
        # observations — the fat tails are the whole point of looking at
        # this distribution.
        y=alt.Y('count:Q', title='Days (symlog)',
                scale=alt.Scale(type='symlog')),
    )
    layers = [bars]
    if len(rets) > 1 and rets.std() > 0:
        # Fitted-normal overlay: expected bin counts under a Gaussian with
        # the sample mean/std. The gap between the dashed line and the bars
        # in the tails is the leptokurtosis, made visible.
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        mu, sigma = rets.mean(), rets.std()
        expected = (len(rets) * width / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((centers - mu) / sigma) ** 2))
        normal = alt.Chart(pd.DataFrame({'x': centers, 'expected': expected})) \
            .mark_line(color='black', strokeDash=[6, 4]).encode(
                x='x:Q', y='expected:Q')
        layers.append(normal)
    return cast(alt.LayerChart, alt.layer(*layers).properties(
        title='Daily return distribution vs fitted normal'))


def monthly_returns_heatmap(report: pd.DataFrame) -> alt.LayerChart:
    resample = report.resample('ME')['total capital'].last()
    monthly_returns = resample.pct_change().reset_index()
    monthly_returns.loc[monthly_returns.index[0], 'total capital'] = (
        resample.iloc[0] / report.iloc[0]['total capital'] - 1)
    monthly_returns.columns = ['date', 'ret']
    monthly_returns['label'] = monthly_returns['ret'] * 100

    # pyfolio convention: RdYlGn diverging colormap centered at zero, cell
    # values annotated as percent×100 with no unit (compact enough to fit).
    # The color domain is clamped to ±10%: a single crash-payoff month
    # (e.g. +73% in Nov 2008) would otherwise stretch the scale until every
    # normal month rendered as the same pale yellow. Outliers saturate; the
    # tooltip keeps the exact value.
    base = alt.Chart(monthly_returns).encode(
        alt.X('year(date):O', title='Year'),
        alt.Y('month(date):O', title='Month'),
    )
    rects = base.mark_rect().encode(
        alt.Color('ret:Q', title='Return',
                  scale=alt.Scale(scheme='redyellowgreen', domainMid=0,
                                  domain=[-0.10, 0.10], clamp=True),
                  legend=alt.Legend(format='.0%')),
        alt.Tooltip('ret:Q', format='.2%'),
    )
    labels = base.mark_text(fontSize=8).encode(
        alt.Text('label:Q', format='.1f'),
        color=alt.condition('abs(datum.ret) > 0.065',
                            alt.value('white'), alt.value('#333')),
    )
    return (rects + labels).properties(
        width=700, height=320, title='Monthly returns (%)')


def annual_returns_chart(balance: pd.DataFrame) -> alt.LayerChart:
    """Horizontal annual-returns bars, pyfolio style (steelblue, dashed mean)."""
    total = balance['total capital'].dropna()
    yearly = total.groupby(pd.DatetimeIndex(total.index).year).agg(['first', 'last'])
    returns = (yearly['last'] / yearly['first'] - 1).rename('return')
    data = returns.rename_axis('year').reset_index()

    bars = alt.Chart(data).mark_bar(color='steelblue', opacity=0.7).encode(
        x=alt.X('return:Q', title='Return', axis=alt.Axis(format='%')),
        y=alt.Y('year:O', title='Year'),
        tooltip=['year:O', alt.Tooltip('return:Q', format='.2%')],
    )
    mean = alt.Chart(pd.DataFrame({'mean': [returns.mean()]})).mark_rule(
        color='steelblue', strokeDash=[6, 4], strokeWidth=2).encode(x='mean:Q')
    zero = alt.Chart(pd.DataFrame({'zero': [0.0]})).mark_rule(
        color='black', strokeWidth=2).encode(x='zero:Q')
    return (bars + mean + zero).properties(
        width=700, height=280, title='Annual returns')


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


# Altair refuses to serialize specs with more than 5000 inline rows, and a
# multi-decade daily backtest exceeds that. Line/area panels are visually
# identical after even subsampling, so thin them before charting.
_MAX_CHART_ROWS = 2000


def thin_for_chart(data, max_rows: int = _MAX_CHART_ROWS):
    """Evenly subsample a Series/DataFrame to at most ``max_rows`` rows,
    always keeping the first and last row."""
    n = len(data)
    if n <= max_rows:
        return data
    step = -(-n // max_rows)  # ceil division
    thinned = data.iloc[::step]
    if thinned.index[-1] != data.index[-1]:
        thinned = pd.concat([thinned, data.iloc[[-1]]])
    return thinned


def equity_curve_chart(balance: pd.DataFrame,
                       benchmark_balance: pd.DataFrame | None = None,
                       log_scale: bool = False,
                       drawdown_periods: pd.DataFrame | None = None) -> alt.LayerChart:
    """Indexed equity curve (start = 1.0), optionally overlaid with a benchmark.

    ``drawdown_periods`` (the frame from ``tearsheet.top_drawdowns``) shades
    the worst drawdown episodes as translucent bands, pyfolio's
    drawdown-periods view folded into the main chart.
    """
    def _indexed(bal: pd.DataFrame, label: str) -> pd.DataFrame:
        total = thin_for_chart(bal["total capital"].dropna())
        return pd.DataFrame({
            "date": total.index,
            "growth": total / total.iloc[0],
            "series": label,
        })

    frames = [_indexed(balance, "strategy")]
    if benchmark_balance is not None and not benchmark_balance.empty:
        frames.append(_indexed(benchmark_balance, "benchmark"))
    data = pd.concat(frames, ignore_index=True)

    # pyfolio convention: forestgreen strategy (lw 3), gray benchmark (lw 2),
    # dashed black reference at 1.0.
    scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    lines = alt.Chart(data).mark_line(opacity=0.8).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("growth:Q", title="Growth of $1", scale=scale),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["strategy", "benchmark"],
                                        range=["forestgreen", "gray"])),
        strokeWidth=alt.StrokeWidth(
            "series:N", legend=None,
            scale=alt.Scale(domain=["strategy", "benchmark"], range=[3, 2])),
        tooltip=["date:T", alt.Tooltip("growth:Q", format=".3f"), "series:N"],
    )
    ref = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        color="black", strokeDash=[6, 4]).encode(y="y:Q")
    layer_list = [lines, ref]
    if drawdown_periods is not None and not drawdown_periods.empty:
        bands = drawdown_periods[["peak", "recovery", "depth"]].copy()
        bands["recovery"] = bands["recovery"].fillna(data["date"].max())
        shade = alt.Chart(bands).mark_rect(color="coral", opacity=0.15).encode(
            x="peak:T", x2="recovery:T",
            tooltip=[alt.Tooltip("peak:T"), alt.Tooltip("recovery:T"),
                     alt.Tooltip("depth:Q", title="Depth (%)")],
        )
        layer_list.insert(0, shade)
    return cast(alt.LayerChart, alt.layer(*layer_list).properties(
        width=700, height=300, title="Equity curve"))


def underwater_chart(drawdown: pd.Series) -> alt.Chart:
    """Underwater plot from a drawdown series (values in [-1, 0])."""
    data = thin_for_chart(drawdown).rename("drawdown").rename_axis("date").reset_index()
    return alt.Chart(data).mark_area(opacity=0.7, color="coral").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
        tooltip=["date:T", alt.Tooltip("drawdown:Q", format=".2%")],
    ).properties(width=700, height=180, title="Underwater plot")


def rolling_sharpe_chart(balance: pd.DataFrame, window: int = 252) -> alt.Chart:
    """Rolling annualized Sharpe (0% risk-free) of daily total-capital returns."""
    rets = balance["total capital"].pct_change().dropna()
    mean = rets.rolling(window).mean()
    std = rets.rolling(window).std()
    sharpe = (mean / std) * (252 ** 0.5)
    data = thin_for_chart(sharpe.dropna()).rename("sharpe").rename_axis("date").reset_index()
    # pyfolio convention: orangered series, dashed steelblue mean line.
    line = alt.Chart(data).mark_line(color="orangered", opacity=0.7,
                                     strokeWidth=2.5).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("sharpe:Q", title=f"Rolling Sharpe ({window}d)"),
        tooltip=["date:T", alt.Tooltip("sharpe:Q", format=".2f")],
    )
    mean_rule = alt.Chart(pd.DataFrame({"mean": [data["sharpe"].mean()]})).mark_rule(
        color="steelblue", strokeDash=[6, 4], strokeWidth=2).encode(y="mean:Q")
    return (line + mean_rule).properties(
        width=700, height=180, title=f"Rolling Sharpe ({window}-day)")


def rolling_volatility_chart(balance: pd.DataFrame, window: int = 126) -> alt.Chart:
    """Rolling annualized volatility of daily total-capital returns."""
    rets = balance["total capital"].pct_change().dropna()
    vol = rets.rolling(window).std() * (252 ** 0.5)
    data = thin_for_chart(vol.dropna()).rename("volatility").rename_axis("date").reset_index()
    line = alt.Chart(data).mark_line(color="orangered", opacity=0.7,
                                     strokeWidth=2.5).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("volatility:Q", title=f"Rolling volatility ({window}d)", axis=alt.Axis(format="%")),
        tooltip=["date:T", alt.Tooltip("volatility:Q", format=".2%")],
    )
    mean_rule = alt.Chart(pd.DataFrame({"mean": [data["volatility"].mean()]})).mark_rule(
        color="steelblue", strokeDash=[6, 4], strokeWidth=2).encode(y="mean:Q")
    return (line + mean_rule).properties(
        width=700, height=180, title=f"Rolling volatility ({window}-day, annualized)")


def weights_area_chart(balance: pd.DataFrame) -> alt.Chart:
    """Stacked area chart of capital allocation (stocks / options / cash).

    Altair counterpart of :func:`weights_chart`; uses the capital columns
    directly instead of reconstructing weights from quantities.
    """
    cols = [c for c in ("stocks capital", "options capital", "cash") if c in balance.columns]
    if not cols:
        # Fall back to an empty chart rather than raising; the tearsheet
        # assembler skips panels whose chart has no rows.
        return alt.Chart(pd.DataFrame({"date": [], "weight": [], "component": []})).mark_area()
    total = balance["total capital"].replace(0, pd.NA)
    # melt multiplies rows by len(cols); thin the wide frame accordingly
    weights = thin_for_chart(balance[cols].div(total, axis=0).dropna(),
                             max_rows=_MAX_CHART_ROWS // len(cols))
    data = weights.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="component", value_name="weight")
    return alt.Chart(data).mark_area(opacity=0.8).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("weight:Q", title="Share of capital", stack="normalize", axis=alt.Axis(format="%")),
        color=alt.Color("component:N", title=None),
        tooltip=["date:T", "component:N", alt.Tooltip("weight:Q", format=".2%")],
    ).properties(width=700, height=200, title="Capital allocation over time")


def apply_pyfolio_style(chart):
    """Top-level chart config approximating pyfolio's seaborn look:
    no frame, light grid, muted axis chrome, left-anchored titles."""
    return chart.configure_view(stroke=None).configure_axis(
        gridColor="#e8e8e8", domainColor="#999999",
        tickColor="#999999", labelColor="#333333", titleColor="#333333",
    ).configure_title(
        anchor="start", fontSize=14, color="#333333",
    ).configure_legend(labelColor="#333333", titleColor="#333333")


__all__ = [
    "thin_for_chart",
    "apply_pyfolio_style",
    "returns_chart",
    "returns_histogram",
    "monthly_returns_heatmap",
    "annual_returns_chart",
    "weights_chart",
    "equity_curve_chart",
    "underwater_chart",
    "rolling_sharpe_chart",
    "rolling_volatility_chart",
    "weights_area_chart",
]
