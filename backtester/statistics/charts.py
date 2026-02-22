"""Generates charts from a portfolio report"""
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
