"""Generates charts from a portfolio report"""

import altair as alt
import pandas as pd


def returns_chart(report):
    # Time interval selector
    time_interval = alt.selection(type='interval', encodings=['x'])

    # Area plot
    areas = alt.Chart().mark_area(opacity=0.7).encode(x='index:T',
                                                      y=alt.Y('accumulated return:Q', axis=alt.Axis(format='%')))

    # Nearest point selector
    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['index'], empty='none')

    points = areas.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

    # Transparent date selector
    selectors = alt.Chart().mark_point().encode(
        x='index:T',
        opacity=alt.value(0),
    ).add_selection(nearest)

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
                        title='Wealth over time')

    lower = areas.properties(width=700, height=70).add_selection(time_interval)

    return alt.vconcat(layered, lower, data=report.reset_index())


def returns_histogram(report):
    bar = alt.Chart(report).mark_bar().encode(x=alt.X('% change:Q',
                                                      bin=alt.BinParams(maxbins=100),
                                                      axis=alt.Axis(format='%')),
                                              y='count():Q')
    return bar


def monthly_returns_heatmap(report):
    resample = report.resample('M')['capital'].last()
    monthly_returns = resample.pct_change().reset_index()
    monthly_returns['capital'].iat[0] = resample.iloc[0] / report.iloc[0]['capital'] - 1
    monthly_returns.columns = ['date', 'capital']

    chart = alt.Chart(monthly_returns).mark_rect().encode(
        alt.X('year(date):O', title='Year'), alt.Y('month(date):O', title='Month'),
        alt.Color('mean(capital)', title='Return', scale=alt.Scale(scheme='redyellowgreen')),
        alt.Tooltip('mean(capital)', format='.2f')).properties(title='Monthly Returns')

    return chart


def sma_graph(data):

    price_chart = alt.Chart(
        data,
        width=700,
        height=350,
    ).mark_line().encode(x='date:T', y=alt.Y('adjClose:Q'), color='symbol:N', opacity=alt.value(0.3))
    sma_chart = alt.Chart(
        data,
        width=700,
        height=350,
    ).mark_line(strokeDash=[1, 1]).encode(x='date:T', y=alt.Y('sma:Q'), color='symbol:N')
    return price_chart + sma_chart