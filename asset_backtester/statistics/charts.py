"""Generates charts from a portfolio report"""

import altair as alt


def returns_chart(report):
    # Time interval selector
    time_interval = alt.selection(type='interval', encodings=['x'])

    # Area plot
    areas = alt.Chart().mark_area(opacity=0.7).encode(
        x='index:T',
        y=alt.Y('accumulated return:Q', axis=alt.Axis(format='%')))

    # Nearest point selector
    nearest = alt.selection(type='single',
                            nearest=True,
                            on='mouseover',
                            fields=['index'],
                            empty='none')

    points = areas.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

    # Transparent date selector
    selectors = alt.Chart().mark_point().encode(
        x='index:T',
        opacity=alt.value(0),
    ).add_selection(nearest)

    text = areas.mark_text(
        align='left', dx=5, dy=-5).encode(text=alt.condition(
            nearest, 'accumulated return:Q', alt.value(' '), format='.2%'))

    layered = alt.layer(selectors,
                        points,
                        text,
                        areas.encode(
                            alt.X('index:T',
                                  axis=alt.Axis(title='date'),
                                  scale=alt.Scale(domain=time_interval))),
                        width=700,
                        height=350,
                        title='Wealth over time')

    lower = areas.properties(width=700, height=70).add_selection(time_interval)

    return alt.vconcat(layered, lower, data=report.reset_index())