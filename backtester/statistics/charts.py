"""Generates charts from a portfolio report"""

import altair as alt


def returns_chart(report):
    # Time interval selector
    time_interval = alt.selection(type="interval", encodings=["x"])

    # Area plot
    areas = alt.Chart().mark_area(opacity=0.6).encode(
        x=alt.X(
            "index:T",
            axis=alt.Axis(title="Date"),
            scale={"domain": time_interval.ref()}),
        y=alt.Y("% Price:Q", axis=alt.Axis(format="%")))

    # Nearest point selector
    nearest = alt.selection(
        type="single",
        nearest=True,
        on="mouseover",
        fields=["index"],
        empty="none")

    points = areas.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

    # Transparent date selector
    selectors = alt.Chart().mark_point().encode(
        x="index:T",
        opacity=alt.value(0),
    ).add_selection(nearest)

    text = areas.mark_text(
        align="left", dx=5, dy=-5).encode(
            text=alt.condition(nearest, "% Price:Q", alt.value(" ")))

    layered = alt.layer(
        areas,
        selectors,
        points,
        text,
        width=700,
        height=350,
        title="Returns over time")

    lower = areas.properties(width=700, height=70).add_selection(time_interval)

    chart = alt.vconcat(layered, lower, data=report.reset_index())

    return chart


def returns_histogram(report):
    bar = alt.Chart(report).mark_bar().encode(
        x=alt.X(
            "Interval Change:Q",
            bin=alt.BinParams(maxbins=100),
            axis=alt.Axis(format='%')),
        y="count():Q")
    return bar


def monthly_returns_heatmap(report):
    monthly_returns = report.resample(
        "M")["Total Portfolio"].last().pct_change().reset_index()
    monthly_returns.columns = ["Date", "Monthly Returns"]

    chart = alt.Chart(monthly_returns).mark_rect().encode(
        alt.X("year(Date):O", title="Year"),
        alt.Y("month(Date):O", title="Month"),
        alt.Color(
            "mean(Monthly Returns)",
            title="Return",
            scale=alt.Scale(scheme="redyellowgreen")),
        alt.Tooltip("mean(Monthly Returns)",
                    format=".2f")).properties(title="Average Monthly Returns")

    return chart
