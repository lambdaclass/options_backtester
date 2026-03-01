"""Visualization: Altair charts for scores, allocations, and P&L."""

from __future__ import annotations

import altair as alt
import pandas as pd


def convexity_scores_chart(scores_df: pd.DataFrame) -> alt.Chart:
    """Line chart of daily convexity ratios over time."""
    data = scores_df.reset_index()
    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("convexity_ratio:Q", title="Convexity Ratio"),
            tooltip=["date:T", "convexity_ratio:Q", "strike:Q", "underlying_price:Q", "implied_vol:Q"],
        )
        .properties(title="Daily Convexity Ratio", width=800, height=300)
    )


def monthly_pnl_chart(records: pd.DataFrame) -> alt.Chart:
    """Bar chart of monthly put P&L."""
    data = records.reset_index()
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("put_pnl:Q", title="Put P&L ($)"),
            color=alt.condition(
                alt.datum.put_pnl > 0,
                alt.value("steelblue"),
                alt.value("salmon"),
            ),
            tooltip=["date:T", "put_pnl:Q", "put_cost:Q", "put_exit_value:Q", "strike:Q", "contracts:Q"],
        )
        .properties(title="Monthly Put P&L", width=800, height=200)
    )


def cumulative_pnl_chart(results: dict[str, pd.DataFrame]) -> alt.Chart:
    """Cumulative portfolio value for multiple strategies."""
    frames = []
    for name, daily_df in results.items():
        df = daily_df[["balance"]].copy()
        df["strategy"] = name
        frames.append(df.reset_index())

    if not frames:
        return alt.Chart(pd.DataFrame()).mark_line()

    data = pd.concat(frames, ignore_index=True)

    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("balance:Q", title="Portfolio Value ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("strategy:N", title="Strategy"),
            tooltip=["date:T", "balance:Q", "strategy:N"],
        )
        .properties(title="Cumulative Portfolio Value", width=800, height=400)
    )
