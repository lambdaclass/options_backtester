"""Options-specific tearsheet panels.

These are the charts no generic returns library provides: P&L attribution
between the equity and options legs, premium spend against the configured
budget, crash-window zooms, and the long-left-cluster / rare-huge-winner
trade distribution that deep-OTM puts produce.

All functions take the engine's ``balance`` frame and/or a *flat* trade
DataFrame (``TradeLog.to_dataframe()`` schema). Use :func:`normalize_trade_log`
to accept whatever the caller has — a :class:`TradeLog`, a flat frame, or the
legacy MultiIndex trade log returned by ``BacktestEngine.run()``.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from options_portfolio_backtester.analytics.charts import thin_for_chart
from options_portfolio_backtester.analytics.trade_log import TradeLog

SHARES_PER_CONTRACT = 100

# Default event windows for crash zoom panels (label -> (start, end)).
DEFAULT_CRASH_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC 2008-09": ("2008-09-01", "2009-06-30"),
    "COVID 2020": ("2020-02-01", "2020-06-30"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}


def normalize_trade_log(trade_log) -> pd.DataFrame:
    """Coerce any trade-log representation to the flat per-trade DataFrame.

    Accepts a :class:`TradeLog`, an already-flat DataFrame (must have a
    ``net_pnl`` column), the legacy MultiIndex frame from ``engine.run()``,
    or ``None``. Returns an empty frame when there is nothing to convert.
    """
    if trade_log is None:
        return pd.DataFrame()
    if isinstance(trade_log, TradeLog):
        return trade_log.to_dataframe()
    if isinstance(trade_log, pd.DataFrame):
        if trade_log.empty:
            return pd.DataFrame()
        if isinstance(trade_log.columns, pd.MultiIndex):
            return TradeLog.from_legacy_trade_log(trade_log).to_dataframe()
        if "net_pnl" in trade_log.columns:
            return trade_log
    raise TypeError(f"Unsupported trade log type: {type(trade_log)!r}")


def _signed_cumulative(trade_df: pd.DataFrame, date_col: str, price_col: str,
                       order_col: str, debit_orders: tuple[str, ...]) -> pd.Series:
    """Cumulative signed cash flow of one trade side, indexed by date."""
    multiplier = (trade_df["shares_per_contract"]
                  if "shares_per_contract" in trade_df.columns
                  else SHARES_PER_CONTRACT)
    if order_col in trade_df.columns:
        sign = trade_df[order_col].map(
            lambda o: 1.0 if str(o).replace("Order.", "") in debit_orders else -1.0)
    else:
        sign = pd.Series(1.0, index=trade_df.index)
    flows = (trade_df[price_col].abs() * trade_df["quantity"].abs()
             * multiplier * sign)
    series = pd.Series(flows.values,
                       index=pd.to_datetime(trade_df[date_col]).dt.normalize())
    return series.groupby(level=0).sum().sort_index().cumsum()


def options_pnl_decomposition_chart(trade_df: pd.DataFrame,
                                    balance: pd.DataFrame) -> alt.Chart:
    """The bleed-vs-payoff picture, built from actual trades.

    Three lines as a share of initial capital: cumulative premium paid
    (plotted negative — the drag), cumulative payoffs received, and their
    net. This is real P&L attribution for the options leg; balance-column
    deltas can't provide it because they conflate rebalancing flows with
    P&L (day-one cash invested into stocks shows up as a −100% "loss").
    """
    if trade_df.empty or balance.empty:
        return alt.Chart(pd.DataFrame({"date": [], "value": [], "series": []})).mark_line()
    initial = balance["total capital"].dropna().iloc[0]
    paid = _signed_cumulative(trade_df, "entry_date", "entry_price",
                              "entry_order", debit_orders=("BTO",))
    received = _signed_cumulative(trade_df, "exit_date", "exit_price",
                                  "exit_order", debit_orders=("BTC",)) * -1.0
    idx = paid.index.union(received.index)
    paid = paid.reindex(idx).ffill().fillna(0.0)
    received = received.reindex(idx).ffill().fillna(0.0)
    frame = pd.DataFrame({
        "Premium paid (drag)": -paid / initial,
        "Payoffs received": received / initial,
        "Net options P&L": (received - paid) / initial,
    }, index=idx)
    data = thin_for_chart(frame, max_rows=2000 // 3).rename_axis("date") \
        .reset_index().melt(id_vars="date", var_name="series", value_name="value")
    return alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Cumulative (share of initial capital)",
                axis=alt.Axis(format="%")),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(
                            domain=["Premium paid (drag)", "Payoffs received",
                                    "Net options P&L"],
                            range=["coral", "forestgreen", "steelblue"])),
        tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".2%")],
    ).properties(width=700, height=250,
                 title="Options leg: premium bleed vs crash payoffs")


def trade_return_histogram(trade_df: pd.DataFrame) -> alt.Chart:
    """Distribution of per-trade return on premium (×), symlog counts.

    For deep-OTM puts this is the signature shape: a mass at −1× (total
    premium loss) and a thin tail of rare large multiples.
    """
    if trade_df.empty or "return_pct" not in trade_df.columns:
        return alt.Chart(pd.DataFrame({"bin_start": [], "count": []})).mark_bar()
    rets = trade_df["return_pct"].dropna()
    counts, edges = np.histogram(rets, bins=40)
    data = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:],
                         "count": counts})
    return alt.Chart(data).mark_bar(color="steelblue", opacity=0.8).encode(
        x=alt.X("bin_start:Q", bin="binned", title="Return on premium (×)",
                axis=alt.Axis(format=".1f")),
        x2="bin_end:Q",
        y=alt.Y("count:Q", title="Trades (symlog)",
                scale=alt.Scale(type="symlog")),
        tooltip=[alt.Tooltip("bin_start:Q", format=".2f"),
                 alt.Tooltip("count:Q")],
    ).properties(width=700, height=200,
                 title="Per-trade return on premium (distribution)")


def holding_period_chart(trade_df: pd.DataFrame) -> alt.Chart:
    """Distribution of days between entry and exit."""
    if trade_df.empty:
        return alt.Chart(pd.DataFrame({"holding_days": []})).mark_bar()
    data = trade_df[["entry_date", "exit_date"]].copy()
    data["entry_date"] = pd.to_datetime(data["entry_date"])
    data["exit_date"] = pd.to_datetime(data["exit_date"])
    data["holding_days"] = (data["exit_date"] - data["entry_date"]).dt.days
    return alt.Chart(data).mark_bar(color="steelblue", opacity=0.8).encode(
        x=alt.X("holding_days:Q", bin=alt.Bin(maxbins=30), title="Holding period (days)"),
        y=alt.Y("count():Q", title="Trades"),
        tooltip=[alt.Tooltip("count():Q", title="Trades")],
    ).properties(width=700, height=180, title="Holding-period distribution")


def yearly_pnl_chart(trade_df: pd.DataFrame) -> alt.Chart:
    """Realized option trade P&L grouped by exit year."""
    if trade_df.empty or "net_pnl" not in trade_df.columns:
        return alt.Chart(pd.DataFrame({"year": [], "net_pnl": []})).mark_bar()
    data = trade_df[["exit_date", "net_pnl"]].copy()
    data["year"] = pd.to_datetime(data["exit_date"]).dt.year
    yearly = data.groupby("year", as_index=False)["net_pnl"].sum()
    yearly["outcome"] = (yearly["net_pnl"] >= 0).map({True: "profit", False: "loss"})
    return alt.Chart(yearly).mark_bar(opacity=0.85).encode(
        x=alt.X("year:O", title="Exit year"),
        y=alt.Y("net_pnl:Q", title="Realized net P&L"),
        color=alt.Color("outcome:N", title=None,
                        scale=alt.Scale(domain=["profit", "loss"],
                                        range=["forestgreen", "coral"])),
        tooltip=["year:O", alt.Tooltip("net_pnl:Q", format="$,.0f")],
    ).properties(width=700, height=200, title="Realized options P&L by year")


def premium_spend_chart(trade_df: pd.DataFrame,
                        balance: pd.DataFrame,
                        budget_annual_pct: float | None = None,
                        window_days: int = 252) -> alt.Chart:
    """Realized rolling annual premium spend vs the configured budget.

    Premium per trade is signed by the entry order — buys (BTO) are debits,
    sells (STO) are credits — and uses the per-trade contract multiplier when
    the frame carries one (``shares_per_contract`` column), falling back to
    the standard 100. The rolling one-year net sum is divided by total
    capital on each day. Drift away from the budget line is a config-bug
    detector.
    """
    empty = alt.Chart(pd.DataFrame({"date": [], "spend": []})).mark_line()
    if trade_df.empty or balance.empty:
        return empty
    multiplier = (trade_df["shares_per_contract"]
                  if "shares_per_contract" in trade_df.columns
                  else SHARES_PER_CONTRACT)
    sign = (trade_df["entry_order"].map(lambda o: -1.0 if str(o) in ("STO", "Order.STO") else 1.0)
            if "entry_order" in trade_df.columns else 1.0)
    premiums = (
        trade_df.assign(premium=trade_df["entry_price"].abs()
                        * trade_df["quantity"].abs() * multiplier * sign)
        .groupby("entry_date")["premium"].sum()
    )
    # Align on normalized dates so intraday timestamps still land on the
    # right balance row instead of being silently dropped.
    premiums.index = pd.to_datetime(premiums.index).normalize()
    premiums = premiums.groupby(level=0).sum()
    balance_days = pd.to_datetime(balance.index).normalize()
    daily = premiums.reindex(balance_days, fill_value=0.0)
    daily.index = balance.index
    rolling_spend = daily.rolling(f"{window_days}D").sum()
    spend_pct = (rolling_spend / balance["total capital"]).dropna()
    data = thin_for_chart(spend_pct).rename("spend").rename_axis("date").reset_index()

    line = alt.Chart(data).mark_line(color="orangered", opacity=0.8,
                                     strokeWidth=2.5).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("spend:Q", title="Rolling 1y premium spend", axis=alt.Axis(format="%")),
        tooltip=["date:T", alt.Tooltip("spend:Q", format=".2%")],
    )
    chart = line
    if budget_annual_pct is not None:
        rule = alt.Chart(pd.DataFrame({"budget": [budget_annual_pct]})).mark_rule(
            strokeDash=[6, 4], color="steelblue", strokeWidth=2).encode(y="budget:Q")
        chart = line + rule
    return chart.properties(width=700, height=200,
                            title="Premium spend vs budget (rolling 1 year)")


def crash_window_chart(balance: pd.DataFrame,
                       benchmark_balance: pd.DataFrame | None = None,
                       windows: dict[str, tuple[str, str]] | None = None) -> alt.VConcatChart | alt.Chart | None:
    """Day-by-day zoom panels over crash windows, indexed to 100 at window start.

    Shows the strategy (and benchmark, when given) through each event so the
    put payoff and the redeploy-into-equity mechanics are visible. Windows
    not covered by the balance index are skipped; returns ``None`` when no
    window overlaps the sample.
    """
    windows = windows or DEFAULT_CRASH_WINDOWS
    panels = []
    for label, (start, end) in windows.items():
        window = balance.loc[start:end]
        if window.empty:
            continue
        frames = [pd.DataFrame({
            "date": window.index,
            "value": window["total capital"] / window["total capital"].iloc[0] * 100,
            "series": "strategy",
        })]
        if benchmark_balance is not None and not benchmark_balance.empty:
            bench = benchmark_balance.loc[start:end]
            if not bench.empty:
                frames.append(pd.DataFrame({
                    "date": bench.index,
                    "value": bench["total capital"] / bench["total capital"].iloc[0] * 100,
                    "series": "benchmark",
                }))
        data = pd.concat(frames, ignore_index=True)
        panels.append(
            alt.Chart(data).mark_line().encode(
                x=alt.X("date:T", title=None,
                        axis=alt.Axis(format="%b %Y", tickCount=8)),
                y=alt.Y("value:Q", title="Indexed (window start = 100)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None),
                tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".1f")],
            ).properties(width=700, height=180, title=f"Crash window: {label}")
        )
    if not panels:
        return None
    return alt.vconcat(*panels) if len(panels) > 1 else panels[0]


def trade_pnl_chart(trade_df: pd.DataFrame) -> alt.Chart:
    """Per-trade net P&L ordered by exit date, on a symlog scale.

    A plain histogram hides the structure of deep-OTM put trades: many small
    losses and a few enormous winners. Symlog keeps both visible.
    """
    if trade_df.empty:
        return alt.Chart(pd.DataFrame({"exit_date": [], "net_pnl": []})).mark_bar()
    data = trade_df[["exit_date", "net_pnl", "contract"]].copy()
    data["outcome"] = (data["net_pnl"] > 0).map({True: "win", False: "loss"})
    # Explicit decade ticks: vega's automatic symlog ticks bunch up near the
    # top of the scale and render illegibly.
    peak = data["net_pnl"].abs().max()
    ticks = [0.0]
    power = 1_000.0
    while power <= peak:
        ticks = [-power] + ticks + [power]
        power *= 10
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("exit_date:T", title="Exit date"),
        y=alt.Y("net_pnl:Q", title="Net P&L per trade ($, symlog)",
                scale=alt.Scale(type="symlog"),
                axis=alt.Axis(values=ticks, format="~s")),
        color=alt.Color("outcome:N", title=None,
                        scale=alt.Scale(domain=["win", "loss"],
                                        range=["forestgreen", "coral"])),
        tooltip=["exit_date:T", "contract:N", alt.Tooltip("net_pnl:Q", format="$,.0f")],
    ).properties(width=700, height=250, title="Per-trade net P&L (symlog scale)")


def exposure_chart(balance: pd.DataFrame) -> alt.Chart:
    """Options-leg market value as a share of total capital over time."""
    if "options capital" not in balance.columns:
        return alt.Chart(pd.DataFrame({"date": [], "exposure": []})).mark_area()
    exposure = (balance["options capital"] / balance["total capital"]).dropna()
    data = thin_for_chart(exposure).rename("exposure").rename_axis("date").reset_index()
    return alt.Chart(data).mark_area(opacity=0.7, color="steelblue").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("exposure:Q", title="Options share of capital", axis=alt.Axis(format="%")),
        tooltip=["date:T", alt.Tooltip("exposure:Q", format=".2%")],
    ).properties(width=700, height=180, title="Options exposure over time")


__all__ = [
    "DEFAULT_CRASH_WINDOWS",
    "normalize_trade_log",
    "options_pnl_decomposition_chart",
    "trade_return_histogram",
    "holding_period_chart",
    "yearly_pnl_chart",
    "premium_spend_chart",
    "crash_window_chart",
    "trade_pnl_chart",
    "exposure_chart",
]
