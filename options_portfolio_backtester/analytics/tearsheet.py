"""Tearsheet report: stats tables plus an embedded-chart HTML document.

``build_tearsheet(...)`` returns a :class:`TearsheetReport`;
``report.to_file("report.html")`` writes a single document with every panel —
the pyfolio experience, one call, one file.

Chart embedding: when ``vl-convert-python`` is installed (the ``charts``
extra), charts are inlined as static SVG and the HTML is fully offline.
Otherwise charts render interactively through the vega-embed CDN scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from options_portfolio_backtester.analytics.results import returns_from_balance
from options_portfolio_backtester.analytics.stats import BacktestStats, extended_stats


@dataclass
class TearsheetReport:
    """Container for common report artifacts."""

    stats: BacktestStats
    stats_table: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown_series: pd.Series
    balance: Optional[pd.DataFrame] = None
    benchmark_balance: Optional[pd.DataFrame] = None
    trade_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    budget_annual_pct: Optional[float] = None
    kpi_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdowns: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    largest_winners: pd.DataFrame = field(default_factory=pd.DataFrame)
    largest_losers: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "stats": self.stats,
            "stats_table": self.stats_table,
            "monthly_returns": self.monthly_returns,
            "drawdown_series": self.drawdown_series,
            "kpi_table": self.kpi_table,
            "benchmark_table": self.benchmark_table,
            "drawdowns": self.drawdowns,
            "trade_summary": self.trade_summary,
            "largest_winners": self.largest_winners,
            "largest_losers": self.largest_losers,
            "yearly_pnl": self.yearly_pnl,
            "metadata": self.metadata,
        }

    def to_csv(self, directory: str | Path) -> dict[str, Path]:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / "stats_table.csv"
        monthly_path = out_dir / "monthly_returns.csv"
        drawdown_path = out_dir / "drawdown_series.csv"
        self.stats_table.to_csv(stats_path)
        self.monthly_returns.to_csv(monthly_path)
        self.drawdown_series.rename("drawdown").to_frame().to_csv(drawdown_path)
        return {
            "stats_table": stats_path,
            "monthly_returns": monthly_path,
            "drawdown_series": drawdown_path,
        }

    def to_directory(self, directory: str | Path, include_charts: bool = True) -> dict[str, Path]:
        """Write an export bundle: HTML, tables, normalized trades, and metadata."""
        written = self.to_csv(directory)
        out_dir = Path(directory)
        tables = {
            "kpi_table": self.kpi_table,
            "benchmark_table": self.benchmark_table,
            "drawdowns": self.drawdowns,
            "trade_summary": self.trade_summary,
            "largest_winners": self.largest_winners,
            "largest_losers": self.largest_losers,
            "yearly_pnl": self.yearly_pnl,
            "trades": self.trade_df,
        }
        for name, table in tables.items():
            if table.empty:
                continue
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=name in {"drawdowns", "largest_winners", "largest_losers", "trades"})
            written[name] = path
        if self.metadata:
            path = out_dir / "metadata.json"
            path.write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
            written["metadata"] = path
        written["report"] = self.to_file(out_dir / "report.html", include_charts=include_charts)
        return written

    def to_markdown(self) -> str:
        lines = ["# Tearsheet", "", "## Summary", ""]
        try:
            lines.extend(self.stats_table.to_markdown().splitlines())
        except Exception:
            lines.extend(self.stats_table.to_string().splitlines())
        lines.extend(["", "## Monthly Returns", ""])
        if self.monthly_returns.empty:
            lines.append("_No monthly returns available._")
        else:
            try:
                lines.extend(self.monthly_returns.to_markdown().splitlines())
            except Exception:
                lines.extend(self.monthly_returns.to_string().splitlines())
        return "\n".join(lines)

    # ------------------------------------------------------------- charts
    def charts(self) -> dict[str, Any]:
        """Assemble every chart panel whose inputs are available.

        Returns an ordered ``{section title: alt.Chart}`` dict. Panels whose
        inputs are missing (no balance, no trade log, no capital-split
        columns) are silently skipped so the report degrades gracefully.
        """
        from options_portfolio_backtester.analytics import charts as c
        from options_portfolio_backtester.analytics import options_charts as oc

        if self.balance is None or self.balance.empty:
            return {}
        out: dict[str, Any] = {}
        # Log scale: over a long sample, linear growth-of-$1 visually
        # exaggerates the recent years and compresses the early crashes.
        out["Equity curve"] = c.equity_curve_chart(
            self.balance, self.benchmark_balance, log_scale=True,
            drawdown_periods=top_drawdowns(self.balance))
        if not self.drawdown_series.empty:
            out["Underwater plot"] = c.underwater_chart(self.drawdown_series)
        out["Rolling Sharpe"] = c.rolling_sharpe_chart(self.balance)
        out["Rolling volatility"] = c.rolling_volatility_chart(self.balance)
        if "% change" in self.balance.columns:
            out["Return distribution"] = c.returns_histogram(
                self.balance[["% change"]].dropna())
        if "total capital" in self.balance.columns:
            out["Annual returns"] = c.annual_returns_chart(self.balance)
            out["Monthly returns heatmap"] = c.monthly_returns_heatmap(self.balance)
        if "stocks capital" in self.balance.columns:
            out["Capital allocation"] = c.weights_area_chart(self.balance)
        if "options capital" in self.balance.columns:
            out["Options exposure"] = oc.exposure_chart(self.balance)
        crash = oc.crash_window_chart(self.balance, self.benchmark_balance)
        if crash is not None:
            out["Crash windows"] = crash
        if not self.trade_df.empty:
            out["Options P&L decomposition"] = oc.options_pnl_decomposition_chart(
                self.trade_df, self.balance)
            out["Per-trade P&L"] = oc.trade_pnl_chart(self.trade_df)
            out["Trade payoff distribution"] = oc.trade_return_histogram(self.trade_df)
            out["Holding periods"] = oc.holding_period_chart(self.trade_df)
            out["Realized P&L by year"] = oc.yearly_pnl_chart(self.trade_df)
            out["Premium spend"] = oc.premium_spend_chart(
                self.trade_df, self.balance, self.budget_annual_pct)
        return out

    def to_html(self, include_charts: bool = True) -> str:
        kpis = _kpi_cards_html(self.kpi_table)
        summary = _format_table_html(self.stats_table, "stats-table")
        benchmark = _format_table_html(self.benchmark_table, "benchmark-table") \
            if not self.benchmark_table.empty else ""
        monthly = (
            _format_table_html(self.monthly_returns, "monthly-returns")
            if not self.monthly_returns.empty
            else "<p>No monthly returns available.</p>"
        )
        dd_table = self.drawdowns
        drawdowns = (
            _format_table_html(dd_table, "top-drawdowns", index=False)
            if not dd_table.empty
            else ""
        )
        trade_summary = _format_table_html(self.trade_summary, "trade-summary") \
            if not self.trade_summary.empty else ""
        yearly_pnl = _format_table_html(self.yearly_pnl, "yearly-pnl") \
            if not self.yearly_pnl.empty else ""
        winners = _format_table_html(self.largest_winners, "largest-winners", index=False) \
            if not self.largest_winners.empty else ""
        losers = _format_table_html(self.largest_losers, "largest-losers", index=False) \
            if not self.largest_losers.empty else ""

        sections = [
            "<h1>Tearsheet</h1>",
            "<nav><a href='#summary'>Summary</a><a href='#returns'>Returns</a>"
            "<a href='#risk'>Risk</a><a href='#trades'>Trades</a>"
            "<a href='#charts'>Charts</a><a href='#appendix'>Appendix</a></nav>",
            "<section id='summary'><h2>Summary</h2>", kpis, summary,
        ]
        if benchmark:
            sections += ["<h3>Benchmark comparison</h3>", benchmark]
        sections += ["</section>", "<section id='returns'><h2>Returns</h2>",
                     "<h3>Monthly returns</h3>", monthly, "</section>"]
        if drawdowns:
            sections += ["<section id='risk'><h2>Risk</h2><h3>Top drawdowns</h3>",
                         drawdowns, "</section>"]
        if self.balance is not None:
            stress = stress_events_table(self.balance, self.benchmark_balance)
            if not stress.empty:
                formatted = stress.copy()
                for col in formatted.columns:
                    if col != "event":
                        formatted[col] = formatted[col].map(
                            lambda v: f"{v:+.1%}" if pd.notna(v) else "")
                sections += ["<section><h2>Stress Events</h2>",
                             formatted.to_html(classes="stress-events", index=False),
                             "</section>"]
        if trade_summary or yearly_pnl or winners or losers:
            sections += ["<section id='trades'><h2>Trades</h2>"]
            if trade_summary:
                sections += ["<h3>Diagnostics</h3>", trade_summary]
            if yearly_pnl:
                sections += ["<h3>P&L by year</h3>", yearly_pnl]
            if winners:
                sections += ["<h3>Largest winners</h3>", winners]
            if losers:
                sections += ["<h3>Largest losers</h3>", losers]
            sections += ["</section>"]

        head_extra = ""
        if include_charts:
            charts = self.charts()
            rendered, needs_vega = _render_charts(charts)
            sections += ["<section id='charts'><h2>Charts</h2>", *rendered, "</section>"]
            if needs_vega:
                head_extra = (
                    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
                    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
                    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
                )
        if self.metadata:
            sections += [
                "<section id='appendix'><h2>Inputs and assumptions</h2>",
                f"<pre>{_html_escape(json.dumps(self.metadata, indent=2, default=str))}</pre>",
                "</section>",
            ]

        style = (
            "<style>"
            "body{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;"
            "color:#333;max-width:1180px;margin:2rem auto;padding:0 1.25rem;"
            "line-height:1.45;background:#f7f8fa}"
            "nav{position:sticky;top:0;background:#fff;border:1px solid #ddd;"
            "padding:.65rem .8rem;margin:0 0 1.25rem;display:flex;gap:1rem;"
            "flex-wrap:wrap;z-index:2}"
            "nav a{color:#1f5f99;text-decoration:none;font-size:.9rem}"
            "section{background:#fff;border:1px solid #ddd;padding:1rem 1.25rem;"
            "margin:1rem 0}"
            "h1{font-size:1.7rem;font-weight:600;border-bottom:2px solid #333;"
            "padding-bottom:.4rem;margin-bottom:1.5rem}"
            "h2{font-size:1.1rem;font-weight:600;color:#444;margin:.25rem 0 .75rem;"
            "border-bottom:1px solid #ddd;padding-bottom:.25rem}"
            "h3{font-size:.98rem;margin:1.25rem 0 .5rem}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));"
            "gap:.75rem;margin:.75rem 0 1rem}.kpi{border:1px solid #ddd;background:#fafafa;"
            "padding:.75rem}.kpi-label{font-size:.75rem;color:#666;text-transform:uppercase}"
            ".kpi-value{font-size:1.35rem;font-weight:650;margin-top:.2rem}"
            "table{border-collapse:collapse;font-size:.85rem;margin:.5rem 0;width:100%}"
            "th{background:#f5f5f5;font-weight:600}"
            "td,th{border:1px solid #e0e0e0;padding:5px 10px;text-align:right}"
            "td:first-child,th:first-child{text-align:left}"
            "tbody tr:nth-child(even){background:#fafafa}"
            "svg{max-width:100%;height:auto}pre{background:#17202a;color:#f7f8fa;"
            "padding:1rem;overflow:auto;font-size:.82rem}"
            "</style>"
        )
        return (
            "<html><head><meta charset='utf-8'><title>Tearsheet</title>"
            f"{style}{head_extra}</head><body>"
            + "".join(sections)
            + "</body></html>"
        )

    def to_file(self, path: str | Path, include_charts: bool = True) -> Path:
        """Write the HTML report to ``path`` and return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_html(include_charts=include_charts), encoding="utf-8")
        return out


def _render_charts(charts: dict[str, Any]) -> tuple[list[str], bool]:
    """Render charts to HTML fragments.

    Prefers static inline SVG via vl-convert (fully offline document); falls
    back to vega-embed divs that need the CDN scripts. Returns the fragments
    and whether the CDN scripts are required.
    """
    from options_portfolio_backtester.analytics.charts import apply_pyfolio_style

    try:
        import vl_convert  # type: ignore
        have_vl = True
    except ImportError:
        have_vl = False

    fragments: list[str] = []
    needs_vega = False
    for i, (title, chart) in enumerate(charts.items()):
        fragments.append(f"<h2>{title}</h2>")
        spec = apply_pyfolio_style(chart).to_json()
        if have_vl:
            try:
                svg = vl_convert.vegalite_to_svg(spec)
                fragments.append(svg)
                continue
            except Exception:
                pass  # fall through to vega-embed for this chart
        needs_vega = True
        div_id = f"chart-{i}"
        fragments.append(
            f'<div id="{div_id}"></div>'
            f"<script>vegaEmbed('#{div_id}', {spec});</script>"
        )
    return fragments, needs_vega


def monthly_return_table(balance: pd.DataFrame) -> pd.DataFrame:
    if balance.empty or "% change" not in balance.columns:
        return pd.DataFrame()
    rets = balance["% change"].dropna()
    if rets.empty:
        return pd.DataFrame()
    monthly = (1.0 + rets).groupby(pd.Grouper(freq="ME")).prod() - 1.0
    out = monthly.to_frame(name="return")
    idx = pd.DatetimeIndex(out.index)
    out["year"] = idx.year
    out["month"] = idx.month
    return out.pivot(index="year", columns="month", values="return").sort_index()


def drawdown_series(balance: pd.DataFrame) -> pd.Series:
    if balance.empty or "total capital" not in balance.columns:
        return pd.Series(dtype=float)
    total = balance["total capital"].dropna()
    if total.empty:
        return pd.Series(dtype=float)
    peak = total.cummax()
    return (total - peak) / peak


def stress_events_table(balance: pd.DataFrame,
                        benchmark_balance: pd.DataFrame | None = None,
                        windows: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """pyfolio's "interesting times" analysis: return and max drawdown per
    stress event window, strategy vs benchmark. Windows outside the sample
    are skipped."""
    from options_portfolio_backtester.analytics.options_charts import DEFAULT_CRASH_WINDOWS

    windows = windows or DEFAULT_CRASH_WINDOWS
    rows = []
    for label, (start, end) in windows.items():
        window = balance.loc[start:end]
        if window.empty:
            continue
        total = window["total capital"].dropna()
        row = {
            "event": label,
            "strategy return": total.iloc[-1] / total.iloc[0] - 1,
            "strategy max DD": ((total - total.cummax()) / total.cummax()).min(),
        }
        if benchmark_balance is not None and not benchmark_balance.empty:
            bench = benchmark_balance.loc[start:end]["total capital"].dropna()
            if not bench.empty:
                row["benchmark return"] = bench.iloc[-1] / bench.iloc[0] - 1
                row["benchmark max DD"] = ((bench - bench.cummax()) / bench.cummax()).min()
        rows.append(row)
    return pd.DataFrame(rows)


def top_drawdowns(balance: pd.DataFrame | None, n: int = 5) -> pd.DataFrame:
    """The ``n`` worst drawdown episodes: peak, trough, recovery, depth, duration.

    Recovery is NaT for a drawdown still open at the end of the sample.
    """
    if balance is None:
        return pd.DataFrame()
    dd = drawdown_series(balance)
    if dd.empty:
        return pd.DataFrame()

    episodes = []
    in_dd = False
    peak_date = dd.index[0]
    for date, value in dd.items():
        if not in_dd and value < 0:
            in_dd = True
            start = peak_date
        elif in_dd and value == 0:
            segment = dd.loc[start:date]
            episodes.append((start, segment.idxmin(), date, segment.min()))
            in_dd = False
        if value == 0:
            peak_date = date
    if in_dd:
        segment = dd.loc[start:]
        episodes.append((start, segment.idxmin(), pd.NaT, segment.min()))

    if not episodes:
        return pd.DataFrame()
    table = pd.DataFrame(episodes, columns=["peak", "trough", "recovery", "depth"])
    table["depth"] = (table["depth"] * 100).round(2)
    end_date = dd.index[-1]
    table["duration_days"] = (
        table["recovery"].fillna(end_date) - table["peak"]
    ).dt.days
    return table.nsmallest(n, "depth").reset_index(drop=True)


def _cagr(total: pd.Series) -> float:
    total = total.dropna()
    if len(total) < 2 or total.iloc[0] <= 0:
        return 0.0
    years = max((total.index[-1] - total.index[0]).days / 365.25, 1e-12)
    return float((total.iloc[-1] / total.iloc[0]) ** (1.0 / years) - 1.0)


def _kpi_table(stats: BacktestStats, balance: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    total = balance["total capital"].dropna() if "total capital" in balance.columns else pd.Series(dtype=float)
    total_return = float(total.iloc[-1] / total.iloc[0] - 1.0) if len(total) > 1 and total.iloc[0] else 0.0
    trades = int(len(trade_df))
    win_rate = float((trade_df["net_pnl"] > 0).mean()) if trades and "net_pnl" in trade_df.columns else 0.0
    rows = [
        ("Total return", total_return, "percent"),
        ("CAGR", _cagr(total), "percent"),
        ("Sharpe", stats.sharpe_ratio, "number"),
        ("Max drawdown", -abs(stats.max_drawdown), "percent"),
        ("Volatility", stats.volatility, "percent"),
        ("Trades", trades, "integer"),
        ("Win rate", win_rate, "percent"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "format"]).set_index("metric")


def benchmark_comparison(balance: pd.DataFrame,
                         benchmark_balance: pd.DataFrame | None) -> pd.DataFrame:
    """First-class strategy-vs-benchmark metrics for the report summary."""
    if benchmark_balance is None or benchmark_balance.empty:
        return pd.DataFrame()
    strat = returns_from_balance(balance, name="strategy")
    bench = returns_from_balance(benchmark_balance, name="benchmark")
    aligned = pd.concat([strat, bench], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.DataFrame()
    strat, bench = aligned["strategy"], aligned["benchmark"]
    excess = strat - bench
    var = float(cast(float, bench.var()))
    beta = float(cast(float, strat.cov(bench))) / var if var > 0 else 0.0
    tracking_error = float(excess.std() * np.sqrt(252))
    downside = bench < 0
    upside = bench > 0
    rows = [
        ("Strategy CAGR", _cagr(balance["total capital"]), "percent"),
        ("Benchmark CAGR", _cagr(benchmark_balance["total capital"]), "percent"),
        ("Excess CAGR", _cagr(balance["total capital"]) - _cagr(benchmark_balance["total capital"]), "percent"),
        ("Correlation", float(strat.corr(bench)) if len(aligned) > 1 else 0.0, "number"),
        ("Beta", beta, "number"),
        ("Tracking error", tracking_error, "percent"),
        ("Information ratio", float(excess.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0, "number"),
        (
            "Upside capture",
            float(strat[upside].mean() / bench[upside].mean())
            if upside.any() and bench[upside].mean() else 0.0,
            "percent",
        ),
        (
            "Downside capture",
            float(strat[downside].mean() / bench[downside].mean())
            if downside.any() and bench[downside].mean() else 0.0,
            "percent",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "format"]).set_index("metric")


def trade_diagnostics(trade_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summary, largest winners/losers, and yearly realized P&L from normalized trades."""
    if trade_df.empty or "net_pnl" not in trade_df.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    df = trade_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["holding_days"] = (df["exit_date"] - df["entry_date"]).dt.days
    wins = df["net_pnl"] > 0
    summary = pd.DataFrame([
        ("Trades", len(df), "integer"),
        ("Win rate", float(wins.mean()), "percent"),
        ("Total net P&L", float(df["net_pnl"].sum()), "number"),
        ("Average win", float(df.loc[wins, "net_pnl"].mean()) if wins.any() else 0.0, "number"),
        ("Average loss", float(df.loc[~wins, "net_pnl"].mean()) if (~wins).any() else 0.0, "number"),
        ("Median return", float(df["return_pct"].median()) if "return_pct" in df.columns else 0.0, "percent"),
        ("Median holding days", float(df["holding_days"].median()), "number"),
    ], columns=["metric", "value", "format"]).set_index("metric")
    cols = [c for c in ("contract", "underlying", "type", "entry_date", "exit_date",
                        "net_pnl", "return_pct", "holding_days") if c in df.columns]
    winners = df.nlargest(min(10, len(df)), "net_pnl")[cols].reset_index(drop=True)
    losers = df.nsmallest(min(10, len(df)), "net_pnl")[cols].reset_index(drop=True)
    yearly = df.groupby(df["exit_date"].dt.year)["net_pnl"].agg(["sum", "count", "mean"])
    yearly.index.name = "year"
    return summary, winners, losers, yearly


def _html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_value(value: Any, kind: str | None = None) -> str:
    if pd.isna(value):
        return ""
    if kind == "integer":
        return f"{int(value):,}"
    if kind == "percent":
        return f"{float(value):.2%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _format_table_html(df: pd.DataFrame, classes: str, index: bool = True) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for col in display.columns:
        if col == "format":
            continue
        if col == "value" and "format" in display.columns:
            display[col] = [
                _fmt_value(v, fmt) for v, fmt in zip(display[col], display["format"])
            ]
        elif any(key in str(col).lower() for key in ("return", "drawdown", "rate", "capture", "volatility", "error")):
            display[col] = display[col].map(lambda v: _fmt_value(v, "percent"))
        elif pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(_fmt_value)
    if "format" in display.columns:
        display = display.drop(columns=["format"])
    return display.to_html(classes=classes, index=index, escape=False)


def _kpi_cards_html(kpis: pd.DataFrame) -> str:
    if kpis.empty:
        return ""
    cards = []
    for label, row in kpis.iterrows():
        cards.append(
            "<div class='kpi'>"
            f"<div class='kpi-label'>{_html_escape(str(label))}</div>"
            f"<div class='kpi-value'>{_fmt_value(row['value'], row.get('format'))}</div>"
            "</div>"
        )
    return "<div class='kpi-grid'>" + "".join(cards) + "</div>"


def build_tearsheet(
    balance: pd.DataFrame,
    trade_pnls=None,
    risk_free_rate: float = 0.0,
    *,
    benchmark_balance: pd.DataFrame | None = None,
    trade_log=None,
    budget_annual_pct: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> TearsheetReport:
    """Build a :class:`TearsheetReport` from a backtest's outputs.

    ``trade_log`` accepts a :class:`TradeLog`, a flat per-trade DataFrame, or
    the legacy MultiIndex frame from ``engine.run()``. When given, it powers
    the per-trade and premium-spend panels, and supplies ``trade_pnls`` for
    the stats block unless those were passed explicitly.
    """
    from options_portfolio_backtester.analytics.options_charts import normalize_trade_log

    trade_df = normalize_trade_log(trade_log)
    if trade_pnls is None and not trade_df.empty:
        trade_pnls = trade_df["net_pnl"].to_numpy()
    trade_arr = None if trade_pnls is None else np.asarray(trade_pnls, dtype=float)
    stats = BacktestStats.from_balance(balance, trade_pnls=trade_arr, risk_free_rate=risk_free_rate)
    table = stats.to_dataframe()

    # pyfolio-parity extras (stability, omega, VaR, alpha/beta) plus a
    # side-by-side benchmark column when a benchmark is supplied.
    rets = returns_from_balance(balance)
    bench_rets = (returns_from_balance(benchmark_balance, name="benchmark")
                  if benchmark_balance is not None else None)
    for label, value in extended_stats(rets, bench_rets).items():
        table.loc[label, "Value"] = value
    if benchmark_balance is not None and bench_rets is not None and not benchmark_balance.empty:
        bench_table = BacktestStats.from_balance(benchmark_balance) \
            .to_dataframe().rename(columns={"Value": "Benchmark"})
        for label, value in extended_stats(bench_rets).items():
            bench_table.loc[label, "Benchmark"] = value
        table = table.join(bench_table, how="left")

    monthly = monthly_return_table(balance)
    dd = drawdown_series(balance)
    drawdowns = top_drawdowns(balance)
    trade_summary, winners, losers, yearly_pnl = trade_diagnostics(trade_df)
    return TearsheetReport(
        stats=stats,
        stats_table=table,
        monthly_returns=monthly,
        drawdown_series=dd,
        balance=balance,
        benchmark_balance=benchmark_balance,
        trade_df=trade_df,
        budget_annual_pct=budget_annual_pct,
        kpi_table=_kpi_table(stats, balance, trade_df),
        benchmark_table=benchmark_comparison(balance, benchmark_balance),
        drawdowns=drawdowns,
        trade_summary=trade_summary,
        largest_winners=winners,
        largest_losers=losers,
        yearly_pnl=yearly_pnl,
        metadata=metadata or {},
    )
