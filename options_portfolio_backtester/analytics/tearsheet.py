"""Simple tearsheet-style report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from options_portfolio_backtester.analytics.stats import BacktestStats


@dataclass
class TearsheetReport:
    """Container for common report artifacts."""

    stats: BacktestStats
    stats_table: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown_series: pd.Series

    def to_dict(self) -> dict[str, object]:
        return {
            "stats": self.stats,
            "stats_table": self.stats_table,
            "monthly_returns": self.monthly_returns,
            "drawdown_series": self.drawdown_series,
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

    def to_html(self) -> str:
        summary = self.stats_table.to_html(classes="stats-table")
        monthly = (
            self.monthly_returns.to_html(classes="monthly-returns")
            if not self.monthly_returns.empty
            else "<p>No monthly returns available.</p>"
        )
        return (
            "<html><head><meta charset='utf-8'><title>Tearsheet</title></head><body>"
            "<h1>Tearsheet</h1>"
            "<h2>Summary</h2>"
            f"{summary}"
            "<h2>Monthly Returns</h2>"
            f"{monthly}"
            "</body></html>"
        )


def monthly_return_table(balance: pd.DataFrame) -> pd.DataFrame:
    if balance.empty or "% change" not in balance.columns:
        return pd.DataFrame()
    rets = balance["% change"].dropna()
    if rets.empty:
        return pd.DataFrame()
    monthly = (1.0 + rets).groupby(pd.Grouper(freq="ME")).prod() - 1.0
    out = monthly.to_frame(name="return")
    out["year"] = out.index.year
    out["month"] = out.index.month
    return out.pivot(index="year", columns="month", values="return").sort_index()


def drawdown_series(balance: pd.DataFrame) -> pd.Series:
    if balance.empty or "total capital" not in balance.columns:
        return pd.Series(dtype=float)
    total = balance["total capital"].dropna()
    if total.empty:
        return pd.Series(dtype=float)
    peak = total.cummax()
    return (total - peak) / peak


def build_tearsheet(
    balance: pd.DataFrame,
    trade_pnls=None,
    risk_free_rate: float = 0.0,
) -> TearsheetReport:
    trade_arr = None if trade_pnls is None else np.asarray(trade_pnls, dtype=float)
    stats = BacktestStats.from_balance(balance, trade_pnls=trade_arr, risk_free_rate=risk_free_rate)
    table = stats.to_dataframe()
    monthly = monthly_return_table(balance)
    dd = drawdown_series(balance)
    return TearsheetReport(
        stats=stats,
        stats_table=table,
        monthly_returns=monthly,
        drawdown_series=dd,
    )
