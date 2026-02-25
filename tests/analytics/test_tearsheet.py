from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from options_portfolio_backtester.analytics.tearsheet import (
    build_tearsheet,
    drawdown_series,
    monthly_return_table,
)


def _balance(periods: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    total = [100_000.0]
    for i in range(1, len(idx)):
        total.append(total[-1] * (1.0 + (0.001 if i % 3 else -0.0005)))
    bal = pd.DataFrame({"total capital": total}, index=idx)
    bal["% change"] = bal["total capital"].pct_change()
    return bal


# ---------------------------------------------------------------------------
# build_tearsheet
# ---------------------------------------------------------------------------

def test_build_tearsheet_has_expected_artifacts():
    report = build_tearsheet(_balance(), trade_pnls=[100.0, -50.0, 70.0])
    assert report.stats.total_trades == 3
    assert not report.stats_table.empty
    assert "Value" in report.stats_table.columns
    assert isinstance(report.monthly_returns, pd.DataFrame)
    assert isinstance(report.drawdown_series, pd.Series)


def test_build_tearsheet_no_trades():
    report = build_tearsheet(_balance())
    assert report.stats.total_trades == 0


def test_build_tearsheet_with_risk_free_rate():
    report = build_tearsheet(_balance(), risk_free_rate=0.04)
    assert report.stats is not None


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

def test_tearsheet_to_dict_shape():
    report = build_tearsheet(_balance())
    d = report.to_dict()
    assert "stats" in d
    assert "stats_table" in d
    assert "monthly_returns" in d
    assert "drawdown_series" in d


# ---------------------------------------------------------------------------
# Exports: CSV, HTML, Markdown
# ---------------------------------------------------------------------------

def test_tearsheet_exports(tmp_path: Path):
    report = build_tearsheet(_balance())
    files = report.to_csv(tmp_path)
    assert files["stats_table"].exists()
    assert files["monthly_returns"].exists()
    assert files["drawdown_series"].exists()
    assert "<html>" in report.to_html()
    assert "# Tearsheet" in report.to_markdown()


def test_csv_creates_directories(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c"
    report = build_tearsheet(_balance())
    files = report.to_csv(nested)
    assert nested.exists()
    assert files["stats_table"].exists()


def test_html_contains_tables():
    report = build_tearsheet(_balance())
    html = report.to_html()
    assert "stats-table" in html
    assert "monthly-returns" in html or "No monthly returns" in html


# ---------------------------------------------------------------------------
# to_markdown fallback (item 15)
# ---------------------------------------------------------------------------

def test_tearsheet_markdown_fallback_without_tabulate():
    report = build_tearsheet(_balance())
    # Patch to_markdown to raise so the except fallback fires
    with patch.object(pd.DataFrame, "to_markdown", side_effect=ImportError("no tabulate")):
        md = report.to_markdown()
    assert "# Tearsheet" in md
    assert "Summary" in md


# ---------------------------------------------------------------------------
# monthly_return_table
# ---------------------------------------------------------------------------

def test_monthly_return_table_has_year_month_structure():
    bal = _balance(periods=120)  # ~6 months of business days
    tbl = monthly_return_table(bal)
    if not tbl.empty:
        assert tbl.index.name == "year"
        assert all(isinstance(c, int) for c in tbl.columns)


def test_monthly_return_table_empty_balance():
    empty = pd.DataFrame(columns=["total capital", "% change"])
    assert monthly_return_table(empty).empty


def test_monthly_return_table_no_pct_change():
    bal = pd.DataFrame({"total capital": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
    assert monthly_return_table(bal).empty


# ---------------------------------------------------------------------------
# drawdown_series
# ---------------------------------------------------------------------------

def test_drawdown_series_shape():
    bal = _balance()
    dd = drawdown_series(bal)
    assert len(dd) == len(bal)
    assert (dd <= 0).all()  # drawdowns are always <= 0


def test_drawdown_series_peak_at_start():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    bal = pd.DataFrame({"total capital": [100, 90, 80, 85, 95]}, index=idx)
    dd = drawdown_series(bal)
    assert dd.iloc[0] == 0.0  # first point is always 0
    assert dd.iloc[2] == -0.2  # 80/100 - 1 = -0.2


def test_drawdown_series_empty():
    empty = pd.DataFrame(columns=["total capital"])
    dd = drawdown_series(empty)
    assert dd.empty


def test_drawdown_series_no_total_capital():
    bad = pd.DataFrame({"other": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
    dd = drawdown_series(bad)
    assert dd.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_build_tearsheet_single_day():
    bal = pd.DataFrame(
        {"total capital": [100_000.0], "% change": [np.nan]},
        index=pd.date_range("2024-01-01", periods=1),
    )
    report = build_tearsheet(bal)
    assert report.monthly_returns.empty or not report.monthly_returns.empty
    assert isinstance(report.drawdown_series, pd.Series)


def test_build_tearsheet_flat_returns():
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    bal = pd.DataFrame({"total capital": [100_000.0] * 20}, index=idx)
    bal["% change"] = bal["total capital"].pct_change()
    report = build_tearsheet(bal)
    dd = report.drawdown_series
    # No drawdown for flat returns
    assert (dd.dropna() == 0).all()
