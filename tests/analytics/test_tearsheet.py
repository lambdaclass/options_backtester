from __future__ import annotations

import pandas as pd
from pathlib import Path

from options_backtester.analytics.tearsheet import build_tearsheet


def _balance() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    total = [100_000.0]
    for i in range(1, len(idx)):
        total.append(total[-1] * (1.0 + (0.001 if i % 3 else -0.0005)))
    bal = pd.DataFrame({"total capital": total}, index=idx)
    bal["% change"] = bal["total capital"].pct_change()
    return bal


def test_build_tearsheet_has_expected_artifacts():
    report = build_tearsheet(_balance(), trade_pnls=[100.0, -50.0, 70.0])
    assert report.stats.total_trades == 3
    assert not report.stats_table.empty
    assert "Value" in report.stats_table.columns
    assert isinstance(report.monthly_returns, pd.DataFrame)
    assert isinstance(report.drawdown_series, pd.Series)


def test_tearsheet_to_dict_shape():
    report = build_tearsheet(_balance())
    d = report.to_dict()
    assert "stats" in d
    assert "stats_table" in d
    assert "monthly_returns" in d
    assert "drawdown_series" in d


def test_tearsheet_exports(tmp_path: Path):
    report = build_tearsheet(_balance())
    files = report.to_csv(tmp_path)
    assert files["stats_table"].exists()
    assert files["monthly_returns"].exists()
    assert files["drawdown_series"].exists()
    assert "<html>" in report.to_html()
    assert "# Tearsheet" in report.to_markdown()
