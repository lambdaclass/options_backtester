"""Tests for chart functions (weights_chart + Altair charts)."""

from __future__ import annotations

import altair as alt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for testing

from options_portfolio_backtester.analytics.charts import (
    monthly_returns_heatmap,
    returns_chart,
    returns_histogram,
    weights_chart,
)


def _make_balance() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame({
        "total capital": np.linspace(10000, 11000, 10),
        "cash": np.linspace(2000, 1000, 10),
        "stocks capital": np.linspace(8000, 10000, 10),
        "SPY qty": [50.0] * 10,
        "TLT qty": [100.0] * 10,
    }, index=dates)


def test_weights_chart_returns_fig_ax():
    balance = _make_balance()
    fig, ax = weights_chart(balance)
    assert fig is not None
    assert ax is not None
    assert ax.get_ylabel() == "Weight"


def test_weights_chart_no_positions():
    dates = pd.bdate_range("2024-01-02", periods=5)
    balance = pd.DataFrame({
        "total capital": [10000.0] * 5,
        "cash": [10000.0] * 5,
    }, index=dates)
    fig, ax = weights_chart(balance)
    assert fig is not None
    assert "no positions" in ax.get_title().lower()


def test_weights_chart_single_symbol():
    dates = pd.bdate_range("2024-01-02", periods=5)
    balance = pd.DataFrame({
        "total capital": [10000.0] * 5,
        "cash": [2000.0] * 5,
        "SPY qty": [80.0] * 5,
    }, index=dates)
    fig, ax = weights_chart(balance)
    assert fig is not None


# ── Altair chart tests (moved from backtester/test/statistics/test_charts.py) ──


def _make_balance_report(days=90):
    """Create a minimal balance-like DataFrame for Altair chart tests."""
    dates = pd.bdate_range('2020-01-01', periods=days, freq='B')
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, size=days)
    capital = 1_000_000 * np.cumprod(1 + returns)

    report = pd.DataFrame({
        'total capital': capital,
        '% change': returns,
        'accumulated return': np.cumprod(1 + returns),
    }, index=dates)
    return report


def test_returns_chart_returns_vconcat():
    """returns_chart should return a VConcatChart (layered + brush)."""
    report = _make_balance_report()
    chart = returns_chart(report)
    assert isinstance(chart, alt.VConcatChart)


def test_returns_chart_has_two_panels():
    """VConcatChart should have exactly 2 panels (main + brush)."""
    report = _make_balance_report()
    chart = returns_chart(report)
    assert len(chart.vconcat) == 2


def test_returns_chart_serializes_to_dict():
    """Chart should serialize to a valid Vega-Lite spec dict."""
    report = _make_balance_report()
    chart = returns_chart(report)
    spec = chart.to_dict()
    assert 'vconcat' in spec
    assert isinstance(spec['vconcat'], list)


def test_returns_histogram_returns_chart():
    """returns_histogram returns bars + fitted-normal overlay (LayerChart)."""
    report = _make_balance_report()
    chart = returns_histogram(report)
    assert isinstance(chart, alt.LayerChart)


def test_returns_histogram_serializes():
    """Histogram serializes with symlog bar counts and a normal overlay."""
    report = _make_balance_report()
    chart = returns_histogram(report)
    spec = chart.to_dict()
    marks = [l['mark'] if isinstance(l['mark'], str) else l['mark']['type']
             for l in spec['layer']]
    assert 'bar' in marks and 'line' in marks
    bar = next(l for l in spec['layer']
               if (l['mark'] if isinstance(l['mark'], str) else l['mark']['type']) == 'bar')
    assert bar['encoding']['y']['scale']['type'] == 'symlog'


def test_monthly_returns_heatmap_returns_chart():
    """monthly_returns_heatmap returns annotated rects (LayerChart)."""
    report = _make_balance_report(days=250)
    chart = monthly_returns_heatmap(report)
    assert isinstance(chart, alt.LayerChart)


def test_monthly_returns_heatmap_serializes():
    """Heatmap serializes with a rect layer, a text annotation layer, and
    percentage tooltips on a zero-centered diverging scale (pyfolio style)."""
    report = _make_balance_report(days=250)
    chart = monthly_returns_heatmap(report)
    spec = chart.to_dict()
    marks = [layer['mark'] if isinstance(layer['mark'], str)
             else layer['mark']['type'] for layer in spec['layer']]
    assert 'rect' in marks and 'text' in marks
    rect = next(l for l in spec['layer']
                if (l['mark'] if isinstance(l['mark'], str) else l['mark']['type']) == 'rect')
    assert rect['encoding']['tooltip']['format'] == '.2%'
    assert rect['encoding']['color']['scale']['domainMid'] == 0


def test_returns_chart_has_interval_and_point_params():
    """Verify the spec has both interval and point selection params (Altair 5 API)."""
    report = _make_balance_report()
    chart = returns_chart(report)
    spec = chart.to_dict()

    # In Altair 5, params are hoisted to the top-level spec
    params = spec.get('params', [])
    param_types = {p.get('select', {}).get('type') for p in params}
    assert 'interval' in param_types, "Expected an 'interval' selection param"
    assert 'point' in param_types, "Expected a 'point' selection param"


def test_returns_chart_data_included():
    """Verify chart spec includes data."""
    report = _make_balance_report()
    chart = returns_chart(report)
    spec = chart.to_dict()
    assert 'data' in spec or 'datasets' in spec
