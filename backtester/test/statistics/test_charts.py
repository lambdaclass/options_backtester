"""Tests for statistics.charts — Altair 5 API compatibility."""
import pandas as pd
import numpy as np
import altair as alt

from backtester.statistics.charts import returns_chart, returns_histogram, monthly_returns_heatmap


def make_balance_report(days=90):
    """Create a minimal balance-like DataFrame for chart tests."""
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
    report = make_balance_report()
    chart = returns_chart(report)
    assert isinstance(chart, alt.VConcatChart)


def test_returns_chart_has_two_panels():
    """VConcatChart should have exactly 2 panels (main + brush)."""
    report = make_balance_report()
    chart = returns_chart(report)
    assert len(chart.vconcat) == 2


def test_returns_chart_serializes_to_dict():
    """Chart should serialize to a valid Vega-Lite spec dict."""
    report = make_balance_report()
    chart = returns_chart(report)
    spec = chart.to_dict()
    assert 'vconcat' in spec
    assert isinstance(spec['vconcat'], list)


def test_returns_histogram_returns_chart():
    """returns_histogram should return a Chart with bar mark."""
    report = make_balance_report()
    chart = returns_histogram(report)
    assert isinstance(chart, alt.Chart)


def test_returns_histogram_serializes():
    """Histogram should serialize without errors."""
    report = make_balance_report()
    chart = returns_histogram(report)
    spec = chart.to_dict()
    assert spec['mark']['type'] == 'bar'


def test_monthly_returns_heatmap_returns_chart():
    """monthly_returns_heatmap should return a Chart with rect mark."""
    report = make_balance_report(days=250)
    chart = monthly_returns_heatmap(report)
    assert isinstance(chart, alt.Chart)


def test_monthly_returns_heatmap_serializes():
    """Heatmap should serialize without errors."""
    report = make_balance_report(days=250)
    chart = monthly_returns_heatmap(report)
    spec = chart.to_dict()
    assert spec['mark'] == 'rect' or spec['mark']['type'] == 'rect'


def test_returns_chart_has_interval_and_point_params():
    """Verify the spec has both interval and point selection params (Altair 5 API)."""
    report = make_balance_report()
    chart = returns_chart(report)
    spec = chart.to_dict()

    # In Altair 5, params are hoisted to the top-level spec
    params = spec.get('params', [])
    param_types = {p.get('select', {}).get('type') for p in params}
    assert 'interval' in param_types, "Expected an 'interval' selection param"
    assert 'point' in param_types, "Expected a 'point' selection param"


def test_returns_chart_data_included():
    """Verify chart spec includes data."""
    report = make_balance_report()
    chart = returns_chart(report)
    spec = chart.to_dict()
    assert 'data' in spec or 'datasets' in spec
