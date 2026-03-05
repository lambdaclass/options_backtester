"""Property-based tests for Schema, Field, and Filter DSL."""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from options_portfolio_backtester.data.schema import Schema, Field, Filter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

numeric_value = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
positive_numeric = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


def _make_df(n_rows, col_name="strike", values=None):
    """Build a simple numeric DataFrame for filter testing."""
    if values is None:
        rng = np.random.default_rng(42)
        values = rng.uniform(50, 500, size=n_rows)
    return pd.DataFrame({col_name: values})


# ---------------------------------------------------------------------------
# Filter properties
# ---------------------------------------------------------------------------

class TestFilterProperties:
    @given(
        st.floats(min_value=100.0, max_value=400.0, allow_nan=False),
        st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_filter_returns_subset(self, threshold, n_rows):
        """Compiled filter result is a subset of input rows."""
        assume(n_rows > 0)
        df = _make_df(n_rows)
        f = Field("strike", "strike")
        filt = f >= threshold
        mask = filt(df)
        filtered = df[mask]
        assert len(filtered) <= len(df)

    @given(st.integers(min_value=5, max_value=200))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_impossible_range_empty(self, n_rows):
        """min > max range → empty result."""
        df = _make_df(n_rows)
        f = Field("strike", "strike")
        filt = (f >= 9999) & (f <= 0)
        mask = filt(df)
        assert mask.sum() == 0

    @given(
        st.floats(min_value=100.0, max_value=400.0, allow_nan=False),
        st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_numeric_filter_bounds(self, threshold, n_rows):
        """All matched values satisfy the filter condition."""
        assume(n_rows > 0)
        df = _make_df(n_rows)
        f = Field("strike", "strike")
        filt = f >= threshold
        mask = filt(df)
        matched = df.loc[mask, "strike"]
        assert (matched >= threshold - 1e-10).all()

    @given(
        st.floats(min_value=100.0, max_value=300.0, allow_nan=False),
        st.floats(min_value=300.0, max_value=500.0, allow_nan=False),
        st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_and_is_intersection(self, lo, hi, n_rows):
        """AND of two filters = intersection of their individual results."""
        assume(lo < hi and n_rows > 0)
        df = _make_df(n_rows)
        f = Field("strike", "strike")
        f1 = f >= lo
        f2 = f <= hi
        combined = f1 & f2

        mask_1 = f1(df)
        mask_2 = f2(df)
        mask_and = combined(df)

        expected = mask_1 & mask_2
        assert (mask_and == expected).all()

    @given(
        st.floats(min_value=100.0, max_value=300.0, allow_nan=False),
        st.floats(min_value=300.0, max_value=500.0, allow_nan=False),
        st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_or_is_union(self, lo, hi, n_rows):
        """OR of two filters = union of their individual results."""
        assume(lo < hi and n_rows > 0)
        df = _make_df(n_rows)
        f = Field("strike", "strike")
        f1 = f <= lo
        f2 = f >= hi
        combined = f1 | f2

        mask_1 = f1(df)
        mask_2 = f2(df)
        mask_or = combined(df)

        expected = mask_1 | mask_2
        assert (mask_or == expected).all()
