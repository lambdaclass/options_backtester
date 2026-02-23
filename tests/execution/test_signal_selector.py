"""Tests for signal selectors."""

import pandas as pd

from options_backtester.execution.signal_selector import (
    FirstMatch, NearestDelta, MaxOpenInterest,
)


def _make_candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "contract": ["A", "B", "C"],
        "delta": [-0.10, -0.30, -0.50],
        "openinterest": [100, 500, 200],
        "ask": [1.0, 2.0, 3.0],
    })


class TestFirstMatch:
    def test_picks_first(self):
        s = FirstMatch()
        result = s.select(_make_candidates())
        assert result["contract"] == "A"


class TestNearestDelta:
    def test_nearest_to_target(self):
        s = NearestDelta(target_delta=-0.30)
        result = s.select(_make_candidates())
        assert result["contract"] == "B"

    def test_nearest_to_different_target(self):
        s = NearestDelta(target_delta=-0.45)
        result = s.select(_make_candidates())
        assert result["contract"] == "C"

    def test_fallback_without_column(self):
        s = NearestDelta(target_delta=-0.30)
        df = _make_candidates().drop(columns=["delta"])
        result = s.select(df)
        assert result["contract"] == "A"  # falls back to first


class TestMaxOpenInterest:
    def test_picks_max_oi(self):
        s = MaxOpenInterest()
        result = s.select(_make_candidates())
        assert result["contract"] == "B"  # OI=500

    def test_fallback_without_column(self):
        s = MaxOpenInterest()
        df = _make_candidates().drop(columns=["openinterest"])
        result = s.select(df)
        assert result["contract"] == "A"
