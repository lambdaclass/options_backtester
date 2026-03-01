"""Tests for allocation strategies."""

import pytest

from options_portfolio_backtester.convexity.allocator import (
    allocate_equal_weight,
    allocate_inverse_vol,
    pick_cheapest,
)


class TestPickCheapest:
    def test_picks_highest_ratio(self):
        scores = {"SPY": 1.5, "HYG": 3.2, "EEM": 2.0}
        assert pick_cheapest(scores) == "HYG"

    def test_single_instrument(self):
        assert pick_cheapest({"SPY": 1.0}) == "SPY"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pick_cheapest({})


class TestEqualWeight:
    def test_splits_evenly(self):
        alloc = allocate_equal_weight(["SPY", "HYG", "EEM"], 6000.0)
        assert alloc == {"SPY": 2000.0, "HYG": 2000.0, "EEM": 2000.0}

    def test_single(self):
        alloc = allocate_equal_weight(["SPY"], 5000.0)
        assert alloc == {"SPY": 5000.0}

    def test_empty(self):
        assert allocate_equal_weight([], 5000.0) == {}


class TestInverseVol:
    def test_lower_vol_gets_more(self):
        alloc = allocate_inverse_vol({"SPY": 0.20, "HYG": 0.40}, 6000.0)
        assert alloc["SPY"] > alloc["HYG"]
        assert abs(alloc["SPY"] + alloc["HYG"] - 6000.0) < 0.01

    def test_equal_vol(self):
        alloc = allocate_inverse_vol({"SPY": 0.20, "HYG": 0.20}, 4000.0)
        assert abs(alloc["SPY"] - 2000.0) < 0.01
        assert abs(alloc["HYG"] - 2000.0) < 0.01

    def test_zero_vol_falls_back(self):
        alloc = allocate_inverse_vol({"SPY": 0.0, "HYG": 0.0}, 4000.0)
        assert abs(alloc["SPY"] - 2000.0) < 0.01
