"""Tests for intrinsic value fallback when options expire/go missing."""

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.engine.engine import _intrinsic_value
from options_portfolio_backtester.core.types import OptionType


# ---------------------------------------------------------------------------
# Unit tests for _intrinsic_value helper
# ---------------------------------------------------------------------------


class TestIntrinsicValue:
    def test_put_itm(self):
        """Put with strike > spot → intrinsic = strike - spot."""
        assert _intrinsic_value("put", 400.0, 380.0) == pytest.approx(20.0)

    def test_put_otm(self):
        """Put with strike < spot → intrinsic = 0."""
        assert _intrinsic_value("put", 400.0, 420.0) == pytest.approx(0.0)

    def test_call_itm(self):
        """Call with spot > strike → intrinsic = spot - strike."""
        assert _intrinsic_value("call", 400.0, 420.0) == pytest.approx(20.0)

    def test_call_otm(self):
        """Call with spot < strike → intrinsic = 0."""
        assert _intrinsic_value("call", 400.0, 380.0) == pytest.approx(0.0)

    def test_put_atm(self):
        """ATM put → intrinsic = 0."""
        assert _intrinsic_value("put", 400.0, 400.0) == pytest.approx(0.0)

    def test_call_atm(self):
        """ATM call → intrinsic = 0."""
        assert _intrinsic_value("call", 400.0, 400.0) == pytest.approx(0.0)

    def test_deep_itm_put(self):
        """Deep ITM put — large intrinsic."""
        assert _intrinsic_value("put", 500.0, 300.0) == pytest.approx(200.0)

    def test_uses_option_type_enum_values(self):
        """Works with OptionType enum .value strings."""
        assert _intrinsic_value(OptionType.PUT.value, 400.0, 380.0) == pytest.approx(20.0)
        assert _intrinsic_value(OptionType.CALL.value, 400.0, 420.0) == pytest.approx(20.0)
