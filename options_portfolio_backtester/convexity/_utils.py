"""Shared utilities for the convexity module."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_ns(series: pd.Series) -> np.ndarray:
    """Convert a datetime Series to int64 nanosecond timestamps."""
    return series.values.astype("datetime64[ns]").view("int64").astype(np.int64)
