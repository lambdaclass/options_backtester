"""Signal selectors — choose which contract to trade from a set of candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SignalSelector(ABC):
    """Picks one entry signal from a DataFrame of candidates."""

    @property
    def column_requirements(self) -> list[str]:
        """Extra columns needed from raw options data beyond standard signal fields."""
        return []

    @abstractmethod
    def select(self, candidates: pd.DataFrame) -> pd.Series:
        """Return the single row (as Series) to execute from candidates.

        Args:
            candidates: DataFrame of entry signals, pre-sorted if entry_sort was set.

        Returns:
            A single row (pd.Series) from candidates.
        """
        ...


class FirstMatch(SignalSelector):
    """Pick the first row — matches original iloc[0] behavior."""

    def select(self, candidates: pd.DataFrame) -> pd.Series:
        return candidates.iloc[0]

    def to_rust_config(self) -> dict:
        return {"type": "FirstMatch"}


class NearestDelta(SignalSelector):
    """Pick the contract whose delta is closest to `target_delta`.

    Requires a 'delta' column in candidates.
    """

    def __init__(self, target_delta: float = -0.30, delta_column: str = "delta") -> None:
        self.target_delta = target_delta
        self.delta_column = delta_column

    @property
    def column_requirements(self) -> list[str]:
        return [self.delta_column]

    def select(self, candidates: pd.DataFrame) -> pd.Series:
        if self.delta_column not in candidates.columns:
            return candidates.iloc[0]
        diffs = (candidates[self.delta_column] - self.target_delta).abs()
        return candidates.loc[diffs.idxmin()]

    def to_rust_config(self) -> dict:
        return {"type": "NearestDelta", "target": self.target_delta, "column": self.delta_column}


class MaxOpenInterest(SignalSelector):
    """Pick the contract with the highest open interest (proxy for liquidity).

    Requires an 'openinterest' or 'open_interest' column.
    """

    def __init__(self, oi_column: str = "openinterest") -> None:
        self.oi_column = oi_column

    @property
    def column_requirements(self) -> list[str]:
        return [self.oi_column]

    def select(self, candidates: pd.DataFrame) -> pd.Series:
        if self.oi_column not in candidates.columns:
            return candidates.iloc[0]
        return candidates.loc[candidates[self.oi_column].idxmax()]

    def to_rust_config(self) -> dict:
        return {"type": "MaxOpenInterest", "column": self.oi_column}
