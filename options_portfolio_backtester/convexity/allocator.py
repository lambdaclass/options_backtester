"""Allocation strategies: pick which instrument(s) to hedge."""

from __future__ import annotations


def pick_cheapest(scores: dict[str, float]) -> str:
    """Pick the instrument with the highest convexity ratio."""
    if not scores:
        raise ValueError("No scores to pick from")
    return max(scores, key=scores.get)


def allocate_equal_weight(symbols: list[str], budget: float) -> dict[str, float]:
    """Split budget equally across all instruments."""
    if not symbols:
        return {}
    per_symbol = budget / len(symbols)
    return {s: per_symbol for s in symbols}


def allocate_inverse_vol(vol_map: dict[str, float], budget: float) -> dict[str, float]:
    """Allocate more to lower-volatility instruments.

    Weight is proportional to 1/vol, normalized to sum to budget.
    """
    if not vol_map:
        return {}

    inv_vols = {}
    for sym, vol in vol_map.items():
        if vol > 0:
            inv_vols[sym] = 1.0 / vol

    if not inv_vols:
        return allocate_equal_weight(list(vol_map.keys()), budget)

    total_inv_vol = sum(inv_vols.values())
    return {sym: (iv / total_inv_vol) * budget for sym, iv in inv_vols.items()}
