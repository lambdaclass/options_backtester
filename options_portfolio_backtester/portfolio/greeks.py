"""Portfolio-level Greeks aggregation."""

from __future__ import annotations

from options_portfolio_backtester.core.types import Greeks
from options_portfolio_backtester.portfolio.position import OptionPosition


def aggregate_greeks(
    positions: dict[int, OptionPosition],
    leg_greeks_by_position: dict[int, dict[str, Greeks]],
) -> Greeks:
    """Compute portfolio-level Greeks by summing across all positions.

    Args:
        positions: {position_id: OptionPosition}.
        leg_greeks_by_position: {position_id: {leg_name: Greeks}}.

    Returns:
        Total portfolio Greeks.
    """
    total = Greeks()
    for pid, pos in positions.items():
        pos_greeks = leg_greeks_by_position.get(pid, {})
        total = total + pos.greeks(pos_greeks)
    return total
