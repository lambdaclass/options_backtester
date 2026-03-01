"""Scoring: compute convexity ratios via Rust backend."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import BacktestConfig

log = logging.getLogger(__name__)


def _to_ns(series: pd.Series) -> np.ndarray:
    """Convert a datetime Series to int64 nanosecond timestamps."""
    return series.values.astype("datetime64[ns]").view("int64").astype(np.int64)


def compute_convexity_scores(
    options_data,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Compute daily convexity ratio scores for an instrument.

    Takes an HistoricalOptionsData object from options_backtester and
    returns a DataFrame indexed by date with convexity_ratio and supporting fields.
    """
    from options_portfolio_backtester._ob_rust import compute_daily_scores

    df = options_data._data
    puts = df[df["type"] == "put"].sort_values("quotedate")

    if puts.empty:
        return pd.DataFrame()

    result = compute_daily_scores(
        dates_ns=_to_ns(puts["quotedate"]),
        strikes=puts["strike"].values.astype(np.float64),
        bids=puts["bid"].values.astype(np.float64),
        asks=puts["ask"].values.astype(np.float64),
        deltas=puts["delta"].values.astype(np.float64),
        underlying_prices=puts["underlying_last"].values.astype(np.float64),
        dtes=puts["dte"].values.astype(np.int32),
        implied_vols=puts["impliedvol"].values.astype(np.float64),
        target_delta=config.target_delta,
        dte_min=config.dte_min,
        dte_max=config.dte_max,
        tail_drop=config.tail_drop,
    )

    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(result["dates_ns"], unit="ns"),
            "convexity_ratio": result["convexity_ratios"],
            "strike": result["strikes"],
            "ask": result["asks"],
            "bid": result["bids"],
            "delta": result["deltas"],
            "underlying_price": result["underlying_prices"],
            "implied_vol": result["implied_vols"],
            "dte": result["dtes"],
            "annual_cost": result["annual_costs"],
            "tail_payoff": result["tail_payoffs"],
        }
    ).set_index("date")

    log.info("Computed %d daily scores (%.1f years)", len(scores), len(scores) / 252)

    return scores
