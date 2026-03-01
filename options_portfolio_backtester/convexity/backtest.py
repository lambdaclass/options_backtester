"""Backtest: run the monthly rebalance loop via Rust backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig

log = logging.getLogger(__name__)


def _to_ns(series: pd.Series) -> np.ndarray:
    """Convert a datetime Series to int64 nanosecond timestamps."""
    return series.values.astype("datetime64[ns]").view("int64").astype(np.int64)


@dataclass
class BacktestResult:
    """Results from a single-instrument backtest."""

    records: pd.DataFrame  # monthly rebalance records
    daily_balance: pd.DataFrame  # daily portfolio values
    config: BacktestConfig


def run_backtest(
    options_data,
    stocks_data,
    config: BacktestConfig,
) -> BacktestResult:
    """Run the full backtest: monthly put overlay on equity portfolio.

    Takes HistoricalOptionsData and TiingoData from options_backtester.
    """
    from options_portfolio_backtester._ob_rust import run_convexity_backtest

    opt_df = options_data._data
    puts = opt_df[opt_df["type"] == "put"].sort_values("quotedate")
    stk_df = stocks_data._data.sort_values("date")

    if puts.empty or stk_df.empty:
        empty_records = pd.DataFrame()
        empty_daily = pd.DataFrame()
        return BacktestResult(records=empty_records, daily_balance=empty_daily, config=config)

    result = run_convexity_backtest(
        put_dates_ns=_to_ns(puts["quotedate"]),
        put_expirations_ns=_to_ns(puts["expiration"]),
        put_strikes=puts["strike"].values.astype(np.float64),
        put_bids=puts["bid"].values.astype(np.float64),
        put_asks=puts["ask"].values.astype(np.float64),
        put_deltas=puts["delta"].values.astype(np.float64),
        put_underlying=puts["underlying_last"].values.astype(np.float64),
        put_dtes=puts["dte"].values.astype(np.int32),
        put_ivs=puts["impliedvol"].values.astype(np.float64),
        stock_dates_ns=_to_ns(stk_df["date"]),
        stock_prices=stk_df["adjClose"].values.astype(np.float64),
        initial_capital=config.initial_capital,
        budget_pct=config.budget_pct,
        target_delta=config.target_delta,
        dte_min=config.dte_min,
        dte_max=config.dte_max,
        tail_drop=config.tail_drop,
    )

    # Build monthly records DataFrame
    rec = result["records"]
    records = pd.DataFrame(
        {
            "date": pd.to_datetime(rec["dates_ns"], unit="ns"),
            "shares": rec["shares"],
            "stock_price": rec["stock_prices"],
            "equity_value": rec["equity_values"],
            "put_cost": rec["put_costs"],
            "put_exit_value": rec["put_exit_values"],
            "put_pnl": rec["put_pnls"],
            "portfolio_value": rec["portfolio_values"],
            "convexity_ratio": rec["convexity_ratios"],
            "strike": rec["strikes"],
            "contracts": rec["contracts"],
        }
    ).set_index("date")

    # Build daily balance DataFrame
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(result["daily_dates_ns"], unit="ns"),
            "balance": result["daily_balances"],
        }
    ).set_index("date")
    daily["pct_change"] = daily["balance"].pct_change()

    log.info(
        "Backtest: %d months, final value $%.0f (started $%.0f)",
        len(records),
        daily["balance"].iloc[-1] if len(daily) > 0 else 0,
        config.initial_capital,
    )

    return BacktestResult(records=records, daily_balance=daily, config=config)


def run_unhedged(stocks_data, config: BacktestConfig) -> pd.DataFrame:
    """Run unhedged equity-only benchmark. Returns daily balance DataFrame."""
    stk_df = stocks_data._data.sort_values("date")
    if stk_df.empty:
        return pd.DataFrame()

    prices = stk_df["adjClose"].values.astype(np.float64)
    dates = stk_df["date"]

    initial_shares = config.initial_capital / prices[0]
    daily_balance = initial_shares * prices

    df = pd.DataFrame({"date": dates, "balance": daily_balance}).set_index("date")
    df["pct_change"] = df["balance"].pct_change()
    return df
