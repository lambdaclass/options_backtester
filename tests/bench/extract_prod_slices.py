#!/usr/bin/env python3
"""Extract diverse time-period slices from raw parquet data for parity testing.

Reads raw options + underlying parquets and outputs backtester-format CSV slices
into tests/data/. Uses the same column mapping as data/fetch_data.py.

Usage:
    python tests/bench/extract_prod_slices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "tests" / "data"

# ── Slice definitions ─────────────────────────────────────────────────

SLICES = {
    "spy_crisis": {
        "options_parquet": "data/raw/release/SPY_options.parquet",
        "underlying_parquet": "data/raw/release/SPY_underlying.parquet",
        "symbol": "SPY",
        "start": "2008-06-01",
        "end": "2009-06-30",
    },
    "spy_lowvol": {
        "options_parquet": "data/raw/release/SPY_options.parquet",
        "underlying_parquet": "data/raw/release/SPY_underlying.parquet",
        "symbol": "SPY",
        "start": "2017-01-01",
        "end": "2018-01-31",
    },
    "spy_covid": {
        "options_parquet": "data/raw/release/SPY_options.parquet",
        "underlying_parquet": "data/raw/release/SPY_underlying.parquet",
        "symbol": "SPY",
        "start": "2020-01-01",
        "end": "2021-03-31",
    },
    "spy_bear": {
        "options_parquet": "data/raw/release/SPY_options.parquet",
        "underlying_parquet": "data/raw/release/SPY_underlying.parquet",
        "symbol": "SPY",
        "start": "2022-01-01",
        "end": "2023-01-31",
    },
    "iwm_2020": {
        "options_parquet": "data/raw/options-data/IWM/options.parquet",
        "underlying_parquet": "data/raw/options-dataset-hist/IWM/underlying_prices.parquet",
        "symbol": "IWM",
        "start": "2020-01-01",
        "end": "2021-01-31",
    },
    "qqq_2020": {
        "options_parquet": "data/raw/options-data/QQQ/options.parquet",
        "underlying_parquet": "data/raw/options-dataset-hist/QQQ/underlying_prices.parquet",
        "symbol": "QQQ",
        "start": "2020-01-01",
        "end": "2021-01-01",
    },
}


# ── Column mapping (matches data/fetch_data.py lines 259-278) ────────

def _convert_options(opts: pd.DataFrame, symbol: str,
                     und_prices: pd.DataFrame | None) -> pd.DataFrame:
    """Convert raw parquet options to backtester CSV format."""
    if und_prices is not None:
        opts = opts.merge(und_prices, on="date", how="left")
    else:
        opts["underlying_last"] = float("nan")

    if "last" in opts.columns:
        _last = opts["last"].fillna((opts["bid"] + opts["ask"]) / 2)
    else:
        _last = (opts["bid"] + opts["ask"]) / 2

    return pd.DataFrame({
        "underlying": symbol,
        "underlying_last": opts["underlying_last"].values,
        "optionroot": opts["contract_id"].values,
        "type": opts["type"].values,
        "expiration": pd.to_datetime(opts["expiration"]).values,
        "quotedate": opts["date"].values,
        "strike": opts["strike"].values,
        "last": _last.values,
        "bid": opts["bid"].values,
        "ask": opts["ask"].values,
        "volume": opts["volume"].values,
        "openinterest": opts["open_interest"].values,
        "impliedvol": opts["implied_volatility"].values,
        "delta": opts["delta"].values,
        "gamma": opts["gamma"].values,
        "theta": opts["theta"].values,
        "vega": opts["vega"].values,
        "optionalias": opts["contract_id"].values,
    })


def _convert_underlying(und: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert underlying parquet to Tiingo-format stocks CSV."""
    # Some datasets have None/NaN for adjusted_close, dividend_amount,
    # split_coefficient.  Fall back to close (no adjustment) and safe defaults.
    adj_close = und["adjusted_close"].fillna(und["close"])
    div_cash = und["dividend_amount"].fillna(0.0)
    split_factor = und["split_coefficient"].fillna(1.0)
    ratio = adj_close / und["close"]

    return pd.DataFrame({
        "symbol": symbol,
        "date": und["date"].values,
        "close": und["close"].values,
        "high": und["high"].values,
        "low": und["low"].values,
        "open": und["open"].values,
        "volume": und["volume"].values,
        "adjClose": adj_close.values,
        "adjHigh": (und["high"] * ratio).values,
        "adjLow": (und["low"] * ratio).values,
        "adjOpen": (und["open"] * ratio).values,
        "adjVolume": und["volume"].values,
        "divCash": div_cash.values,
        "splitFactor": split_factor.values,
    })


def extract_slice(slice_id: str, spec: dict) -> None:
    """Extract a single slice to CSV files."""
    options_path = PROJECT_ROOT / spec["options_parquet"]
    underlying_path = PROJECT_ROOT / spec["underlying_parquet"]
    symbol = spec["symbol"]
    start = pd.Timestamp(spec["start"])
    end = pd.Timestamp(spec["end"])

    if not options_path.exists():
        print(f"  SKIP {slice_id}: {options_path} not found")
        return
    if not underlying_path.exists():
        print(f"  SKIP {slice_id}: {underlying_path} not found")
        return

    # Read underlying
    print(f"  Reading underlying from {underlying_path.name}...")
    und = pd.read_parquet(underlying_path)
    und["date"] = pd.to_datetime(und["date"])
    und = und[(und["date"] >= start) & (und["date"] <= end)]
    if und.empty:
        print(f"  SKIP {slice_id}: no underlying data in [{start.date()}, {end.date()}]")
        return

    und = und.sort_values("date")
    stocks_df = _convert_underlying(und, symbol)
    und_prices = und[["date", "close"]].rename(columns={"close": "underlying_last"})

    # Read options (use filters for efficiency on large parquets)
    print(f"  Reading options from {options_path.name} "
          f"[{start.date()}, {end.date()}]...")
    opts = pd.read_parquet(options_path)
    opts["date"] = pd.to_datetime(opts["date"])
    opts = opts[(opts["date"] >= start) & (opts["date"] <= end)]
    if opts.empty:
        print(f"  SKIP {slice_id}: no options data in [{start.date()}, {end.date()}]")
        return

    options_df = _convert_options(opts, symbol, und_prices)
    options_df = options_df.sort_values(
        ["quotedate", "underlying", "expiration", "strike", "type"]
    )

    # Align dates: keep only days present in both stocks and options
    stock_dates = set(pd.to_datetime(stocks_df["date"]).dt.normalize())
    option_dates = set(pd.to_datetime(options_df["quotedate"]).dt.normalize())
    shared = stock_dates & option_dates

    stocks_df = stocks_df[
        pd.to_datetime(stocks_df["date"]).dt.normalize().isin(shared)
    ]
    options_df = options_df[
        pd.to_datetime(options_df["quotedate"]).dt.normalize().isin(shared)
    ]

    # Write CSVs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stocks_out = DATA_DIR / f"{slice_id}_stocks.csv"
    options_out = DATA_DIR / f"{slice_id}_options.csv"

    stocks_df.to_csv(stocks_out, index=False)
    options_df.to_csv(options_out, index=False)

    print(f"  {slice_id}: {len(stocks_df)} stock rows, "
          f"{len(options_df)} option rows, "
          f"{len(shared)} trading days → {stocks_out.name}, {options_out.name}")


def main():
    print("Extracting production slices for parity testing...")
    print(f"Output directory: {DATA_DIR}\n")

    for slice_id, spec in SLICES.items():
        print(f"[{slice_id}]")
        extract_slice(slice_id, spec)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
