#!/usr/bin/env python3
"""Convert OptionsDX wide-format CSV to backtester long-format CSV.

OptionsDX provides one row per strike/date/expiry with both call and put data
in wide format. The backtester expects one row per contract in long format.

Usage:
    python data/convert_optionsdx.py data/raw/spx_eod.csv --output data/processed/spx_options.csv
"""

import argparse
import sys

import pandas as pd


# OptionsDX columns we need (they have trailing spaces, stripped on read)
CALL_COLS = {
    "C_BID": "bid",
    "C_ASK": "ask",
    "C_LAST": "last",
    "C_VOLUME": "volume",
    "C_IV": "impliedvol",
    "C_DELTA": "delta",
    "C_GAMMA": "gamma",
    "C_THETA": "theta",
    "C_VEGA": "vega",
}

PUT_COLS = {
    "P_BID": "bid",
    "P_ASK": "ask",
    "P_LAST": "last",
    "P_VOLUME": "volume",
    "P_IV": "impliedvol",
    "P_DELTA": "delta",
    "P_GAMMA": "gamma",
    "P_THETA": "theta",
    "P_VEGA": "vega",
}

OUTPUT_COLUMNS = [
    "underlying",
    "underlying_last",
    "optionroot",
    "type",
    "expiration",
    "quotedate",
    "strike",
    "last",
    "bid",
    "ask",
    "volume",
    "openinterest",
    "impliedvol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "optionalias",
]


def make_optionroot(expire_dates, option_type, strikes):
    """Generate OCC-format option root symbols vectorized.

    Format: SPX{YYMMDD}{C|P}{strike*1000:08d}
    Example: SPX170317C00300000
    """
    date_str = expire_dates.dt.strftime("%y%m%d")
    type_char = "C" if option_type == "call" else "P"
    strike_str = (strikes * 1000).astype(int).astype(str).str.zfill(8)
    return "SPX" + date_str + type_char + strike_str


def convert(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=["QUOTE_DATE", "EXPIRE_DATE"])
    # Strip whitespace from column names (OptionsDX CSVs have trailing spaces)
    df.columns = df.columns.str.strip()

    shared = {
        "underlying": "SPX",
        "underlying_last": df["UNDERLYING_LAST"],
        "expiration": df["EXPIRE_DATE"],
        "quotedate": df["QUOTE_DATE"],
        "strike": df["STRIKE"],
        "openinterest": 0,
    }

    # Build call rows
    calls = pd.DataFrame(shared)
    calls["type"] = "call"
    for src, dst in CALL_COLS.items():
        calls[dst] = df[src].values
    calls["optionroot"] = make_optionroot(df["EXPIRE_DATE"], "call", df["STRIKE"])
    calls["optionalias"] = calls["optionroot"]

    # Build put rows
    puts = pd.DataFrame(shared)
    puts["type"] = "put"
    for src, dst in PUT_COLS.items():
        puts[dst] = df[src].values
    puts["optionroot"] = make_optionroot(df["EXPIRE_DATE"], "put", df["STRIKE"])
    puts["optionalias"] = puts["optionroot"]

    result = pd.concat([calls, puts], ignore_index=True)
    result = result[OUTPUT_COLUMNS]
    result = result.sort_values(["quotedate", "expiration", "strike", "type"])
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OptionsDX wide CSV to backtester long CSV"
    )
    parser.add_argument("input", help="Path to OptionsDX CSV file")
    parser.add_argument(
        "--output",
        default="data/processed/spx_options.csv",
        help="Output path (default: data/processed/spx_options.csv)",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
