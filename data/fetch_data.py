#!/usr/bin/env python3
"""Unified data fetch script for the options backtester.

Downloads stock and options data, converts to backtester CSV formats,
and aligns dates between datasets.

Data source: philippdubach/options-data (MIT license)
  https://github.com/philippdubach/options-data
  104 US equity/ETF symbols, 2008-2025, Parquet via Cloudflare R2.

Usage:
    python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01
    python data/fetch_data.py stocks --symbols SPY --start 2020-01-01 --end 2023-01-01
    python data/fetch_data.py options --symbols SPY --start 2020-01-01 --end 2023-01-01
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

OPTIONS_DATA_URL = "https://static.philippdubach.com/data/options"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_parquet(symbol, kind, dest_dir, force=False):
    """Download a parquet file from options-data if not already cached.

    kind: 'options' or 'underlying'
    Returns path to the downloaded file, or None on failure.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{kind}.parquet"
    dest = dest_dir / filename

    if dest.exists() and not force:
        print(f"  Using cached {dest}")
        return dest

    url = f"{OPTIONS_DATA_URL}/{symbol.lower()}/{filename}"
    print(f"  Downloading {url} ...")
    try:
        urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  Error downloading {symbol} {kind}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return None
    return dest


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def fetch_options(symbols, start, end, output, force=False):
    """Download options parquets and convert to backtester CSV format."""
    frames = []

    for symbol in symbols:
        sym = symbol.upper()
        sym_dir = RAW_DIR / "options-data" / sym

        print(f"Fetching options for {sym}...")
        opt_path = download_parquet(sym, "options", sym_dir, force)
        und_path = download_parquet(sym, "underlying", sym_dir, force)

        if opt_path is None:
            print(f"  Skipping {sym} options (download failed)", file=sys.stderr)
            continue

        opts = pd.read_parquet(opt_path)
        opts["date"] = pd.to_datetime(opts["date"])
        opts = opts[(opts["date"] >= start) & (opts["date"] <= end)]

        if opts.empty:
            print(f"  No options data for {sym} in [{start}, {end}]")
            continue

        # Join underlying close price
        if und_path is not None:
            und = pd.read_parquet(und_path)
            und["date"] = pd.to_datetime(und["date"])
            und_prices = und[["date", "close"]].rename(columns={"close": "underlying_last"})
            opts = opts.merge(und_prices, on="date", how="left")
        else:
            opts["underlying_last"] = float("nan")

        # Last price: use column if present, else mid
        if "last" in opts.columns:
            opts["_last"] = opts["last"].fillna((opts["bid"] + opts["ask"]) / 2)
        else:
            opts["_last"] = (opts["bid"] + opts["ask"]) / 2

        out = pd.DataFrame({
            "underlying": sym,
            "underlying_last": opts["underlying_last"].values,
            "optionroot": opts["contract_id"].values,
            "type": opts["type"].values,
            "expiration": pd.to_datetime(opts["expiration"]).values,
            "quotedate": opts["date"].values,
            "strike": opts["strike"].values,
            "last": opts["_last"].values,
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
        frames.append(out)
        print(f"  {len(out)} option rows for {sym}")

    if not frames:
        print("No options data fetched.", file=sys.stderr)
        return None

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["quotedate", "underlying", "expiration", "strike", "type"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} option rows to {output}")
    return result


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

def underlying_to_tiingo(symbol, und_path, start, end):
    """Convert an underlying.parquet to Tiingo-format DataFrame."""
    und = pd.read_parquet(und_path)
    und["date"] = pd.to_datetime(und["date"])
    und = und[(und["date"] >= start) & (und["date"] <= end)]

    if und.empty:
        return pd.DataFrame()

    ratio = und["adjusted_close"] / und["close"]

    return pd.DataFrame({
        "symbol": symbol,
        "date": und["date"].values,
        "close": und["close"].values,
        "high": und["high"].values,
        "low": und["low"].values,
        "open": und["open"].values,
        "volume": und["volume"].values,
        "adjClose": und["adjusted_close"].values,
        "adjHigh": (und["high"] * ratio).values,
        "adjLow": (und["low"] * ratio).values,
        "adjOpen": (und["open"] * ratio).values,
        "adjVolume": und["volume"].values,
        "divCash": und["dividend_amount"].values,
        "splitFactor": und["split_coefficient"].values,
    })


def fetch_yfinance(symbol, start, end):
    """Fetch one symbol via yfinance (fallback for symbols not in options-data)."""
    try:
        import yfinance as yf
    except ImportError:
        print(f"  yfinance not installed, cannot fetch {symbol}", file=sys.stderr)
        return pd.DataFrame()

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=str(start.date()), end=str(end.date()), auto_adjust=False)

    if df.empty:
        return pd.DataFrame()

    # Strip timezone for naive dates
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ratio = df["Adj Close"] / df["Close"]

    return pd.DataFrame({
        "symbol": symbol,
        "date": df.index,
        "close": df["Close"].values,
        "high": df["High"].values,
        "low": df["Low"].values,
        "open": df["Open"].values,
        "volume": df["Volume"].values,
        "adjClose": df["Adj Close"].values,
        "adjHigh": (df["High"] * ratio).values,
        "adjLow": (df["Low"] * ratio).values,
        "adjOpen": (df["Open"] * ratio).values,
        "adjVolume": df["Volume"].values,
        "divCash": 0.0,
        "splitFactor": 1.0,
    })


def fetch_stocks(symbols, start, end, output, force=False):
    """Download stock data.  Uses options-data underlying if available, else yfinance."""
    frames = []

    for symbol in symbols:
        sym = symbol.upper()
        print(f"Fetching stocks for {sym}...")

        # Try options-data underlying first
        sym_dir = RAW_DIR / "options-data" / sym
        und_path = sym_dir / "underlying.parquet"

        if not und_path.exists() or force:
            downloaded = download_parquet(sym, "underlying", sym_dir, force)
            if downloaded is not None:
                und_path = downloaded

        if und_path.exists():
            df = underlying_to_tiingo(sym, und_path, start, end)
            if not df.empty:
                frames.append(df)
                print(f"  {len(df)} stock rows for {sym} (from options-data)")
                continue

        # Fallback to yfinance
        print(f"  Trying yfinance for {sym}...")
        df = fetch_yfinance(sym, start, end)
        if not df.empty:
            frames.append(df)
            print(f"  {len(df)} stock rows for {sym} (from yfinance)")
        else:
            print(f"  No stock data for {sym}", file=sys.stderr)

    if not frames:
        print("No stock data fetched.", file=sys.stderr)
        return None

    result = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} stock rows to {output}")
    return result


# ---------------------------------------------------------------------------
# Date alignment
# ---------------------------------------------------------------------------

def align_dates(stocks_path, options_path):
    """Align stock and option dates to their intersection.

    The backtester asserts np.array_equal on dates, so both CSVs must
    cover exactly the same set of trading days.
    """
    stocks = pd.read_csv(stocks_path, parse_dates=["date"])
    options = pd.read_csv(options_path, parse_dates=["quotedate", "expiration"])

    stock_dates = set(stocks["date"].dt.normalize())
    option_dates = set(options["quotedate"].dt.normalize())
    shared = stock_dates & option_dates

    if not shared:
        print("Warning: no overlapping dates between stocks and options!", file=sys.stderr)
        return

    stocks_filtered = stocks[stocks["date"].dt.normalize().isin(shared)]
    options_filtered = options[options["quotedate"].dt.normalize().isin(shared)]

    stocks_filtered.to_csv(stocks_path, index=False)
    options_filtered.to_csv(options_path, index=False)

    dropped_stock = len(stocks) - len(stocks_filtered)
    dropped_opt = len(options) - len(options_filtered)
    print(f"Aligned dates: {len(shared)} shared trading days")
    if dropped_stock:
        print(f"  Dropped {dropped_stock} stock rows without matching option dates")
    if dropped_opt:
        print(f"  Dropped {dropped_opt} option rows without matching stock dates")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch stock and options data for the backtester"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "stocks", "options"],
        help="What to fetch (default: all)",
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="Ticker symbols (e.g. SPY IWM QQQ)",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--stocks-output",
        default="data/processed/stocks.csv",
        help="Stock CSV output path (default: data/processed/stocks.csv)",
    )
    parser.add_argument(
        "--options-output",
        default="data/processed/options.csv",
        help="Options CSV output path (default: data/processed/options.csv)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if parquets are cached locally",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if args.command in ("all", "options"):
        fetch_options(args.symbols, start, end, args.options_output, args.force)

    if args.command in ("all", "stocks"):
        fetch_stocks(args.symbols, start, end, args.stocks_output, args.force)

    if args.command == "all":
        if Path(args.stocks_output).exists() and Path(args.options_output).exists():
            print("\nAligning dates...")
            align_dates(args.stocks_output, args.options_output)

    print("\nDone!")


if __name__ == "__main__":
    main()
