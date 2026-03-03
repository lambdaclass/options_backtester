#!/usr/bin/env python3
"""Unified data fetch script for the options backtester.

Downloads stock and options data, converts to backtester CSV formats,
and aligns dates between datasets.

Download priority (for each symbol):
  1. Self-hosted GitHub Release (lambdaclass/options_backtester data-v1)
  2. philippdubach/options-data CDN — 104 symbols
  3. philippdubach/options-dataset-hist — SPY/IWM/QQQ underlying prices
  4. yfinance (last resort, stocks only)

Usage:
    python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01
    python data/fetch_data.py stocks --symbols SPY --start 2020-01-01 --end 2023-01-01
    python data/fetch_data.py options --symbols SPY --start 2020-01-01 --end 2023-01-01
    python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01 --update
"""

import argparse
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

# Self-hosted data on GitHub Releases (primary source)
RELEASE_URL = "https://github.com/lambdaclass/options_backtester/releases/download/data-v1"

# philippdubach/options-data — 104 symbols, options + underlying (underlying empty for some ETFs)
OPTIONS_DATA_URL = "https://static.philippdubach.com/data/options"

# philippdubach/options-dataset-hist — SPY/IWM/QQQ, proper underlying_prices via GitHub LFS
HIST_REPO_RAW = "https://github.com/philippdubach/options-dataset-hist/raw/main/data"
HIST_SYMBOLS = {"SPY", "IWM", "QQQ"}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url, dest, force=False):
    """Download url to dest. Returns dest on success, None on failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"  Using cached {dest}")
        return dest

    print(f"  Downloading {url} ...")
    try:
        req = Request(url, headers={"User-Agent": "options-backtester/1.0"})
        with urlopen(req) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return None
    return dest


def download_options_parquet(symbol, force=False):
    """Download options parquet. Priority: GitHub Release > options-data CDN."""
    sym = symbol.upper()

    # 1. Self-hosted GitHub Release
    dest = RAW_DIR / "release" / f"{sym}_options.parquet"
    url = f"{RELEASE_URL}/{sym}_options.parquet"
    result = _download(url, dest, force)
    if result is not None:
        return result

    # 2. options-data CDN
    dest = RAW_DIR / "options-data" / sym / "options.parquet"
    url = f"{OPTIONS_DATA_URL}/{sym.lower()}/options.parquet"
    return _download(url, dest, force)


def download_underlying(symbol, force=False):
    """Download underlying prices.

    Priority: GitHub Release > options-dataset-hist > options-data > None (caller falls back to yfinance).
    """
    sym = symbol.upper()

    # 1. Self-hosted GitHub Release
    dest = RAW_DIR / "release" / f"{sym}_underlying.parquet"
    url = f"{RELEASE_URL}/{sym}_underlying.parquet"
    result = _download(url, dest, force)
    if result is not None:
        df = pd.read_parquet(result)
        if not df.empty:
            return result
        print(f"  Warning: release underlying empty for {sym}")

    # 2. options-dataset-hist has proper underlying for SPY/IWM/QQQ
    if sym in HIST_SYMBOLS:
        dest = RAW_DIR / "options-dataset-hist" / sym / "underlying_prices.parquet"
        url = f"{HIST_REPO_RAW}/parquet_{sym.lower()}/underlying_prices.parquet"
        result = _download(url, dest, force)
        if result is not None:
            df = pd.read_parquet(result)
            if not df.empty:
                return result
            print(f"  Warning: options-dataset-hist underlying empty for {sym}")

    # 3. options-data underlying
    dest = RAW_DIR / "options-data" / sym / "underlying.parquet"
    url = f"{OPTIONS_DATA_URL}/{sym.lower()}/underlying.parquet"
    result = _download(url, dest, force)
    if result is not None:
        df = pd.read_parquet(result)
        if not df.empty:
            return result
        print(f"  Warning: options-data underlying empty for {sym}")

    return None


# ---------------------------------------------------------------------------
# Underlying price reading
# ---------------------------------------------------------------------------

def read_underlying_prices(symbol, und_path, start, end):
    """Read underlying parquet and return (date, close) DataFrame for joining."""
    und = pd.read_parquet(und_path)
    und["date"] = pd.to_datetime(und["date"])
    und = und[(und["date"] >= start) & (und["date"] <= end)]
    if und.empty:
        return None
    return und[["date", "close"]].rename(columns={"close": "underlying_last"})


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
    """Fetch one symbol via yfinance (last resort)."""
    try:
        import yfinance as yf
    except ImportError:
        print(f"  yfinance not installed, cannot fetch {symbol}", file=sys.stderr)
        return pd.DataFrame()

    print(f"  Last resort: fetching {symbol} from yfinance...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=str(start.date()), end=str(end.date()), auto_adjust=False)

    if df.empty:
        return pd.DataFrame()

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


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def fetch_options(symbols, start, end, output, force=False):
    """Download options parquets and convert to backtester CSV format."""
    frames = []

    for symbol in symbols:
        sym = symbol.upper()
        print(f"Fetching options for {sym}...")

        opt_path = download_options_parquet(sym, force)
        if opt_path is None:
            print(f"  Skipping {sym} options (download failed)", file=sys.stderr)
            continue

        opts = pd.read_parquet(opt_path)
        opts["date"] = pd.to_datetime(opts["date"])
        opts = opts[(opts["date"] >= start) & (opts["date"] <= end)]

        if opts.empty:
            print(f"  No options data for {sym} in [{start}, {end}]")
            continue

        # Get underlying close prices for underlying_last
        und_path = download_underlying(sym, force)
        und_prices = None
        if und_path is not None:
            und_prices = read_underlying_prices(sym, und_path, start, end)

        if und_prices is None:
            yf_df = fetch_yfinance(sym, start, end)
            if not yf_df.empty:
                und_prices = pd.DataFrame({
                    "date": pd.to_datetime(yf_df["date"]),
                    "underlying_last": yf_df["close"].values,
                })

        if und_prices is not None:
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

def fetch_stocks(symbols, start, end, output, force=False):
    """Download stock data. Priority: options-dataset-hist > options-data > yfinance."""
    frames = []

    for symbol in symbols:
        sym = symbol.upper()
        print(f"Fetching stocks for {sym}...")

        und_path = download_underlying(sym, force)
        if und_path is not None:
            df = underlying_to_tiingo(sym, und_path, start, end)
            if not df.empty:
                source = "options-dataset-hist" if "options-dataset-hist" in str(und_path) else "options-data"
                frames.append(df)
                print(f"  {len(df)} stock rows for {sym} (from {source})")
                continue

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
    """Align stock and option dates to their intersection."""
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
        "command", nargs="?", default="all",
        choices=["all", "stocks", "options"],
        help="What to fetch (default: all)",
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="Ticker symbols (e.g. SPY IWM QQQ AAPL)",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--stocks-output", default="data/processed/stocks.csv",
        help="Stock CSV output path",
    )
    parser.add_argument(
        "--options-output", default="data/processed/options.csv",
        help="Options CSV output path",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Re-download parquets to get latest data",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    force = args.update

    if args.command in ("all", "options"):
        fetch_options(args.symbols, start, end, args.options_output, force)

    if args.command in ("all", "stocks"):
        fetch_stocks(args.symbols, start, end, args.stocks_output, force)

    if args.command == "all":
        if Path(args.stocks_output).exists() and Path(args.options_output).exists():
            print("\nAligning dates...")
            align_dates(args.stocks_output, args.options_output)

    print("\nDone!")


if __name__ == "__main__":
    main()
