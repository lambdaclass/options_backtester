"""Generate large deterministic synthetic datasets for 3-way parity tests.

Produces stock and options CSVs with the same schema as the real test data
in backtester/test/test_data/, but covering 500 trading days with fixed
strikes per expiration cycle (like real listed options).

Usage:
    python -m tests.bench.generate_test_data
"""

from __future__ import annotations

import os
from math import erf, sqrt, log, exp
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# 7 stocks matching test_data_stocks.csv symbols
STOCK_SYMBOLS = ["VOO", "TLT", "EWY", "PDBC", "IAU", "VNQI", "VTIP"]
STOCK_INITIAL_PRICES = [210.0, 120.0, 55.0, 18.0, 23.0, 52.0, 80.0]

UNDERLYING = "SPX"
N_TRADING_DAYS = 500


def _generate_trading_dates(n: int, start: str = "2017-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _random_walk(rng: np.random.Generator, initial: float, n: int,
                 drift: float = 0.0002, vol: float = 0.015) -> np.ndarray:
    log_returns = rng.normal(drift, vol, size=n)
    prices = initial * np.exp(np.cumsum(log_returns))
    return np.round(prices, 6)


def generate_stocks(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for sym, p0 in zip(STOCK_SYMBOLS, STOCK_INITIAL_PRICES):
        prices = _random_walk(rng, p0, len(dates))
        for date, price in zip(dates, prices):
            high = round(price * (1 + rng.uniform(0.001, 0.015)), 6)
            low = round(price * (1 - rng.uniform(0.001, 0.015)), 6)
            opn = round(price * (1 + rng.uniform(-0.005, 0.005)), 6)
            vol = int(rng.integers(500_000, 10_000_000))
            rows.append({
                "symbol": sym, "date": date.strftime("%Y-%m-%d"),
                "close": round(price, 2), "high": round(high, 2),
                "low": round(low, 2), "open": round(opn, 2),
                "volume": vol, "adjClose": price, "adjHigh": high,
                "adjLow": low, "adjOpen": opn, "adjVolume": vol,
                "divCash": 0.0, "splitFactor": 1.0,
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _bs_approx(S: float, K: float, T: float, vol: float, is_call: bool) -> dict:
    """Quick Black-Scholes approximation for option pricing."""
    T = max(T, 1 / 365)
    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (0.02 + 0.5 * vol**2) * T) / (vol * sqrt_T)
    nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
    if is_call:
        delta = round(nd1, 4)
        intrinsic = max(S - K, 0)
    else:
        delta = round(nd1 - 1, 4)
        intrinsic = max(K - S, 0)
    time_value = vol * S * sqrt_T * 0.4
    mid_price = max(intrinsic + time_value * abs(delta), 0.05)
    spread = max(mid_price * 0.03, 0.05)
    bid = round(max(mid_price - spread / 2, 0.0), 2)
    ask = round(mid_price + spread / 2, 2)
    if bid <= 0:
        bid = 0.0
    gamma = round(max(0.0001, 0.01 * exp(-0.5 * d1**2) / (S * vol * sqrt_T)), 4)
    theta = round(-S * vol * gamma / (2 * sqrt_T), 4)
    vega = round(S * sqrt_T * gamma * 0.01, 4)
    return {
        "bid": bid, "ask": ask, "last": round((bid + ask) / 2, 2),
        "delta": delta, "gamma": gamma, "theta": theta, "vega": vega,
        "impliedvol": round(vol, 4),
    }


def generate_options(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate options data with FIXED strikes per expiration cycle.

    Like real listed options: once an expiration is listed, its strikes
    remain constant across all quote dates until expiration.
    """
    underlying_prices = _random_walk(rng, 2260.0, len(dates), drift=0.0003, vol=0.01)

    # Monthly expirations (3rd Friday)
    all_expirations = pd.bdate_range(
        start=dates[0] + pd.Timedelta(days=30),
        end=dates[-1] + pd.Timedelta(days=150),
        freq="WOM-3FRI",
    )

    # Pre-compute FIXED strikes for each expiration based on the underlying
    # price at the time the expiration first becomes active (~90 DTE).
    # Use 5 strike levels: 90%, 95%, 100%, 105%, 110% of the reference price.
    exp_strikes: dict[pd.Timestamp, list[int]] = {}
    date_to_price = dict(zip(dates, underlying_prices))
    for exp_date in all_expirations:
        # Reference date: ~90 days before expiration
        ref_date_target = exp_date - pd.Timedelta(days=90)
        # Find the closest actual trading date
        ref_date = min(dates, key=lambda d: abs((d - ref_date_target).days))
        ref_price = date_to_price.get(ref_date, 2260.0)
        # Round strikes to nearest 50 (like real SPX options)
        exp_strikes[exp_date] = [
            int(round(ref_price * pct / 50) * 50)
            for pct in [0.90, 0.95, 1.00, 1.05, 1.10]
        ]

    rows = []
    for i, (qdate, spx_price) in enumerate(zip(dates, underlying_prices)):
        active_exps = [
            exp_date for exp_date in all_expirations
            if 5 <= (exp_date - qdate).days <= 120
        ]
        if not active_exps:
            continue

        exps_to_use = active_exps[:4]  # up to 4 active cycles
        vol_base = 0.15 + 0.05 * np.sin(i / 50)

        for exp_date in exps_to_use:
            dte = (exp_date - qdate).days
            exp_str = exp_date.strftime("%Y-%m-%d")
            T = dte / 365.0
            strikes = exp_strikes[exp_date]

            for strike in strikes:
                for opt_type in ["call", "put"]:
                    is_call = opt_type == "call"
                    vol = vol_base + rng.uniform(-0.02, 0.02)
                    greeks = _bs_approx(spx_price, strike, T, vol, is_call)

                    exp_code = exp_date.strftime("%y%m%d")
                    type_code = "C" if is_call else "P"
                    strike_code = f"{int(strike):08d}00"
                    contract = f"SPX{exp_code}{type_code}{strike_code}"

                    rows.append({
                        "underlying": UNDERLYING,
                        "underlying_last": round(spx_price, 2),
                        " exchange": "*",
                        "optionroot": contract,
                        "optionext": "",
                        "type": opt_type,
                        "expiration": exp_str,
                        "quotedate": qdate.strftime("%Y-%m-%d"),
                        "strike": strike,
                        "last": greeks["last"],
                        "bid": greeks["bid"],
                        "ask": greeks["ask"],
                        "volume": int(rng.integers(0, 5000)),
                        "openinterest": int(rng.integers(0, 50000)),
                        "impliedvol": greeks["impliedvol"],
                        "delta": greeks["delta"],
                        "gamma": greeks["gamma"],
                        "theta": greeks["theta"],
                        "vega": greeks["vega"],
                        "optionalias": contract,
                        "dte": dte,
                    })

    df = pd.DataFrame(rows)
    df = df.sort_values(["quotedate", "optionroot"]).reset_index(drop=True)
    return df


def main():
    rng = np.random.default_rng(SEED)
    dates = _generate_trading_dates(N_TRADING_DAYS)

    print(f"Generating {N_TRADING_DAYS} trading days: {dates[0].date()} to {dates[-1].date()}")

    stocks_df = generate_stocks(dates, rng)
    print(f"Stocks: {len(stocks_df)} rows, {stocks_df['symbol'].nunique()} symbols")

    options_df = generate_options(dates, rng)
    print(f"Options: {len(options_df)} rows, {options_df['quotedate'].nunique()} dates, "
          f"~{len(options_df) / options_df['quotedate'].nunique():.0f} contracts/date")

    # Verify contracts persist across dates
    sample_contract = options_df['optionroot'].iloc[0]
    dates_for_contract = options_df[options_df['optionroot'] == sample_contract]['quotedate'].nunique()
    print(f"Sample contract {sample_contract} appears on {dates_for_contract} dates")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stocks_path = OUTPUT_DIR / "large_stocks.csv"
    options_path = OUTPUT_DIR / "large_options.csv"

    stocks_df.to_csv(stocks_path, index=False)
    options_df.to_csv(options_path, index=False)
    print(f"\nWritten:\n  {stocks_path}\n  {options_path}")


if __name__ == "__main__":
    main()
