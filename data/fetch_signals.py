#!/usr/bin/env python3
"""Download macro signal data from FRED for use in backtest signal filters.

Downloads:
  - GDP (quarterly) — for Buffett Indicator proxy
  - VIX (daily) — CBOE Volatility Index
  - High Yield Spread (daily) — credit stress indicator
  - 10Y-2Y Yield Curve (daily) — recession predictor
  - Nonfinancial Corporate Equity Market Value (quarterly) — for Tobin's Q
  - Nonfinancial Corporate Net Worth (quarterly) — for Tobin's Q
  - Dollar Index (daily) — broad trade-weighted USD

Outputs:
  data/processed/signals.csv — daily signal data, forward-filled from quarterly
"""

import io
import urllib.request

import pandas as pd

FRED_SERIES = {
    'gdp': 'GDP',
    'vix': 'VIXCLS',
    'hy_spread': 'BAMLH0A0HYM2',
    'yield_curve_10y2y': 'T10Y2Y',
    'nfc_equity_mv': 'NCBEILQ027S',
    'nfc_net_worth': 'NCBCMDPMVCE',
    'dollar_index': 'DTWEXBGS',
}

START = '2007-01-01'
END = '2025-12-31'


def fetch_fred(series_id: str) -> pd.Series:
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={series_id}&cosd={START}&coed={END}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read().decode()
    df = pd.read_csv(io.StringIO(data), parse_dates=['observation_date'],
                     index_col='observation_date')
    col = df.columns[0]
    s = pd.to_numeric(df[col], errors='coerce')
    s.index.name = 'date'
    return s.dropna()


def main():
    signals = {}

    for name, sid in FRED_SERIES.items():
        print(f'Fetching {name} ({sid})...', end=' ', flush=True)
        try:
            s = fetch_fred(sid)
            signals[name] = s
            print(f'{len(s)} obs, {s.index[0].date()} to {s.index[-1].date()}')
        except Exception as e:
            print(f'FAILED: {e}')

    if not signals:
        print('No data fetched.')
        return

    # Build daily DataFrame
    all_dates = sorted(set().union(*(s.index for s in signals.values())))
    daily = pd.DataFrame(index=pd.DatetimeIndex(all_dates, name='date'))

    for name, s in signals.items():
        daily[name] = s.reindex(daily.index)

    # Forward-fill quarterly data to daily
    daily = daily.ffill()

    # Compute derived signals
    if 'gdp' in daily.columns:
        # Buffett Indicator proxy: we don't have total market cap, but
        # nfc_equity_mv is corporate equity market value (in millions)
        # GDP is in billions. Scale NFC equity to billions to match.
        if 'nfc_equity_mv' in daily.columns:
            daily['buffett_indicator'] = daily['nfc_equity_mv'] / (daily['gdp'] * 1000) * 100
            print(f'Computed buffett_indicator (nfc_equity_mv / GDP)')

    if 'nfc_equity_mv' in daily.columns and 'nfc_net_worth' in daily.columns:
        # Tobin's Q proxy: market value / net worth
        # nfc_net_worth is in weird units (ratio), use nfc_equity_mv levels
        # Actually NCBCMDPMVCE is "market value / cost" already
        daily['tobin_q'] = daily['nfc_net_worth']
        print(f'Computed tobin_q (NCBCMDPMVCE is already MV/replacement cost)')

    daily = daily.dropna(how='all')

    out = 'data/processed/signals.csv'
    daily.to_csv(out)
    print(f'\nSaved {len(daily)} rows to {out}')
    print(f'Columns: {list(daily.columns)}')
    print(f'Date range: {daily.index[0].date()} to {daily.index[-1].date()}')
    print(daily.describe().round(2))


if __name__ == '__main__':
    main()
