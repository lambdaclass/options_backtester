# Databento Market Data

Historical futures and options data from [Databento](https://databento.com/),
schema `ohlcv-1d` (daily OHLCV bars). Datasets: `GLBX.MDP3` (CME), `IFLL.IMPACT` (ICE Europe).

All data is stored as Parquet files with `ts_event` as the index (UTC timestamps).

## Data Inventory

### Equity Index

| File | Instrument | Rows | Date Range | Size |
|------|-----------|------|------------|------|
| `ES_FUT_ohlcv1d.parquet` | S&P 500 E-mini futures | 29K | 2010-06 to 2026-02 | 0.6 MB |
| `ES_OPT_ohlcv1d.parquet` | S&P 500 E-mini options | 3.4M | 2010-06 to 2026-02 | 53 MB |

### US Treasuries

| File | Instrument | Rows | Date Range | Size |
|------|-----------|------|------------|------|
| `ZN_FUT_ohlcv1d.parquet` | 10-Year T-Note futures | 17K | 2010-06 to 2026-02 | 0.3 MB |
| `ZB_FUT_ohlcv1d.parquet` | 30-Year T-Bond futures | 15K | 2010-06 to 2026-02 | 0.3 MB |
| `OZN_OPT_ohlcv1d.parquet` | 10-Year T-Note options | 1.5M | 2010-07 to 2026-02 | 25 MB |
| `OZB_OPT_ohlcv1d.parquet` | 30-Year T-Bond options | 1.3M | 2010-06 to 2026-02 | 18 MB |

### UK Gilts (ICE Europe -- `IFLL.IMPACT`)

| File | Instrument | Rows | Date Range | Size |
|------|-----------|------|------------|------|
| `R_FUT_ohlcv1d.parquet` | Long Gilt futures (10yr UK) | 7.6K | 2018-12 to 2026-02 | 0.2 MB |
| `R_OPT_ohlcv1d.parquet` | Long Gilt options | 4.1K | 2018-12 to 2026-02 | 0.1 MB |

Note: Gilt options are very sparse (4K rows vs 1.5M for OZN). Futures include outrights + calendar spreads.
Symbol format: `R   FMH0022!` (outright), spreads contain `-`. Options: `R   FMG0019_OMPA...` (P=put, C=call).

### SOFR (Short-Term Interest Rates)

| File | Instrument | Rows | Date Range | Size |
|------|-----------|------|------------|------|
| `SR1_FUT_ohlcv1d.parquet` | 1-Month SOFR futures | 76K | 2018-05 to 2026-02 | 0.8 MB |
| `SR3_FUT_ohlcv1d.parquet` | 3-Month SOFR futures | 365K | 2018-05 to 2026-02 | 4.0 MB |

### FX Futures (CME)

| File | Instrument | Rows | Date Range |
|------|-----------|------|------------|
| `6A_FUT` / `6A_OPT` | AUD/USD | 30K / 173K | 2010-06 to 2026-02 / 2010-2017 |
| `6B_FUT` / `6B_OPT` | GBP/USD | 31K / 168K | 2010-06 to 2026-02 / 2010-2022 |
| `6C_FUT` / `6C_OPT` | CAD/USD | 39K / 128K | 2010-06 to 2026-02 / 2010-2019 |
| `6E_FUT` / `6E_OPT` | EUR/USD | 45K / 440K | 2010-06 to 2026-02 / 2010-2017 |
| `6J_FUT` / `6J_OPT` | JPY/USD | 33K / 224K | 2010-06 to 2026-02 / 2010-2017 |
| `6M_FUT` / `6M_OPT` | MXN/USD | 12K / 9K | 2010-06 to 2026-02 |
| `6N_FUT` / `6N_OPT` | NZD/USD | 10K / 1K | 2010-06 to 2026-02 |
| `6S_FUT` / `6S_OPT` | CHF/USD | 14K / 50K | 2010-06 to 2026-02 / 2010-2017 |

### FX Weekly Options (CME)

| File | Instrument | Rows | Date Range |
|------|-----------|------|------------|
| `ADU_OPT` | AUD/USD weekly opts | 240K | 2016-08 to 2026-02 |
| `CAU_OPT` | CAD/USD weekly opts | 156K | 2016-08 to 2026-02 |
| `EUU_OPT` | EUR/USD weekly opts | 586K | 2016-08 to 2026-02 |
| `GBU_OPT` | GBP/USD weekly opts | 256K | 2016-08 to 2026-02 |
| `JPU_OPT` | JPY/USD weekly opts | 258K | 2016-08 to 2026-02 |

### Commodities

| File | Instrument | Rows | Date Range |
|------|-----------|------|------------|
| `GC_FUT` | Gold (COMEX) futures | 147K | 2010-06 to 2026-02 |
| `OG_OPT` | Gold options | 2.1M | 2010-06 to 2025-03 |
| `CL_FUT` | Crude Oil (WTI) futures | 734K | 2010-06 to 2026-02 |
| `LO_OPT` | Crude Oil options | 1.7M | 2010-06 to 2025-12 |
| `HG_FUT` | Copper (COMEX) futures | 166K | 2010-06 to 2026-02 |
| `HXE_OPT` | Copper options | 172K | 2010-06 to 2026-02 |
| `NG_FUT` | Natural Gas futures | 807K | 2010-06 to 2026-02 |
| `ON_OPT` | Natural Gas options | 621K | 2010-06 to 2026-02 |

## Schema

Each Parquet file contains these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ts_event` | datetime (index) | Trading date (UTC) |
| `rtype` | int | Record type (35 = OHLCV daily) |
| `publisher_id` | int | Data publisher ID |
| `instrument_id` | int | Databento instrument ID |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing/settlement price |
| `volume` | int | Trading volume |
| `symbol` | str | CME symbol (e.g., `ESM5`, `OZN H5 P119`) |

## Symbol Conventions

### Futures
Format: `{ROOT}{MONTH}{YEAR}` — e.g., `ESM5` = ES June 2025, `ZNZ3` = ZN December 2013/2023

Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

Year digit is ambiguous across decades (e.g., `3` = 2013 or 2023). Use trade date context for disambiguation.

Spread symbols contain `-` (e.g., `ESM5-ESU5`) and should be filtered out for front-month analysis.
Symbols starting with `UD:` are user-defined spreads and should also be filtered.

### Options
- **ES options**: `ESM5 P4200` (root + month/year + P/C + strike)
- **Treasury options**: `OZN H5 P119` (root + space + month/year + space + P/C + strike)
- **FX options**: `6EH5 P10800` (root + month/year + P/C + strike, strike in pips)
- **Commodity options**: `LOM5 P6500` (root + month/year + P/C + strike)

## How Data Was Downloaded

```python
import databento as db

client = db.Historical("YOUR_API_KEY")
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["ES.FUT"],       # or "ES.OPT", "OZN.OPT", etc.
    stype_in="parent",        # gets all contracts under parent symbol
    schema="ohlcv-1d",
    start="2010-06-06",
    end="2026-02-28",
)
df = data.to_df()
df.to_parquet("ES_FUT_ohlcv1d.parquet")
```

Large option datasets (>500K rows/year) require quarterly chunking to avoid 504 timeouts.

## Notebooks Using This Data

| Notebook | Data Used |
|----------|-----------|
| `multi_asset_carry.ipynb` | FX futures (6A-6S) + FX options |
| `commodity_carry.ipynb` | GC, CL, HG, NG futures + options |
| `equity_spitznagel.ipynb` | ES futures + options |
| `leverage_analysis.ipynb` | FX futures + options |
| `carry_portfolio.ipynb` | All FX + commodity futures |
| `treasury_spitznagel.ipynb` | ZN, ZB futures + OZN, OZB options + SR1, SR3 |
