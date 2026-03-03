# Data Scripts

Scripts for fetching and converting market data into the formats expected by the backtester.

## Quick Start

Fetch both stock and options data for SPY, aligned by date:

```bash
python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01
```

Data is first fetched from the [self-hosted GitHub Release](https://github.com/lambdaclass/options_backtester/releases/tag/data-v1), falling back to [philippdubach/options-data](https://github.com/philippdubach/options-data) CDN and yfinance. Outputs:
- `data/processed/stocks.csv` — Tiingo-format stock data
- `data/processed/options.csv` — options data with Greeks

## Subcommands

```bash
# Stocks only (GitHub Release > options-data > yfinance)
python data/fetch_data.py stocks --symbols SPY --start 2020-01-01 --end 2023-01-01

# Options only
python data/fetch_data.py options --symbols SPY --start 2020-01-01 --end 2023-01-01

# Both + date alignment (default)
python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01

# Multiple symbols
python data/fetch_data.py all --symbols SPY IWM QQQ --start 2020-01-01 --end 2023-01-01

# Custom output paths
python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01 \
    --stocks-output data/processed/spy_stocks.csv \
    --options-output data/processed/spy_options.csv

# Force re-download (skip cache)
python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01 --force
```

## OptionsDX Conversion (separate)

For SPX index options from [optionsdx.com](https://www.optionsdx.com/):

```bash
python data/convert_optionsdx.py data/raw/spx_eod_2020.csv --output data/processed/spx_options.csv
```

## Loading Data in the Backtester

```python
from backtester.datahandler import HistoricalOptionsData, TiingoData

options = HistoricalOptionsData("data/processed/options.csv")
stocks = TiingoData("data/processed/stocks.csv")
```

The `all` subcommand automatically aligns stock and option dates so the backtester's `np.array_equal` assertion passes.

## Directory Structure

- `raw/` — Cached parquet downloads (gitignored)
- `processed/` — Converted CSV output ready for the backtester (gitignored)
