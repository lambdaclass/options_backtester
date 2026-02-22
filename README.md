Options Backtester
==================

Simple backtester to evaluate and analyse options strategies over historical price data.

- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Recommended Reading](#recommended-reading)

## Setup

### With Nix (recommended)

```shell
nix develop
```

This gives you Python 3.12 with all dependencies (pandas, numpy, altair, pytest, etc.).

### Without Nix

Requires Python >= 3.12.

```shell
pip install pandas numpy altair pyprind seaborn matplotlib pyarrow pytest yapf flake8
```

### Fetch data

Download SPY options and stock data for 2020-2023:

```shell
python data/fetch_data.py all --symbols SPY --start 2020-01-01 --end 2023-01-01
```

This fetches from the [self-hosted GitHub Release](https://github.com/lambdaclass/options_backtester/releases/tag/data-v1), falling back to external sources. See [data/README.md](data/README.md) for details.

### Run tests

```shell
python -m pytest -v backtester
```

## Usage

### Sample backtest

```python
from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg

# Load data
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

# Create a strategy: buy a call and a put with 52 < DTE < 80, exit at DTE <= 52
strategy = Strategy(schema)

leg1 = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.BUY)
leg1.entry_filter = (schema.dte < 80) & (schema.dte > 52)
leg1.exit_filter = (schema.dte <= 52)

leg2 = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=Direction.BUY)
leg2.entry_filter = (schema.dte < 80) & (schema.dte > 52)
leg2.exit_filter = (schema.dte <= 52)

strategy.add_legs([leg1, leg2])

# Define portfolio
stocks = [Stock('SPY', 1.0)]
allocation = {'stocks': 0.5, 'options': 0.5, 'cash': 0.0}

# Run backtest with monthly rebalancing
bt = Backtest(allocation, initial_capital=1_000_000)
bt.stocks = stocks
bt.stocks_data = stocks_data
bt.options_strategy = strategy
bt.options_data = options_data
bt.run(rebalance_freq=1)

# Results
bt.trade_log   # DataFrame of executed trades
bt.balance     # Daily portfolio balance
```

### Strangle preset

For common strategies there are presets:

```python
from backtester.strategy import Strangle

# Short strangle: sell OTM call + put, 30-60 DTE entry, exit at 7 DTE
strangle = Strangle(schema, 'short', 'SPY',
                    dte_entry_range=(30, 60), dte_exit=7,
                    otm_pct=5, pct_tolerance=1,
                    exit_thresholds=(0.2, 0.2))
```

### Custom strategies

The `Strategy` and `StrategyLeg` classes support arbitrary multi-leg strategies:

```python
# Long strangle
leg_1 = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
leg_1.entry_filter = (schema.underlying == 'SPY') & (schema.dte >= 60) \
                   & (schema.underlying_last <= 1.1 * schema.strike)
leg_1.exit_filter = (schema.dte <= 30)

leg_2 = StrategyLeg('leg_2', schema, option_type=Type.CALL, direction=Direction.BUY)
leg_2.entry_filter = (schema.underlying == 'SPY') & (schema.dte >= 60) \
                   & (schema.underlying_last >= 0.9 * schema.strike)
leg_2.exit_filter = (schema.dte <= 30)

strategy = Strategy(schema)
strategy.add_legs([leg_1, leg_2])
strategy.add_exit_thresholds(profit_pct=0.2, loss_pct=0.2)
```

More examples in the Jupyter [notebooks](backtester/examples/).

## Data

Data is hosted on [GitHub Releases](https://github.com/lambdaclass/options_backtester/releases/tag/data-v1) and downloaded on demand by `data/fetch_data.py`. Available symbols: SPY, IWM, QQQ (options + underlying, 2008-2025).

Fallback sources: [philippdubach/options-data](https://github.com/philippdubach/options-data) (104 symbols), [philippdubach/options-dataset-hist](https://github.com/philippdubach/options-dataset-hist) (SPY/IWM/QQQ underlying), yfinance.

See [data/README.md](data/README.md) for the full data pipeline documentation.

## Recommended reading

For complete novices in finance and economics, this [post](https://notamonadtutorial.com/how-to-earn-your-macroeconomics-and-finance-white-belt-as-a-software-developer-136e7454866f) gives a comprehensive introduction.

### Books

**Introductory**
- Option Volatility and Pricing 2nd Ed. - Natemberg, 2014
- Options, Futures, and Other Derivatives 10th Ed. - Hull 2017
- Trading Options Greeks 2nd Ed. - Passarelli 2012

**Intermediate**
- Trading Volatility - Bennet 2014
- Volatility Trading 2nd Ed. - Sinclair 2013

**Advanced**
- Dynamic Hedging - Taleb 1997
- The Volatility Surface: A Practitioner's Guide - Gatheral 2006
- The Volatility Smile - Derman & Miller 2016

### Papers

- [Volatility: A New Return Driver?](http://static.squarespace.com/static/53974e3ae4b0039937edb698/t/53da6400e4b0d5d5360f4918/1406821376095/Directional%20Volatility%20Research.pdf)
- [Easy Volatility Investing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2255327)
- [Everybody's Doing It: Short Volatility Strategies and Shadow Financial Insurers](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3071457)
- [Volatility-of-Volatility Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2497759)
- [The Distribution of Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2828744)
- [Safe Haven Investing Part I - Not all risk mitigation is created equal](https://www.universa.net/UniversaResearch_SafeHavenPart1_RiskMitigation.pdf)
- [Safe Haven Investing Part II - Not all risk is created equal](https://www.universa.net/UniversaResearch_SafeHavenPart2_NotAllRisk.pdf)
- [Safe Haven Investing Part III - Those wonderful tenbaggers](https://www.universa.net/UniversaResearch_SafeHavenPart3_Tenbaggers.pdf)
- [Insurance makes wealth grow faster](https://arxiv.org/abs/1507.04655)
- [Ergodicity economics](https://ergodicityeconomics.files.wordpress.com/2018/06/ergodicity_economics.pdf)
- [The Rate of Return on Everything, 1870-2015](https://economics.harvard.edu/files/economics/files/ms28533.pdf)
- [Volatility and the Alchemy of Risk](https://static1.squarespace.com/static/5581f17ee4b01f59c2b1513a/t/59ea16dbbe42d6ff1cae589f/1508513505640/Artemis_Volatility+and+the+Alchemy+of+Risk_2017.pdf)
