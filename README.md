[![Build Status](https://travis-ci.com/lambdaclass/options_backtester.svg?branch=master)](https://travis-ci.com/lambdaclass/options_backtester)

Options Backtester
==============================

Simple backtester to evaluate and analyse options strategies over historical price data.


- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Recommended Reading](#recommended-reading)
- [Data Sources](#data-sources)


## Requirements

- Python >= 3.6
- pipenv

## Setup

Install [pipenv](https://pipenv.pypa.io/en/latest/)

```shell
$> pip install pipenv
```

Create environment and download dependencies

```shell
$> make install
```

Activate environment

```shell
$> make env
```

Run [Jupyter](https://jupyter.org) notebook

```shell
$> make notebook
```

Run tests

```shell
$> make test
```

## Usage

### Example:

We'll run a backtest of a stock portfolio holding `$AAPL` and `$GOOG`, and simultaneously buying 10% OTM calls and puts on `$SPX` ([long strangle](https://www.investopedia.com/terms/s/strangle.asp)).  
We'll allocate 97% of our capital to stocks and the rest to options, and do a rebalance every month.

```python
from backtester import Backtest, Type, Direction, Stock
from backtester.strategy import Strategy, StrategyLeg
from backtester.datahandler import HistoricalOptionsData, TiingoData

# Stocks data
stocks_data = TiingoData('stocks.csv')
stocks = [Stock(symbol='AAPL', percentage=0.5), Stock(symbol='GOOG', percentage=0.5)]

# Options data
options_data = HistoricalOptionsData('options.h5', key='/SPX')
schema = options_data.schema

# Long strangle
leg_1 = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
leg_1.entry_filter = (schema.underlying == 'SPX') & (schema.dte >= 60) & (schema.underlying_last <=
                                                                          1.1 * schema.strike)
leg_1.exit_filter = (schema.dte <= 30)

leg_2 = StrategyLeg('leg_2', schema, option_type=Type.CALL, direction=Direction.BUY)
leg_2.entry_filter = (schema.underlying == 'SPX') & (schema.dte >= 60) & (schema.underlying_last >=
                                                                          0.9 * schema.strike)
leg_2.exit_filter = (schema.dte <= 30)

strategy = Strategy(schema)
strategy.add_legs([leg_1, leg_2])

allocation = {'stocks': .97, 'options': .03}
initial_capital = 1_000_000
bt = Backtest(allocation, initial_capital)
bt.stocks = stocks
bt.stocks_data = stocks_data
bt.options_data = options_data
bt.options_strategy = strategy

bt.run(rebalance_freq=1)
```

You can explore more usage examples in the Jupyter [notebooks](backtester/examples/).

## Recommended reading

For complete novices in finance and economics, this [post](https://notamonadtutorial.com/how-to-earn-your-macroeconomics-and-finance-white-belt-as-a-software-developer-136e7454866f) gives a comprehensive introduction.


### Books

#### Introductory
- Option Volatility and Pricing 2nd Ed. - Natemberg, 2014
- Options, Futures, and Other Derivatives 10th Ed. - Hull 2017
- Trading Options Greeks: How Time, Volatility, and Other Pricing Factors Drive Profits 2nd Ed. - Passarelli 2012

#### Intermediate
- Trading Volatility - Bennet 2014
- Volatility Trading 2nd Ed. - Sinclair 2013

#### Advanced
- Dynamic Hedging - Taleb 1997
- The Volatility Surface: A Practitioner's Guide - Gatheral 2006
- The Volatility Smile - Derman & Miller 2016

### Papers
- [Volatility: A New Return Driver?](http://static.squarespace.com/static/53974e3ae4b0039937edb698/t/53da6400e4b0d5d5360f4918/1406821376095/Directional%20Volatility%20Research.pdf)
- [Easy Volatility Investing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2255327)
- [Everybody’s Doing It: Short Volatility Strategies and Shadow Financial Insurers](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3071457)
- [Volatility-of-Volatility Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2497759)
- [The Distribution of Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2828744)
- [Safe Haven Investing Part I - Not all risk mitigation is created equal](https://www.universa.net/UniversaResearch_SafeHavenPart1_RiskMitigation.pdf)
- [Safe Haven Investing Part II - Not all risk is created equal](https://www.universa.net/UniversaResearch_SafeHavenPart2_NotAllRisk.pdf)
- [Safe Haven Investing Part III - Those wonderful tenbaggers](https://www.universa.net/UniversaResearch_SafeHavenPart3_Tenbaggers.pdf)
- [Insurance makes wealth grow faster](https://arxiv.org/abs/1507.04655)
- [Ergodicity economics](https://ergodicityeconomics.files.wordpress.com/2018/06/ergodicity_economics.pdf)
- [The Rate of Return on Everything, 1870–2015](https://economics.harvard.edu/files/economics/files/ms28533.pdf)
- [Volatility and the Alchemy of Risk](https://static1.squarespace.com/static/5581f17ee4b01f59c2b1513a/t/59ea16dbbe42d6ff1cae589f/1508513505640/Artemis_Volatility+and+the+Alchemy+of+Risk_2017.pdf)

## Data sources

### Exchanges

- [IEX](https://iextrading.com/developer/)
- [Tiingo](https://api.tiingo.com/)
- [CBOE Options Data](http://www.cboe.com/delayedquote/quote-table-download)

### Historical Data

- [Shiller's US Stocks, Dividends, Earnings, Inflation (CPI), and long term interest rates](http://www.econ.yale.edu/~shiller/data.htm)
- [Fama/French US Stock Index Data](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [FRED CPI, Interest Rates, Trade Data](https://fred.stlouisfed.org)
- [REIT Data](https://www.reit.com/data-research/reit-market-data/reit-industry-financial-snapshot)
