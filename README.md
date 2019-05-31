Options Backtester
==============================

Simple backtester to evaluate and analyse options strategies over historical price data.


- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Recommended Reading](#recommended-reading)
- [Data Sources](#data-sources)


## Requirements

- Python >= 3.5
- pipenv


## Setup

For backtesting, set `$OPTIONS_DATA_PATH` to the appropriate directory where the data is located. All file paths parsed by the backtester will be relative to this directory.  

To use the data scraper the following environment variables need to be set:
- `$SAVE_DATA_PATH`: where the data will be saved to (default is `./data/scraped`)
- `$TIINGO_API_KEY`: used to fetch data from [Tiingo](https://api.tiingo.com)
- `$S3_BUCKET`: name of the S3 bucket to backup data
- `$AWS_ACCESS_KEY_ID`: AWS acces key id
- `$AWS_SECRET_ACCESS_KEY`: AWS secret key
- `$SLACK_WEBHOOK`: used to send Slack notifications 

You can configure the data scraper by editing the configuration file `data_scraper.conf` (json-formated).  

Sample file:

```json
{
  "cboe": {
    "mute_notifications": ["BFB", "CBSA"]
  }
}
```

**HINT**: store environment variables in an `.env` file and pipenv will load them automatically when using `make env`.


## Usage

### Create environment and download dependencies

```shell
$> make init
```

### Activate environment

```shell
$> make env
```

### Run tests

```shell
$> make test
```

### Scrape data (supported scrapers: CBOE, Tiingo)

```shell
$> make scrape scraper=cboe

$> make scrape scraper=tiingo
```

### Run backtester with benchmark strategy

```shell
$> make bench
```


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

# Options trading strategies backtests

- [DTR Trading](https://dtr-trading.blogspot.com/)

### Exchanges

- [IEX](https://iextrading.com/developer/)
- [Tiingo](https://api.tiingo.com/)
- [CBOE Options Data](http://www.cboe.com/delayedquote/quote-table-download)

### Historical Data

- [Shiller's US Stocks, Dividends, Earnings, Inflation (CPI), and long term interest rates](http://www.econ.yale.edu/~shiller/data.htm)
- [Fama/French US Stock Index Data](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [FRED CPI, Interest Rates, Trade Data](https://fred.stlouisfed.org)
- [REIT Data](https://www.reit.com/data-research/reit-market-data/reit-industry-financial-snapshot)
