Options Backtester
==================

Backtester for evaluating options strategies over historical data. Includes tools for strategy sweeps, tail-risk hedge analysis, and signal-based timing research.

- [Setup](#setup)
- [Usage](#usage)
- [Tail-Risk Hedge Research](#tail-risk-hedge-research)
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

Download SPY options and stock data (2008-2025):

```shell
python data/fetch_data.py all --symbols SPY --start 2008-01-01 --end 2025-12-31
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

# Create a strategy: buy puts with 60-120 DTE, exit at DTE <= 30
strategy = Strategy(schema)

leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
leg.entry_filter = (
    (schema.underlying == 'SPY') &
    (schema.dte >= 60) & (schema.dte <= 120) &
    (schema.delta >= -0.25) & (schema.delta <= -0.10)
)
leg.entry_sort = ('delta', False)  # deepest OTM within range
leg.exit_filter = (schema.dte <= 30)

strategy.add_leg(leg)

# Define portfolio — 100% stocks, puts funded by % of capital
bt = Backtest({'stocks': 1.0, 'options': 0.0, 'cash': 0.0}, initial_capital=1_000_000)
bt.options_budget = lambda date, total_capital: total_capital * 0.001  # 0.1% of capital
bt.stocks = [Stock('SPY', 1.0)]
bt.stocks_data = stocks_data
bt.options_strategy = strategy
bt.options_data = options_data
bt.run(rebalance_freq=1)  # monthly rebalancing

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

### Notebooks

| Notebook | Description |
|----------|-------------|
| [quickstart](notebooks/quickstart.ipynb) | Getting started — load data, define strategy, run backtest, plot results |
| [paper_comparison](notebooks/paper_comparison.ipynb) | **Master comparison**: 10 strategies vs academic paper claims with VRP math, crash heatmap, risk/return scatter |
| [findings](notebooks/findings.ipynb) | Full research: allocation sweep, puts vs calls, macro signals, crash-period analysis |
| [volatility_premium](notebooks/volatility_premium.ipynb) | Sell vol vs buy vol deep dive — tests the Variance Risk Premium (Carr & Wu 2009, Berman 2014) |
| [strategies](notebooks/strategies.ipynb) | 4-strategy showcase: OTM puts, OTM calls, long straddle, short strangle |
| [trade_analysis](notebooks/trade_analysis.ipynb) | Per-trade P&L deep dive: bar charts, cumulative P&L, crash breakdowns, winner/loser analysis |
| [iron_condor](notebooks/iron_condor.ipynb) | 4-leg iron condor income strategy with options capital breakdown |
| [ivy_portfolio](notebooks/ivy_portfolio.ipynb) | Endowment-style portfolio (Ivy Portfolio) with long straddle hedge overlay |
| [gold_sp500](notebooks/gold_sp500.ipynb) | Multi-asset portfolio with cash/gold proxy + options overlay (7 configs) |
| [spitznagel_case](notebooks/spitznagel_case.ipynb) | **The main analysis.** AQR vs Spitznagel tested with real data. Multi-dimensional parameter sweep (DTE, delta, exit, budget). Spitznagel's leveraged framing: 13.8–28.8%/yr with lower drawdowns. Implementation guide. |

See also [REFERENCES.md](REFERENCES.md) for 25+ academic papers on options overlay strategies.

## Tail-Risk Hedge Research

The main research question: **can a small allocation to SPY puts improve risk-adjusted returns over buy-and-hold?** Inspired by Universa Investments' approach to tail-risk hedging.

### Scripts

| Script | Purpose |
|--------|---------|
| `run_spy_otm_puts.py` | Strategy sweep: tests 6 hedge variants (delta, DTE, budget, profit caps) vs SPY buy-and-hold |
| `sweep_otm.py` | Delta band sweep: tests different OTM levels from near-ATM to deep OTM |
| `sweep_beat_spy3.py` | Diagnoses backtester rebalancing drag, then tries hedge configs that beat pure-stock baseline |
| `analyze_entries_exits.py` | **Per-trade analysis** with signal overlay — the main research tool |

### Key findings

**The framing matters more than the strategy.** The same deep OTM puts produce opposite conclusions depending on how you structure the portfolio:

| Framing | 1% deep OTM puts | Annual Return | Max DD | Verdict |
|---------|-------------------|---------------|--------|---------|
| AQR (reduce equity) | 99% SPY + 1% puts | +3.15% | -50.8% | Puts are a drag |
| Spitznagel (leverage) | 100% SPY + 1% puts on top | +16.46% | -45.0% | **Puts + leverage outperforms** |
| SPY B&H | 100% SPY | +11.11% | -51.9% | Baseline |

**Spitznagel's leveraged tail hedge** (100% SPY + deep OTM puts via budget callable) works:
- +0.5% budget: **13.79%/yr** (+2.75% excess), DD -48.1%
- +1.0% budget: **16.46%/yr** (+5.41% excess), DD -45.0%
- +3.3% budget: **28.78%/yr** (+17.74% excess), DD -31.9%

The outperformance comes from leverage, but the **drawdown reduction is real** — crash protection allows you to stay fully invested and even add exposure during drawdowns.

**Other findings:**
- Selling options (covered calls, put-writing, short strangles) harvests the Variance Risk Premium and outperforms on a risk-adjusted basis
- Buying OTM calls adds modest alpha (~0.8-1.4%/yr) in the no-leverage framing
- Macro signals (VIX, Buffett Indicator, Tobin's Q) don't improve put timing

### Configuration

All strategy parameters in `analyze_entries_exits.py` are configurable via the `CONFIG` dict:

```python
CONFIG = {
    'budget_pct': 0.1,      # % of capital per rebalance
    'rebalance_months': 1,  # monthly exit checks
    'delta_min': -0.25,     # delta range
    'delta_max': -0.10,
    'dte_min': 60,          # max 4 months DTE
    'dte_max': 120,
    'exit_dte': 30,         # sell with ~1 month left
    'profit_pct': math.inf, # no profit cap
    'loss_pct': math.inf,   # no loss cap
}
```

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
