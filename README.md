Options Portfolio Backtester
============================

Open-source backtesting framework for options, equities, and multi-asset portfolios. A strict superset of [bt](https://github.com/pmorissette/bt) — every bt feature is implemented, plus options support, richer execution modeling, and a Rust performance core.

**v0.3** — 444 tests, 6x faster than bt, full options Greeks chain.

- [Why this over bt?](#why-this-over-bt)
- [Setup](#setup)
- [Architecture](#architecture)
- [Usage](#usage)
- [Pipeline Algos](#pipeline-algos)
- [Pluggable Components](#pluggable-components)
- [Rust Performance Core](#rust-performance-core)
- [Notebooks](#notebooks)
- [Tail-Risk Hedge Research](#tail-risk-hedge-research)
- [Data](#data)
- [Recommended Reading](#recommended-reading)

## Why this over bt?

### Performance

| Benchmark | Time | vs bt |
|-----------|------|-------|
| Stock-only monthly rebalance | 0.6s | **6x faster** |
| Full options backtest (24.7M rows) | 4.2s (Rust) | bt can't do this |
| Parallel grid sweep (100 configs) | Rust + Rayon | **5-8x faster** |

### Everything bt has

All bt pipeline algos with matching names and semantics:

- **Scheduling**: `RunDaily`, `RunWeekly`, `RunMonthly`, `RunQuarterly`, `RunYearly`, `RunOnce`, `RunOnDate`, `RunAfterDate`, `RunAfterDays`, `RunEveryNPeriods`, `RunIfOutOfBounds`, `Or`, `Not`, `Require`
- **Selection**: `SelectAll`, `SelectThese`, `SelectHasData`, `SelectN`, `SelectMomentum`, `SelectWhere`, `SelectRandomly`, `SelectActive`, `SelectRegex`
- **Weighting**: `WeighEqually`, `WeighSpecified`, `WeighTarget`, `WeighInvVol`, `WeighMeanVar`, `WeighERC`, `TargetVol`, `WeighRandomly`
- **Weight limits**: `LimitWeights`, `LimitDeltas`, `ScaleWeights`
- **Rebalancing & position management**: `Rebalance`, `RebalanceOverTime`, `CapitalFlow`, `CloseDead`, `ClosePositionsAfterDates`, `ReplayTransactions`, `CouponPayingPosition`
- **Risk**: `MaxDrawdownGuard`, `HedgeRisks`, `Margin`
- **Analytics**: Sharpe/Sortino/Calmar, drawdowns, skew/kurtosis, lookback returns, turnover, Herfindahl, weights chart, `set_date_range()`, `to_dot()`, `benchmark_random()`

### Everything bt doesn't have

| Category | What we add |
|----------|-------------|
| **Options** | Multi-leg strategies (strangle, iron condor, butterfly, collar, covered call, cash-secured put), per-position Greeks (delta, gamma, theta, vega), strike/DTE/delta/IV filtering via Schema DSL, profit/loss exit thresholds, per-contract inventory, options chain iteration, dynamic budget callable |
| **Execution** | 4 cost models (per-contract, tiered volume, spread slippage), 3 fill models (bid/ask, mid, volume-aware), 3 signal selectors (first match, nearest delta, max OI), 4 position sizers, per-leg overrides |
| **Risk** | Pre-trade gating with composable constraints (`MaxDelta`, `MaxVega`, `MaxDrawdown`), Greeks aggregation |
| **Analytics** | Profit factor, tail ratio, win rate, monthly heatmap (Altair), tearsheet export (CSV/Markdown/HTML), round-trip trade P&L tracking, property-based fuzz testing |
| **Infrastructure** | Rust backend (PyO3/Polars/Rayon), parallel parameter sweeps, walk-forward optimization, structured event log, run metadata (git SHA, config hash), strategy tree with budget caps, Schema DSL for type-safe data access |

## Setup

### With Nix (recommended)

```shell
nix develop
```

This gives you Python 3.12 with all dependencies (pandas, numpy, altair, pytest, etc.) plus Rust toolchain (stable) and maturin.

### Without Nix

Requires Python >= 3.12.

```shell
python -m venv .venv
source .venv/bin/activate
make install-dev
```

For the optional Rust extension:
```shell
maturin develop --manifest-path rust/ob_python/Cargo.toml --release
```

### Fetch data

Download SPY options and stock data (2008-2025):

```shell
python data/fetch_data.py all --symbols SPY --start 2008-01-01 --end 2025-12-31
```

This fetches from the [self-hosted GitHub Release](https://github.com/lambdaclass/options_backtester/releases/tag/data-v1), falling back to external sources. See [data/README.md](data/README.md) for details.

### Run tests

```shell
make test          # all tests (Python + legacy)
make test-new      # new framework tests only
make test-bench    # benchmark/property tests (explicit opt-in)
make lint          # ruff linter
make typecheck     # mypy type checking
make rust-test     # Rust unit tests
make rust-build    # build Rust extension
make compare-bt    # head-to-head stock-only comparison vs bt
make parity-gate   # bt overlap tolerance gate
```

## Architecture

```
options_portfolio_backtester/
├── core/            # Types: Direction, OptionType, Greeks, Fill, Order
├── data/            # Schema DSL, CSV providers
├── strategy/        # Strategy, StrategyLeg, presets (strangle, iron condor, etc.)
├── execution/       # Pluggable: CostModel, FillModel, Sizer, SignalSelector
├── portfolio/       # Portfolio, OptionPosition, RiskManager, Greeks aggregation
├── engine/          # BacktestEngine, AlgoPipelineBacktester, StrategyTreeEngine
└── analytics/       # BacktestStats, TradeLog, TearsheetReport, charts

rust/
├── ob_core/         # Pure Rust: types, inventory join, balance, filter parser,
│                    #   entry/exit, full backtest loop, stats, cost/fill models
└── ob_python/       # PyO3 bindings, parallel sweep, zero-copy Arrow bridge
```

```
Data → Strategy (legs + filters) → Engine → Execution (cost, fill, sizer, selector)
                                      ↓
                                  Risk Manager → Portfolio → Analytics
```

## Usage

### Options backtest

```python
from options_portfolio_backtester import (
    BacktestEngine, Stock,
    NearestDelta, PerContractCommission,
    RiskManager, MaxDelta, MaxDrawdown,
)
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Type, Direction

options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

strategy = Strategy(schema)
leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
leg.entry_filter = (
    (schema.underlying == "SPY")
    & (schema.dte >= 60) & (schema.dte <= 120)
    & (schema.delta >= -0.25) & (schema.delta <= -0.10)
)
leg.entry_sort = ("delta", False)
leg.exit_filter = schema.dte <= 30
strategy.add_leg(leg)

engine = BacktestEngine(
    allocation={"stocks": 0.97, "options": 0.03, "cash": 0.0},
    initial_capital=1_000_000,
    cost_model=PerContractCommission(rate=0.65),
    signal_selector=NearestDelta(target_delta=-0.20),
    risk_manager=RiskManager([
        MaxDelta(limit=100.0),
        MaxDrawdown(max_dd_pct=0.20),
    ]),
)
engine.stocks = [Stock("SPY", 1.0)]
engine.stocks_data = stocks_data
engine.options_data = options_data
engine.options_strategy = strategy
engine.run(rebalance_freq=1)
```

### Stock portfolio (bt-style pipeline)

```python
from options_portfolio_backtester import (
    AlgoPipelineBacktester,
    RunMonthly, SelectAll, WeighInvVol, LimitWeights, Rebalance,
)
import pandas as pd

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

bt = AlgoPipelineBacktester(
    prices=prices,
    initial_capital=1_000_000,
    algos=[
        RunMonthly(),
        SelectAll(),
        WeighInvVol(lookback=252),
        LimitWeights(limit=0.25),
        Rebalance(),
    ],
)
balance = bt.run()

# Post-hoc analysis on a sub-period
stats = bt.set_date_range(start="2020-01-01", end="2023-12-31")
print(stats.summary())
```

### Strategy tree (capital allocation)

```python
from options_portfolio_backtester import StrategyTreeNode, StrategyTreeEngine

leaf_us = StrategyTreeNode(name="US", weight=2.0, max_share=0.60, engine=us_engine)
leaf_intl = StrategyTreeNode(name="INTL", weight=1.0, engine=intl_engine)
root = StrategyTreeNode(name="Global", children=[leaf_us, leaf_intl])

tree = StrategyTreeEngine(root, initial_capital=1_000_000)
tree.run(rebalance_freq=1)
print(tree.to_dot())  # Graphviz export
```

### Strategy presets

```python
from backtester.strategy import Strangle

strangle = Strangle(schema, "short", "SPY",
                    dte_entry_range=(30, 60), dte_exit=7,
                    otm_pct=5, pct_tolerance=1,
                    exit_thresholds=(0.2, 0.2))
```

Presets: `strangle`, `iron_condor`, `covered_call`, `cash_secured_put`, `collar`, `butterfly`.

## Pipeline Algos

Full bt-compatible composable pipeline. All algos follow the `Algo` protocol: `__call__(ctx: PipelineContext) -> StepDecision`.

### Scheduling

| Algo | Description |
|------|-------------|
| `RunDaily()` | Every trading day |
| `RunWeekly()` | First day of each week |
| `RunMonthly()` | First day of each month |
| `RunQuarterly()` | First day of each quarter |
| `RunYearly()` | First day of each year |
| `RunOnce()` | First date only |
| `RunOnDate(dates)` | Specific dates |
| `RunAfterDate(date)` | After a date (inclusive) |
| `RunAfterDays(n)` | Skip first n trading days (warmup) |
| `RunEveryNPeriods(n)` | Every nth trading day |
| `RunIfOutOfBounds(tolerance)` | When positions drift beyond tolerance |
| `Or(*algos)` | Pass if any child passes |
| `Not(algo)` | Invert child decision |
| `Require(algo)` | Guard: only continue if child passes |

### Selection

| Algo | Description |
|------|-------------|
| `SelectAll()` | All symbols with valid prices |
| `SelectThese(symbols)` | Fixed list |
| `SelectHasData(min_days)` | Minimum history length |
| `SelectN(n)` | First n from current selection |
| `SelectMomentum(n, lookback)` | Top n by trailing return |
| `SelectWhere(fn)` | Custom callable filter |
| `SelectRandomly(n, seed)` | Random sample |
| `SelectActive()` | Filter out zero/NaN prices |
| `SelectRegex(pattern)` | Regex match on symbol name |

### Weighting

| Algo | Description |
|------|-------------|
| `WeighEqually()` | 1/N across selected |
| `WeighSpecified(weights)` | Fixed weights dict |
| `WeighTarget(weights_df)` | Date-indexed weight DataFrame |
| `WeighInvVol(lookback)` | Inverse-volatility (risk parity lite) |
| `WeighMeanVar(lookback)` | Mean-variance optimization (max Sharpe) |
| `WeighERC(lookback)` | Equal risk contribution |
| `TargetVol(target, lookback)` | Scale to target annualized vol |
| `WeighRandomly(seed)` | Random Dirichlet weights |

### Weight Limits, Risk, and Rebalancing

| Algo | Description |
|------|-------------|
| `LimitWeights(limit)` | Cap individual weights, renormalize |
| `LimitDeltas(limit)` | Cap per-period weight changes |
| `ScaleWeights(scale)` | Multiply all weights (leverage/deleverage) |
| `HedgeRisks(target_delta, hedge_symbols)` | Auto-hedge portfolio delta via Jacobian solve |
| `Margin(leverage, interest_rate, maintenance_pct)` | Leveraged simulation with margin calls |
| `MaxDrawdownGuard(max_drawdown_pct)` | Circuit breaker during drawdowns |
| `Rebalance()` | Full rebalance to target weights |
| `RebalanceOverTime(n)` | Gradual rebalance over n periods |
| `CapitalFlow(flows)` | Scheduled cash additions/withdrawals |
| `CloseDead()` | Close zero-price positions |
| `ClosePositionsAfterDates(schedule)` | Close positions on scheduled dates |
| `ReplayTransactions(blotter)` | Replay a pre-recorded trade blotter |
| `CouponPayingPosition(amount, frequency)` | Periodic coupon cash flows (fixed income) |

## Pluggable Components

### Signal Selectors

| Selector | Description |
|----------|-------------|
| `FirstMatch()` | First matching contract (default) |
| `NearestDelta(target)` | Closest delta to target |
| `MaxOpenInterest()` | Highest open interest |

### Risk Manager

```python
from options_portfolio_backtester import RiskManager, MaxDelta, MaxVega, MaxDrawdown

rm = RiskManager([
    MaxDelta(limit=50.0),
    MaxVega(limit=30.0),
    MaxDrawdown(max_dd_pct=0.15),
])
```

### Cost Models

| Model | Description |
|-------|-------------|
| `NoCosts()` | Zero costs (default) |
| `PerContractCommission(rate)` | Fixed per-contract fee |
| `TieredCommission(tiers)` | Volume-based tiered pricing |
| `SpreadSlippage(pct)` | Fraction of bid-ask spread |

### Fill Models

| Model | Description |
|-------|-------------|
| `MarketAtBidAsk()` | Bid for sells, ask for buys (default) |
| `MidPrice()` | Midpoint of bid-ask |
| `VolumeAwareFill(threshold)` | Interpolates based on volume |

### Position Sizers

| Sizer | Description |
|-------|-------------|
| `CapitalBased()` | qty = allocation // cost (default) |
| `FixedQuantity(qty)` | Fixed number of contracts |
| `FixedDollar(amount)` | Target dollar amount |
| `PercentOfPortfolio(pct)` | Percent of total portfolio |

## Rust Performance Core

The optional Rust extension runs the full backtest loop via PyO3/Polars/Rayon. Falls back transparently to Python when not installed.

| Engine | Options Backtest | Stock-Only |
|--------|-----------------|------------|
| **Rust** | **4.2s** | **0.6s** |
| Python | 9.9s | -- |
| bt | -- | 3.7s |

- **2.4x faster** than Python on full options backtest (24.7M rows)
- **6.0x faster** than bt on stock-only monthly rebalancing
- Exact numerical parity with Python path
- Parallel grid sweep: **5-8x faster** via Rayon (bypasses GIL)

### How it's fast

| Optimization | Impact |
|--------------|--------|
| Pre-partition by date | `HashMap<i64, DayOptions>` for O(1) lookups |
| i64 nanosecond keys | Eliminates 21s of pandas strftime overhead |
| PyArrow bridge | pandas -> PyArrow -> Polars (2x faster than direct) |
| Column pruning | Drops 5 unused columns before conversion |
| Skip-sort detection | Samples 8 points to detect pre-sorted data |
| Compiled filter AST | Query strings compiled once, reused across all dates |
| Rayon parallel sweep | Shared-memory, per-config timing, scoped thread pool |

### Building

```shell
make rust-build
# or without nix:
maturin develop --manifest-path rust/ob_python/Cargo.toml --release
```

Data flows: `pandas.DataFrame` -> `pyarrow.Table` -> Arrow C Data Interface -> Rust `Polars DataFrame`. Numeric columns are zero-copy.

The engine automatically dispatches to Rust when available:

```python
from options_portfolio_backtester.engine._dispatch import use_rust, rust

if use_rust():
    result = rust.compute_stats(daily_returns, trade_pnls, risk_free_rate)
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [quickstart](notebooks/quickstart.ipynb) | Getting started: load data, define strategy, run backtest, plot results |
| [paper_comparison](notebooks/paper_comparison.ipynb) | 10 strategies vs academic paper claims with VRP math, crash heatmap, risk/return scatter |
| [findings](notebooks/findings.ipynb) | Full research: allocation sweep, puts vs calls, macro signals, crash-period analysis |
| [volatility_premium](notebooks/volatility_premium.ipynb) | Sell vol vs buy vol deep dive (Carr & Wu 2009, Berman 2014) |
| [strategies](notebooks/strategies.ipynb) | 4-strategy showcase: OTM puts, OTM calls, long straddle, short strangle |
| [trade_analysis](notebooks/trade_analysis.ipynb) | Per-trade P&L: bar charts, cumulative P&L, crash breakdowns |
| [iron_condor](notebooks/iron_condor.ipynb) | 4-leg iron condor income strategy |
| [ivy_portfolio](notebooks/ivy_portfolio.ipynb) | Endowment-style portfolio with straddle hedge overlay |
| [gold_sp500](notebooks/gold_sp500.ipynb) | Multi-asset portfolio with cash/gold proxy + options overlay |
| [benchmark_vs_bt](notebooks/benchmark_vs_bt.ipynb) | Head-to-head vs bt: 6x faster, return parity, equity curves |
| [spitznagel_case](notebooks/spitznagel_case.ipynb) | AQR vs Spitznagel tested with real data. Multi-dimensional parameter sweep. |

## Tail-Risk Hedge Research

**Can a small allocation to SPY puts improve risk-adjusted returns over buy-and-hold?** Inspired by Universa Investments' approach to tail-risk hedging.

### Key findings

**The framing matters more than the strategy.** Same deep OTM puts, opposite conclusions:

| Framing | Setup | Return | Max DD |
|---------|-------|--------|--------|
| AQR (reduce equity) | 99% SPY + 1% puts | +3.15% | -50.8% |
| **Spitznagel (leverage)** | **100% SPY + 1% puts on top** | **+16.46%** | **-45.0%** |
| SPY buy-and-hold | 100% SPY | +11.11% | -51.9% |

Spitznagel's leveraged tail hedge works because crash protection allows full investment and the drawdown reduction is real:
- +0.5% budget: 13.79%/yr, DD -48.1%
- +1.0% budget: 16.46%/yr, DD -45.0%
- +3.3% budget: 28.78%/yr, DD -31.9%

Other findings:
- Selling options (covered calls, put-writing, short strangles) harvests the Variance Risk Premium
- Buying OTM calls adds ~0.8-1.4%/yr in the no-leverage framing
- Macro signals (VIX, Buffett Indicator, Tobin's Q) don't improve put timing

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_entries_exits.py` | Per-trade analysis with signal overlay |
| `scripts/run_spy_otm_puts.py` | 6 hedge variants vs SPY buy-and-hold |
| `scripts/parallel_sweep.py` | Parallel grid sweep across all CPU cores |
| `scripts/sweep_otm.py` | Delta band sweep: near-ATM to deep OTM |
| `scripts/sweep_allocation.py` | Stock/options allocation splits |
| `scripts/sweep_comprehensive.py` | Puts, calls, strangles with macro signal filters |
| `scripts/sweep_volatility.py` | Long vol vs short vol across allocations |

## Data

Data is hosted on [GitHub Releases](https://github.com/lambdaclass/options_backtester/releases/tag/data-v1) and downloaded on demand by `data/fetch_data.py`. Available symbols: SPY, IWM, QQQ (options + underlying, 2008-2025).

Fallback sources: [philippdubach/options-data](https://github.com/philippdubach/options-data), [philippdubach/options-dataset-hist](https://github.com/philippdubach/options-dataset-hist), yfinance.

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
