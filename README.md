Options Backtester
==================

Backtester for evaluating options strategies over historical data. Includes tools for strategy sweeps, tail-risk hedge analysis, and signal-based timing research.

**v0.3** — Modular pluggable framework with optional Rust performance core (10-50x speedup on hot paths).

- [Setup](#setup)
- [Architecture](#architecture)
- [Usage](#usage)
- [Pluggable Components](#pluggable-components)
- [Rust Performance Core](#rust-performance-core)
- [Notebooks](#notebooks)
- [Tail-Risk Hedge Research](#tail-risk-hedge-research)
- [Data](#data)
- [Recommended Reading](#recommended-reading)

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
make rust-bench    # Rust criterion benchmarks
make bench         # Python benchmark suite
make compare-bt    # head-to-head stock-only comparison vs bt
make benchmark-matrix   # matrix benchmark across ranges/rebalance configs
make walk-forward-report # walk-forward IS/OOS report
make parity-gate        # bt overlap tolerance gate (bench test)
```

`make` always runs via `nix develop`.
Default pytest runs exclude the `bench` marker; run `make test-bench` for parity/fuzz benchmarks.

New bt-style extensions:
- Engine algo adapters (`EngineRunMonthly`, `BudgetPercent`, `SelectByDelta`, `SelectByDTE`, `IVRankFilter`, `MaxGreekExposure`, `ExitOnThreshold`)
- Structured engine event log via `BacktestEngine.events_dataframe()`
- Tearsheet exports: `to_csv()`, `to_markdown()`, `to_html()`
- Strategy-tree throttling with per-leaf `max_share` and `unallocated_cash` tracking

## Architecture

```
options_backtester/
├── core/            # Types: Direction, OptionType, Greeks, Fill, Order
├── data/            # Schema DSL, CSV providers
├── strategy/        # Strategy, StrategyLeg, presets (strangle, iron condor, etc.)
├── execution/       # Pluggable: CostModel, FillModel, Sizer, SignalSelector
├── portfolio/       # Portfolio, OptionPosition, RiskManager, Greeks aggregation
├── engine/          # BacktestEngine orchestrator, dispatch layer
└── analytics/       # BacktestStats, TradeLog

rust/
├── ob_core/         # Pure Rust lib: types, inventory join, balance, filter parser,
│                    #   entry/exit computation, full backtest loop, stats
└── ob_python/       # PyO3 cdylib bindings, parallel sweep, zero-copy Arrow bridge
```

The engine composes all components:

```
Data → Strategy (legs + filters) → Engine → Execution (cost, fill, sizer, selector)
                                      ↓
                                  Risk Manager → Portfolio → Analytics
```

## Usage

### New framework (recommended)

```python
from options_backtester import (
    BacktestEngine, Stock,
    NearestDelta, PerContractCommission,
    RiskManager, MaxDelta, MaxDrawdown,
)
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Type, Direction

# Load data
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

# Create strategy
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

# Create engine with pluggable components
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

# Results
engine.trade_log   # DataFrame of executed trades
engine.balance     # Daily portfolio balance with returns
```

### Legacy API

The original `Backtest` class is still fully supported:

```python
from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg

bt = Backtest({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000)
bt.options_budget = lambda date, total_capital: total_capital * 0.001
bt.stocks = [Stock("SPY", 1.0)]
bt.stocks_data = TiingoData("data/processed/stocks.csv")
bt.options_data = HistoricalOptionsData("data/processed/options.csv")
bt.options_strategy = strategy
bt.run(rebalance_freq=1)
```

### Strategy presets

```python
from backtester.strategy import Strangle

# Short strangle: sell OTM call + put
strangle = Strangle(schema, "short", "SPY",
                    dte_entry_range=(30, 60), dte_exit=7,
                    otm_pct=5, pct_tolerance=1,
                    exit_thresholds=(0.2, 0.2))
```

Presets: `strangle`, `iron_condor`, `covered_call`, `cash_secured_put`, `collar`, `butterfly`.

## Pluggable Components

All components are set at engine construction time. Defaults are used when not specified.

### Signal Selectors

Choose which contract to trade from filtered candidates:

| Selector | Description | Default |
|----------|-------------|---------|
| `FirstMatch()` | Picks first row (original behavior) | Yes |
| `NearestDelta(target=-0.30)` | Closest delta to target | |
| `MaxOpenInterest()` | Highest open interest (liquidity) | |

The selector is wired into the engine — it receives enriched candidate data including any extra columns it needs (delta, openinterest) from the raw options data.

### Risk Manager

Pre-trade risk checks that can block entries:

| Constraint | Description | Default |
|------------|-------------|---------|
| `MaxDelta(limit=100)` | Blocks if portfolio delta would exceed limit | |
| `MaxVega(limit=50)` | Blocks if portfolio vega would exceed limit | |
| `MaxDrawdown(max_dd_pct=0.20)` | Blocks new entries during drawdowns | |

The risk manager computes portfolio Greeks from current inventory positions and proposed entry Greeks, then checks all constraints before allowing a trade.

```python
from options_backtester import RiskManager, MaxDelta, MaxDrawdown

rm = RiskManager([
    MaxDelta(limit=50.0),
    MaxDrawdown(max_dd_pct=0.15),
])
```

### Cost Models

| Model | Description |
|-------|-------------|
| `NoCosts()` | Zero transaction costs (default) |
| `PerContractCommission(rate=0.65)` | Fixed per-contract fee |
| `TieredCommission(tiers)` | Volume-based tiered pricing |
| `SpreadSlippage(pct=0.5)` | Fraction of bid-ask spread |

### Fill Models

| Model | Description |
|-------|-------------|
| `MarketAtBidAsk()` | Bid for sells, ask for buys (default) |
| `MidPrice()` | Midpoint of bid-ask |
| `VolumeAwareFill(threshold=100)` | Interpolates based on volume |

### Position Sizers

| Sizer | Description |
|-------|-------------|
| `CapitalBased()` | qty = allocation // cost (default) |
| `FixedQuantity(qty=1)` | Always trade fixed number |
| `FixedDollar(amount=10000)` | Target fixed dollar amount |
| `PercentOfPortfolio(pct=0.01)` | Percent of total portfolio |

## Rust Performance Core

The optional Rust extension provides 10-50x speedup on hot paths via PyO3/Polars/Rayon. Falls back transparently to Python when not installed.

### What's accelerated

| Hot Path | Python | Rust | Expected Speedup |
|----------|--------|------|-----------------|
| Inventory→options join | `pd.merge` | Polars hash-join | 10-50x |
| Balance groupby+pivot | `groupby().sum()` | Polars groupby + Arrow | 3-8x |
| Filter evaluation | `data.eval(query)` per leg per day | Compiled AST → Polars predicates | 3-8x |
| Entry signal generation | sort + reindex + MultiIndex | Polars lazy plan: anti-join → filter → sort | 5-15x |
| Exit mask + thresholds | concat + fillna + reduce(OR) | Arrow SIMD boolean ops | 2-3x |
| Stats (Sharpe/Sortino) | numpy loops | Vectorized Rust | 2-5x |
| Grid sweep | ProcessPoolExecutor (pickle) | Rayon scoped pool (shared memory, per-config `elapsed_ms` timing) | N_cores |

### Building

```shell
# With nix:
make rust-build

# Without nix:
maturin develop --manifest-path rust/ob_python/Cargo.toml --release
```

### Zero-copy data bridge

Data flows: `pandas.DataFrame` → `pyarrow.Table` → Arrow C Data Interface → Rust `Polars DataFrame`. Return path is the reverse. Numeric columns (float64/int64) are zero-copy. String columns require one copy to Arrow UTF-8 format.

### Filter compilation

The Rust filter parser compiles pandas-eval query strings (generated by the Schema DSL) into an AST, then evaluates against Polars DataFrames:

```
"(type == 'put') & (ask > 0)"           → And(Eq("type", "put"), Gt("ask", 0))
"(underlying == 'SPX') & (dte >= 60)"   → And(Eq("underlying", "SPX"), Ge("dte", 60))
"strike >= underlying_last * 1.02"      → ColArith("underlying_last", Mul, 1.02, Le, Col("strike"))
```

Compiled once per strategy setup, reused across all dates.

### Dispatch layer

```python
from options_backtester.engine._dispatch import use_rust, rust

if use_rust():
    result = rust.compute_stats(daily_returns, trade_pnls, risk_free_rate)
```

The engine automatically dispatches to Rust when available — zero API change for users.

## Notebooks

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

See also [REFERENCES.md](REFERENCES.md) for 20 academic papers on options overlay strategies.

## Tail-Risk Hedge Research

The main research question: **can a small allocation to SPY puts improve risk-adjusted returns over buy-and-hold?** Inspired by Universa Investments' approach to tail-risk hedging.

### Scripts

`scripts/backtest_runner.py` provides shared helpers (data loading, backtest execution, result formatting, charting) used by all sweep scripts.

| Script | Purpose |
|--------|---------|
| `scripts/analyze_entries_exits.py` | **Per-trade analysis** with signal overlay — the main research tool |
| `scripts/run_spy_otm_puts.py` | Strategy sweep: 6 hedge variants (delta, DTE, budget, profit caps) vs SPY buy-and-hold |
| `scripts/parallel_sweep.py` | Parallel grid sweep using multiprocessing across all CPU cores |
| `scripts/sweep_otm.py` | Delta band sweep: near-ATM to deep OTM levels |
| `scripts/sweep_allocation.py` | Allocation sweep: different stock/options splits (0-100%) |
| `scripts/sweep_leverage.py` | Leverage sweep: options budgets on top of 100% stock allocation |
| `scripts/sweep_comprehensive.py` | Puts, calls, strangles with macro signal filters (VIX, Buffett, Tobin Q) |
| `scripts/sweep_iv_signal.py` | IV-signal-filtered budget: buy puts based on IV percentile vs rolling median |
| `scripts/sweep_volatility.py` | Long vol (straddles) vs short vol (strangles) across allocations |
| `scripts/sweep_beat_spy.py` | Tail-risk configs with tiny budgets and low-frequency rebalancing |
| `scripts/sweep_beat_spy2.py` | Second iteration: extreme premium reduction with semi-annual/annual rebalancing |
| `scripts/sweep_beat_spy3.py` | Diagnoses rebalancing drag, tries hedge configs that beat pure-stock baseline |

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

All strategy parameters in `scripts/analyze_entries_exits.py` are configurable via the `CONFIG` dict:

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
