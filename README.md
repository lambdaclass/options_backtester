Options Portfolio Backtester
============================

Most backtesting tools only handle equities. If you want to test options strategies — strangles, iron condors, tail-risk hedges — with realistic execution, Greeks-aware risk management, and contract-level inventory, you've had to build it yourself.

This is the open-source framework that does all of that, plus everything equity-only tools do, with an optional Rust core that processes 24.7M options rows in 4 seconds.

## At a glance

- **Options + equities + multi-asset** in one framework
- **Multi-leg strategies**: strangle, iron condor, butterfly, collar, covered call, cash-secured put
- **Per-position Greeks**: delta, gamma, theta, vega with portfolio-level aggregation
- **40+ composable pipeline algos**: scheduling, selection, weighting, rebalancing, risk guards
- **Realistic execution**: cost models, fill models, signal selectors, position sizers — all pluggable, all with per-leg overrides
- **Risk management**: pre-trade gating (`MaxDelta`, `MaxVega`, `MaxDrawdown`), margin simulation, circuit breakers, auto-hedging
- **Strategy tree**: hierarchical capital allocation with budget caps and Graphviz export
- **Rust acceleration**: optional PyO3/Polars/Rayon backend, transparent fallback to Python, parallel parameter sweeps that bypass the GIL
- **Walk-forward optimization**: in-sample/out-of-sample with parallel grid sweep
- **Rich analytics**: Sharpe, Sortino, Calmar, profit factor, tail ratio, win rate, monthly heatmap, weights chart, tearsheet export, round-trip trade P&L

| Benchmark | Time |
|-----------|------|
| Stock-only monthly rebalance | **0.6s** |
| Full options backtest (24.7M rows) | **4.2s** |
| Parallel grid sweep (100 configs) | **5-8x** faster via Rayon |

## Quickstart

### Options backtest

```python
from options_portfolio_backtester import (
    BacktestEngine, Stock, Type, Direction,
    HistoricalOptionsData, TiingoData,
    Strategy, StrategyLeg,
    NearestDelta, PerContractCommission,
    RiskManager, MaxDelta, MaxDrawdown,
)

options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

# Buy OTM puts on SPY, exit when DTE drops below 30
strategy = Strategy(schema)
leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
leg.entry_filter = (
    (schema.underlying == "SPY")
    & (schema.dte >= 60) & (schema.dte <= 120)
    & (schema.delta >= -0.25) & (schema.delta <= -0.10)
)
leg.exit_filter = schema.dte <= 30
strategy.add_leg(leg)

engine = BacktestEngine(
    allocation={"stocks": 0.97, "options": 0.03, "cash": 0.0},
    initial_capital=1_000_000,
    cost_model=PerContractCommission(rate=0.65),
    signal_selector=NearestDelta(target_delta=-0.20),
    risk_manager=RiskManager([MaxDelta(limit=100.0), MaxDrawdown(max_dd_pct=0.20)]),
)
engine.stocks = [Stock("SPY", 1.0)]
engine.stocks_data = stocks_data
engine.options_data = options_data
engine.options_strategy = strategy
engine.run(rebalance_freq=1)
```

### Stock portfolio with algo pipeline

```python
from options_portfolio_backtester.engine.pipeline import (
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
bt.run()

# Post-hoc analysis on a sub-period
stats = bt.set_date_range(start="2020-01-01", end="2023-12-31")
print(stats.summary())
```

### Strategy presets

```python
from options_portfolio_backtester import Strangle

strangle = Strangle(schema, "short", "SPY",
                    dte_entry_range=(30, 60), dte_exit=7,
                    otm_pct=5, pct_tolerance=1,
                    exit_thresholds=(0.2, 0.2))
```

Presets: `strangle`, `iron_condor`, `covered_call`, `cash_secured_put`, `collar`, `butterfly`.

## Tail-Risk Hedge Research

**Can a small allocation to SPY puts improve risk-adjusted returns over buy-and-hold?** Inspired by Universa Investments' approach to tail-risk hedging.

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

See the [notebooks](#notebooks) and [scripts](#scripts) for the full analysis.

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
make test          # all tests
make test-bench    # benchmark/property tests (explicit opt-in)
make lint          # ruff linter
make typecheck     # mypy type checking
make rust-test     # Rust unit tests
make rust-build    # build Rust extension
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

## Pipeline Algos

40+ composable pipeline algos. All follow the `Algo` protocol: `__call__(ctx: PipelineContext) -> StepDecision`.

<details>
<summary><strong>Scheduling</strong> (14 algos)</summary>

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

</details>

<details>
<summary><strong>Selection</strong> (9 algos)</summary>

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

</details>

<details>
<summary><strong>Weighting</strong> (8 algos)</summary>

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

</details>

<details>
<summary><strong>Weight limits, risk, and rebalancing</strong> (13 algos)</summary>

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

</details>

## Pluggable Execution

Every component is swappable. Mix and match per strategy or per leg.

| Component | Options |
|-----------|---------|
| **Signal selectors** | `FirstMatch()`, `NearestDelta(target)`, `MaxOpenInterest()` |
| **Cost models** | `NoCosts()`, `PerContractCommission(rate)`, `TieredCommission(tiers)`, `SpreadSlippage(pct)` |
| **Fill models** | `MarketAtBidAsk()`, `MidPrice()`, `VolumeAwareFill(threshold)` |
| **Position sizers** | `CapitalBased()`, `FixedQuantity(qty)`, `FixedDollar(amount)`, `PercentOfPortfolio(pct)` |
| **Risk constraints** | `MaxDelta(limit)`, `MaxVega(limit)`, `MaxDrawdown(max_dd_pct)` |

## Rust Core

Optional. Falls back transparently to Python when not installed.

**4.2s** on a full options backtest (24.7M rows), **0.6s** stock-only, **5-8x faster** on parallel sweeps (Rayon bypasses the GIL). Pure Python fallback included — exact numerical parity.

The data bridge is zero-copy where possible: `pandas.DataFrame` -> `pyarrow.Table` -> Arrow C Data Interface -> Rust `Polars DataFrame`.

Key optimizations: pre-partition by date into `HashMap<i64, DayOptions>` for O(1) lookups, i64 nanosecond keys (eliminates pandas strftime overhead), compiled filter AST reused across all dates, column pruning before conversion, skip-sort detection via 8-point sampling.

```shell
make rust-build
# or: maturin develop --manifest-path rust/ob_python/Cargo.toml --release
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [quickstart](notebooks/quickstart.ipynb) | Load data, define strategy, run backtest, plot results |
| [paper_comparison](notebooks/paper_comparison.ipynb) | 10 strategies vs academic paper claims, VRP math, crash heatmap |
| [findings](notebooks/findings.ipynb) | Allocation sweep, puts vs calls, macro signals, crash-period analysis |
| [volatility_premium](notebooks/volatility_premium.ipynb) | Sell vol vs buy vol (Carr & Wu 2009, Berman 2014) |
| [strategies](notebooks/strategies.ipynb) | OTM puts, OTM calls, long straddle, short strangle |
| [trade_analysis](notebooks/trade_analysis.ipynb) | Per-trade P&L, cumulative P&L, crash breakdowns |
| [iron_condor](notebooks/iron_condor.ipynb) | 4-leg iron condor income strategy |
| [ivy_portfolio](notebooks/ivy_portfolio.ipynb) | Endowment-style portfolio with straddle hedge overlay |
| [gold_sp500](notebooks/gold_sp500.ipynb) | Multi-asset portfolio with gold proxy + options overlay |
| [comparison_with_bt](notebooks/comparison_with_bt.ipynb) | Side-by-side comparison with bt: runtime, return parity, feature gap |
| [spitznagel_case](notebooks/spitznagel_case.ipynb) | AQR vs Spitznagel with real data, multi-dimensional parameter sweep |

## Scripts

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
- [Variance Risk Premia - Carr & Wu, 2009](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=577222)
- [Portfolio Selection - Markowitz, 1952](https://www.jstor.org/stable/2975974)
- [The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market - Thorp, 2006](https://www.edwardothorp.com/wp-content/uploads/2016/11/TheKellyCriterionAndTheStockMarket.pdf)
- [Tail Risk Hedging: Creating Robust Portfolios for Volatile Markets - Bhansali, 2014](https://www.amazon.com/Tail-Risk-Hedging-Creating-Portfolios/dp/0071791760)

### Backtesting methodology

- [The Backtest Overfitting Problem - Bailey et al., 2017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [Pseudo-Mathematics and Financial Charlatanism - Bailey et al., 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659)
- [Advances in Financial Machine Learning - de Prado, 2018](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) (chapters on backtesting, cross-validation, bet sizing)
