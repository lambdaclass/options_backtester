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
- **Risk management**: pre-trade gating (`MaxDelta`, `MaxVega`, `MaxDrawdown`), notional cap (`max_notional_pct`), margin simulation, circuit breakers, auto-hedging
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

options_data = HistoricalOptionsData("options.csv")
stocks_data = TiingoData("stocks.csv")
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

### Run tests

```shell
make test            # all tests
make test-bench      # benchmark/property tests (explicit opt-in)
make test-regression # regression snapshot tests (locked golden values)
make test-chaos      # chaos/fault-injection tests (corrupted data)
make muttest         # mutation testing on high-value modules
make muttest-results # show mutation testing results
make lint            # ruff linter
make typecheck       # mypy type checking
make rust-test       # Rust unit tests
make rust-build      # build Rust extension
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
| **Risk constraints** | `MaxDelta(limit)`, `MaxVega(limit)`, `MaxDrawdown(max_dd_pct)`, `max_notional_pct` (engine param) |

## Rust Core

Optional. Falls back transparently to Python when not installed.

**4.2s** on a full options backtest (24.7M rows), **0.6s** stock-only, **5-8x faster** on parallel sweeps (Rayon bypasses the GIL). Pure Python fallback included — exact numerical parity.

The data bridge is zero-copy where possible: `pandas.DataFrame` -> `pyarrow.Table` -> Arrow C Data Interface -> Rust `Polars DataFrame`.

Key optimizations: pre-partition by date into `HashMap<i64, DayOptions>` for O(1) lookups, i64 nanosecond keys (eliminates pandas strftime overhead), compiled filter AST reused across all dates, column pruning before conversion, skip-sort detection via 8-point sampling.

```shell
make rust-build
# or: maturin develop --manifest-path rust/ob_python/Cargo.toml --release
```

## Data

Pre-built SPY data (stocks + options) is available as a [GitHub Release](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1). Download `stocks.csv` and `options.csv` from the release assets and place them wherever you like:

```python
options_data = HistoricalOptionsData("path/to/options.csv")
stocks_data = TiingoData("path/to/stocks.csv")
```

You can also bring your own CSVs. Expected schemas:

- **Stocks**: columns `date`, `symbol`, `adjClose` (and optionally `close`, `high`, `low`, `volume`)
- **Options**: columns `quotedate`, `underlying`, `type`, `strike`, `expiration`, `dte`, `bid`, `ask`, `volume`, `openinterest`, `delta` (and optionally `gamma`, `theta`, `vega`, `impliedvol`)

For CME futures and options data, see the Databento setup in [finance_research](https://github.com/unbalancedparentheses/finance_research).

## Research

Research notebooks, scripts, and data live in a separate repo: [finance_research](https://github.com/unbalancedparentheses/finance_research).

## Documentation

TODO: write proper docs from scratch. The old `docs/` directory contained stale notebook exports with broken paths and embedded JS blobs. It was deleted. Docs to write:

- Getting started guide with a working end-to-end example (including where to get data)
- Strategy cookbook: strangle, iron condor, tail hedge, covered call
- Execution model reference: cost models, fill models, sizers, signal selectors
- Pipeline algo reference with examples
- Rust extension: build, deploy, benchmarks, troubleshooting
- Multi-strategy and walk-forward optimization guide

