Options Portfolio Backtester
============================

Backtest options strategies with realistic execution, Greeks-aware risk management, and contract-level inventory. Also handles equities and multi-asset portfolios. Optional Rust core for speed.

## Get started

### Install

With Nix:
```shell
nix develop
```

Without Nix (Python >= 3.12):
```shell
python -m venv .venv && source .venv/bin/activate
make install-dev
```

### Get data

```shell
python data/fetch_data.py all --symbols SPY
```

Downloads SPY stock prices and options chains to `data/processed/`. Supports 104+ symbols. See [`data/README.md`](data/README.md) for details.

### Run your first backtest

```python
from options_portfolio_backtester import (
    BacktestEngine, Stock, Type, Direction,
    HistoricalOptionsData, TiingoData,
    Strategy, StrategyLeg,
    NearestDelta, PerContractCommission,
    RiskManager, MaxDelta, MaxDrawdown,
)

# Load data
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema

# Define strategy: buy OTM puts on SPY, exit when DTE drops below 30
strategy = Strategy(schema)
leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
leg.entry_filter = (
    (schema.underlying == "SPY")
    & (schema.dte >= 60) & (schema.dte <= 120)
    & (schema.delta >= -0.25) & (schema.delta <= -0.10)
)
leg.exit_filter = schema.dte <= 30
strategy.add_leg(leg)

# Run backtest: 97% stocks, 3% options
engine = BacktestEngine(
    allocation={"stocks": 0.97, "options": 0.03, "cash": 0.0},
    initial_capital=1_000_000,
    cost_model=PerContractCommission(rate=0.65),
    signal_selector=NearestDelta(target_delta=-0.20),
    risk_manager=RiskManager([MaxDelta(100.0), MaxDrawdown(0.20)]),
)
engine.stocks = [Stock("SPY", 1.0)]
engine.stocks_data = stocks_data
engine.options_data = options_data
engine.options_strategy = strategy
engine.run(rebalance_freq=1)

# Results
print(engine.balance["total capital"].iloc[-1])  # final capital
print(len(engine.trade_log))                      # number of trades
```

### Strategy presets

Instead of building legs manually:

```python
from options_portfolio_backtester import Strangle

strangle = Strangle(schema, "short", "SPY",
                    dte_entry_range=(30, 60), dte_exit=7,
                    otm_pct=5, pct_tolerance=1,
                    exit_thresholds=(0.2, 0.2))
```

Available presets: `Strangle`, `IronCondor`, `CoveredCall`, `CashSecuredPut`, `Collar`, `Butterfly`.

### Stock-only backtest with algo pipeline

For equity portfolios without options, use the pipeline API:

```python
from options_portfolio_backtester.engine.pipeline import (
    AlgoPipelineBacktester,
    RunMonthly, SelectAll, WeighInvVol, LimitWeights, Rebalance,
)
import pandas as pd

prices = pd.read_csv("data/processed/stocks.csv", parse_dates=["date"])
prices = prices.pivot(index="date", columns="symbol", values="adjClose")

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
```

## Execution models

Every component is swappable. Pass them to `BacktestEngine(...)` or override per-leg.

**Signal selectors** — which contract to pick from candidates:
`FirstMatch()`, `NearestDelta(target)`, `MaxOpenInterest()`

**Cost models** — commissions and fees:
`NoCosts()`, `PerContractCommission(rate)`, `TieredCommission(tiers)`, `SpreadSlippage(pct)`

**Fill models** — execution price:
`MarketAtBidAsk()`, `MidPrice()`, `VolumeAwareFill(threshold)`

**Position sizers** — how many contracts:
`CapitalBased()`, `FixedQuantity(qty)`, `FixedDollar(amount)`, `PercentOfPortfolio(pct)`

**Risk constraints** — pre-trade gating:
`MaxDelta(limit)`, `MaxVega(limit)`, `MaxDrawdown(max_dd_pct)`

## Rust acceleration

Optional. Falls back to Python when not installed.

```shell
make rust-build
```

| Benchmark | Python | Rust |
|-----------|--------|------|
| Full options backtest (24.7M rows) | 10.0s | **4.2s** |
| Stock-only monthly rebalance | 3.7s | **0.6s** |
| Parallel grid sweep (100 configs) | — | **5-8x** faster (Rayon, bypasses GIL) |

## Data

```shell
# SPY stock + options data
python data/fetch_data.py all --symbols SPY

# Multiple symbols
python data/fetch_data.py all --symbols SPY IWM QQQ --start 2020-01-01 --end 2023-01-01

# FRED macro signals (VIX, GDP, Buffett Indicator, etc.)
python data/fetch_signals.py

# Convert OptionsDX format
python data/convert_optionsdx.py data/raw/spx_eod_2020.csv --output data/processed/spx_options.csv
```

You can also bring your own CSVs. Required columns:
- **Stocks**: `date`, `symbol`, `adjClose`
- **Options**: `quotedate`, `underlying`, `type`, `strike`, `expiration`, `dte`, `bid`, `ask`, `volume`, `openinterest`, `delta`

## Tests

```shell
make test            # all tests (1300+)
make test-regression # regression snapshots (locked golden values)
make test-chaos      # fault injection (corrupted/adversarial data)
make muttest         # mutation testing on core modules
make lint            # ruff
make typecheck       # mypy
make rust-test       # Rust unit tests
```

## Architecture

```
options_portfolio_backtester/
├── core/            # Types: Direction, OptionType, Greeks, Fill, Order
├── data/            # Schema DSL, CSV providers
├── strategy/        # Strategy, StrategyLeg, presets
├── execution/       # CostModel, FillModel, Sizer, SignalSelector
├── portfolio/       # Portfolio, OptionPosition, RiskManager
├── engine/          # BacktestEngine, AlgoPipelineBacktester, StrategyTreeEngine
└── analytics/       # BacktestStats, TradeLog, TearsheetReport, charts

rust/
├── ob_core/         # Backtest loop, stats, execution models, filter parser
└── ob_python/       # PyO3 bindings, parallel sweep, Arrow bridge
```

## Pipeline algos

40+ composable algos for the `AlgoPipelineBacktester`. All follow `__call__(ctx) -> StepDecision`.

**Scheduling**: `RunDaily`, `RunWeekly`, `RunMonthly`, `RunQuarterly`, `RunYearly`, `RunOnce`, `RunOnDate`, `RunAfterDate`, `RunAfterDays`, `RunEveryNPeriods`, `RunIfOutOfBounds`, `Or`, `Not`, `Require`

**Selection**: `SelectAll`, `SelectThese`, `SelectHasData`, `SelectN`, `SelectMomentum`, `SelectWhere`, `SelectRandomly`, `SelectActive`, `SelectRegex`

**Weighting**: `WeighEqually`, `WeighSpecified`, `WeighTarget`, `WeighInvVol`, `WeighMeanVar`, `WeighERC`, `TargetVol`, `WeighRandomly`

**Risk & rebalancing**: `LimitWeights`, `LimitDeltas`, `ScaleWeights`, `HedgeRisks`, `Margin`, `MaxDrawdownGuard`, `Rebalance`, `RebalanceOverTime`, `CapitalFlow`, `CloseDead`, `ClosePositionsAfterDates`, `ReplayTransactions`, `CouponPayingPosition`

## Research

Research notebooks and CME/Databento data: [finance_research](https://github.com/unbalancedparentheses/finance_research).
