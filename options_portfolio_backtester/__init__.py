"""options_portfolio_backtester — the open-source options backtesting framework."""

# Core types
from options_portfolio_backtester.core.types import (
    Direction,
    OptionType,
    Type,
    Order,
    Signal,
    Fill,
    Greeks,
    OptionContract,
    StockAllocation,
    Stock,
    get_order,
)

# Data
from options_portfolio_backtester.data.schema import Schema, Field, Filter
from options_portfolio_backtester.data.providers import (
    CsvOptionsProvider, CsvStocksProvider,
    TiingoData, HistoricalOptionsData,
)

# Strategy
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.strategy.presets import Strangle

# Execution
from options_portfolio_backtester.execution.cost_model import (
    NoCosts, PerContractCommission, TieredCommission, SpreadSlippage,
)
from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk, MidPrice, VolumeAwareFill
from options_portfolio_backtester.execution.sizer import (
    CapitalBased, FixedQuantity, FixedDollar, PercentOfPortfolio,
)
from options_portfolio_backtester.execution.signal_selector import (
    FirstMatch, NearestDelta, MaxOpenInterest,
)

# Portfolio
from options_portfolio_backtester.portfolio.portfolio import Portfolio
from options_portfolio_backtester.portfolio.position import OptionPosition
from options_portfolio_backtester.portfolio.greeks import aggregate_greeks
from options_portfolio_backtester.portfolio.risk import RiskManager, MaxDelta, MaxVega, MaxDrawdown

# Engine
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.engine.clock import TradingClock
from options_portfolio_backtester.engine.pipeline import (
    AlgoPipelineBacktester, PipelineContext, PipelineLogRow, StepDecision,
    # Scheduling
    RunMonthly, RunWeekly, RunQuarterly, RunYearly, RunDaily,
    RunOnce, RunOnDate, RunAfterDate, RunAfterDays, RunEveryNPeriods,
    RunIfOutOfBounds,
    Or, Not,
    # Selection
    SelectThese, SelectAll, SelectHasData, SelectMomentum, SelectN, SelectWhere,
    SelectRandomly, SelectActive, SelectRegex,
    # Weighting
    WeighSpecified, WeighEqually, WeighInvVol, WeighMeanVar, WeighERC, TargetVol,
    WeighRandomly, WeighTarget,
    # Weight limits
    LimitWeights, LimitDeltas, ScaleWeights,
    # Capital flows
    CapitalFlow,
    # Risk
    MaxDrawdownGuard, HedgeRisks, Margin,
    # Fixed income
    CouponPayingPosition,
    # Position management
    CloseDead, ClosePositionsAfterDates, Require, ReplayTransactions,
    # Rebalancing
    Rebalance, RebalanceOverTime,
    # Random benchmarking
    RandomBenchmarkResult, benchmark_random,
)
from options_portfolio_backtester.engine.algo_adapters import (
    EngineAlgo,
    EngineStepDecision,
    EnginePipelineContext,
    EngineRunMonthly,
    BudgetPercent,
    RangeFilter,
    SelectByDelta,
    SelectByDTE,
    IVRankFilter,
    MaxGreekExposure,
    ExitOnThreshold,
)
from options_portfolio_backtester.engine.strategy_tree import StrategyTreeNode, StrategyTreeEngine

# Analytics
from options_portfolio_backtester.analytics.stats import BacktestStats, PeriodStats, LookbackReturns
from options_portfolio_backtester.analytics.trade_log import TradeLog
from options_portfolio_backtester.analytics.tearsheet import TearsheetReport, build_tearsheet
from options_portfolio_backtester.analytics.summary import summary

__all__ = [
    # Core types
    "Direction", "OptionType", "Type", "Order", "Signal", "Fill", "Greeks",
    "OptionContract", "StockAllocation", "Stock", "get_order",
    # Data
    "Schema", "Field", "Filter", "CsvOptionsProvider", "CsvStocksProvider",
    "TiingoData", "HistoricalOptionsData",
    # Strategy
    "Strategy", "StrategyLeg", "Strangle",
    # Execution
    "NoCosts", "PerContractCommission", "TieredCommission", "SpreadSlippage",
    "MarketAtBidAsk", "MidPrice", "VolumeAwareFill",
    "CapitalBased", "FixedQuantity", "FixedDollar", "PercentOfPortfolio",
    "FirstMatch", "NearestDelta", "MaxOpenInterest",
    # Portfolio
    "Portfolio", "OptionPosition", "aggregate_greeks",
    "RiskManager", "MaxDelta", "MaxVega", "MaxDrawdown",
    # Engine
    "BacktestEngine", "TradingClock",
    "AlgoPipelineBacktester", "PipelineContext", "PipelineLogRow", "StepDecision",
    "RunMonthly", "RunWeekly", "RunQuarterly", "RunYearly", "RunDaily",
    "RunOnce", "RunOnDate", "RunAfterDate", "RunAfterDays", "RunEveryNPeriods",
    "RunIfOutOfBounds",
    "Or", "Not",
    "SelectThese", "SelectAll", "SelectHasData", "SelectMomentum", "SelectN", "SelectWhere",
    "SelectRandomly", "SelectActive", "SelectRegex",
    "WeighSpecified", "WeighEqually", "WeighInvVol", "WeighMeanVar", "WeighERC", "TargetVol",
    "WeighRandomly", "WeighTarget",
    "LimitWeights", "LimitDeltas", "ScaleWeights",
    "CapitalFlow",
    "MaxDrawdownGuard", "HedgeRisks", "Margin",
    "CouponPayingPosition",
    "CloseDead", "ClosePositionsAfterDates", "Require", "ReplayTransactions",
    "Rebalance", "RebalanceOverTime",
    "RandomBenchmarkResult", "benchmark_random",
    "EngineAlgo", "EngineStepDecision", "EnginePipelineContext", "EngineRunMonthly",
    "BudgetPercent", "RangeFilter", "SelectByDelta", "SelectByDTE", "IVRankFilter",
    "MaxGreekExposure", "ExitOnThreshold",
    "StrategyTreeNode", "StrategyTreeEngine",
    # Analytics
    "BacktestStats", "PeriodStats", "LookbackReturns",
    "TradeLog", "TearsheetReport", "build_tearsheet",
    "summary",
]
