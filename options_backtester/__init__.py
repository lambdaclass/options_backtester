"""options_backtester — the open-source options backtesting framework."""

# Core types
from options_backtester.core.types import (
    Direction,
    OptionType,
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
from options_backtester.data.schema import Schema, Field, Filter
from options_backtester.data.providers import CsvOptionsProvider, CsvStocksProvider

# Strategy
from options_backtester.strategy.strategy import Strategy
from options_backtester.strategy.strategy_leg import StrategyLeg

# Execution
from options_backtester.execution.cost_model import (
    NoCosts, PerContractCommission, TieredCommission, SpreadSlippage,
)
from options_backtester.execution.fill_model import MarketAtBidAsk, MidPrice, VolumeAwareFill
from options_backtester.execution.sizer import (
    CapitalBased, FixedQuantity, FixedDollar, PercentOfPortfolio,
)
from options_backtester.execution.signal_selector import (
    FirstMatch, NearestDelta, MaxOpenInterest,
)

# Portfolio
from options_backtester.portfolio.portfolio import Portfolio
from options_backtester.portfolio.position import OptionPosition
from options_backtester.portfolio.greeks import aggregate_greeks
from options_backtester.portfolio.risk import RiskManager, MaxDelta, MaxVega, MaxDrawdown

# Engine
from options_backtester.engine.engine import BacktestEngine
from options_backtester.engine.clock import TradingClock
from options_backtester.engine.pipeline import (
    AlgoPipelineBacktester, PipelineContext, PipelineLogRow, StepDecision,
    # Scheduling
    RunMonthly, RunWeekly, RunQuarterly, RunYearly, RunDaily,
    RunOnce, RunOnDate, RunAfterDate, RunEveryNPeriods,
    Or, Not,
    # Selection
    SelectThese, SelectAll, SelectHasData, SelectMomentum, SelectN, SelectWhere,
    # Weighting
    WeighSpecified, WeighEqually, WeighInvVol, WeighMeanVar, WeighERC, TargetVol,
    # Weight limits
    LimitWeights,
    # Capital flows
    CapitalFlow,
    # Risk
    MaxDrawdownGuard,
    # Rebalancing
    Rebalance, RebalanceOverTime,
)
from options_backtester.engine.algo_adapters import (
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
from options_backtester.engine.strategy_tree import StrategyTreeNode, StrategyTreeEngine

# Analytics
from options_backtester.analytics.stats import BacktestStats
from options_backtester.analytics.trade_log import TradeLog
from options_backtester.analytics.tearsheet import TearsheetReport, build_tearsheet

__all__ = [
    # Core types
    "Direction", "OptionType", "Order", "Signal", "Fill", "Greeks",
    "OptionContract", "StockAllocation", "Stock", "get_order",
    # Data
    "Schema", "Field", "Filter", "CsvOptionsProvider", "CsvStocksProvider",
    # Strategy
    "Strategy", "StrategyLeg",
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
    "RunOnce", "RunOnDate", "RunAfterDate", "RunEveryNPeriods",
    "Or", "Not",
    "SelectThese", "SelectAll", "SelectHasData", "SelectMomentum", "SelectN", "SelectWhere",
    "WeighSpecified", "WeighEqually", "WeighInvVol", "WeighMeanVar", "WeighERC", "TargetVol",
    "LimitWeights", "CapitalFlow",
    "MaxDrawdownGuard", "Rebalance", "RebalanceOverTime",
    "EngineAlgo", "EngineStepDecision", "EnginePipelineContext", "EngineRunMonthly",
    "BudgetPercent", "RangeFilter", "SelectByDelta", "SelectByDTE", "IVRankFilter",
    "MaxGreekExposure", "ExitOnThreshold",
    "StrategyTreeNode", "StrategyTreeEngine",
    # Analytics
    "BacktestStats", "TradeLog", "TearsheetReport", "build_tearsheet",
]
