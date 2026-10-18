"""options_portfolio_backtester — the open-source options backtesting framework."""

from options_portfolio_backtester.analytics.financing import apply_financing
from options_portfolio_backtester.analytics.results import BacktestResults, hash_data_file
from options_portfolio_backtester.analytics.robustness import (
    calendar_summary,
    entry_calendars,
    run_over_calendars,
)
from options_portfolio_backtester.analytics.stats import BacktestStats, LookbackReturns, PeriodStats
from options_portfolio_backtester.analytics.summary import summary
from options_portfolio_backtester.analytics.tearsheet import TearsheetReport, build_tearsheet
from options_portfolio_backtester.analytics.trade_log import TradeLog
from options_portfolio_backtester.core.types import (
    Direction,
    Fill,
    Greeks,
    OptionContract,
    OptionType,
    Order,
    Signal,
    Stock,
    StockAllocation,
    Type,
    get_order,
)
from options_portfolio_backtester.data.cleaning import clean_options_chain, flag_bad_quotes
from options_portfolio_backtester.data.providers import (
    CsvOptionsProvider,
    CsvStocksProvider,
    HistoricalOptionsData,
    TiingoData,
)
from options_portfolio_backtester.data.schema import Field, Filter, Schema
from options_portfolio_backtester.engine.clock import TradingClock
from options_portfolio_backtester.engine.engine import BacktestEngine
from options_portfolio_backtester.execution.cost_model import (
    NoCosts,
    PerContractCommission,
    SpreadSlippage,
    TieredCommission,
)
from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk, MidPrice, VolumeAwareFill
from options_portfolio_backtester.execution.signal_selector import (
    FirstMatch,
    MaxOpenInterest,
    NearestDelta,
)
from options_portfolio_backtester.execution.sizer import (
    CapitalBased,
    FixedDollar,
    FixedQuantity,
    PercentOfPortfolio,
)
from options_portfolio_backtester.portfolio.greeks import aggregate_greeks
from options_portfolio_backtester.portfolio.portfolio import Portfolio
from options_portfolio_backtester.portfolio.position import OptionPosition
from options_portfolio_backtester.portfolio.risk import MaxDelta, MaxDrawdown, MaxVega, RiskManager
from options_portfolio_backtester.strategy.presets import (
    Strangle,
    deep_otm_put,
    near_atm_put_protection,
)
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

__all__ = [
    # Core types
    "Direction", "OptionType", "Type", "Order", "Signal", "Fill", "Greeks",
    "OptionContract", "StockAllocation", "Stock", "get_order",
    # Data
    "Schema", "Field", "Filter", "CsvOptionsProvider", "CsvStocksProvider",
    "clean_options_chain", "flag_bad_quotes",
    "TiingoData", "HistoricalOptionsData",
    # Strategy
    "Strategy", "StrategyLeg", "Strangle",
    "deep_otm_put", "near_atm_put_protection",
    # Results
    "BacktestResults", "hash_data_file",
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
    # Analytics
    "BacktestStats", "PeriodStats", "LookbackReturns",
    "entry_calendars", "run_over_calendars", "calendar_summary", "apply_financing",
    "TradeLog", "TearsheetReport", "build_tearsheet",
    "summary",
]
