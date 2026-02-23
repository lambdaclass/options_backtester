"""Smoke tests — verify all public imports work."""


def test_top_level_imports():
    """All public symbols importable from options_backtester."""
    from options_backtester import (
        # Core
        Direction, OptionType, Order, Signal, Fill, Greeks,
        OptionContract, StockAllocation, Stock, get_order,
        # Data
        Schema, Field, Filter, CsvOptionsProvider, CsvStocksProvider,
        # Strategy
        Strategy, StrategyLeg,
        # Execution
        NoCosts, PerContractCommission, TieredCommission, SpreadSlippage,
        MarketAtBidAsk, MidPrice, VolumeAwareFill,
        CapitalBased, FixedQuantity, FixedDollar, PercentOfPortfolio,
        FirstMatch, NearestDelta, MaxOpenInterest,
        # Portfolio
        Portfolio, OptionPosition, aggregate_greeks,
        RiskManager, MaxDelta, MaxVega, MaxDrawdown,
        # Engine
        BacktestEngine, TradingClock,
        # Analytics
        BacktestStats, TradeLog,
    )
    # Quick sanity: verify some aren't None
    assert Direction is not None
    assert BacktestEngine is not None
    assert BacktestStats is not None


def test_compat_import():
    """Backward-compatible Backtest class works."""
    from options_backtester.compat.v0 import Backtest, Stock
    bt = Backtest({"stocks": 0.90, "options": 0.10, "cash": 0.0})
    assert bt.allocation["stocks"] == 0.90
    assert Stock is not None


def test_strategy_presets_import():
    """Strategy presets importable."""
    from options_backtester.strategy.presets import (
        strangle, iron_condor, covered_call, cash_secured_put, collar, butterfly,
    )
    assert callable(strangle)
    assert callable(iron_condor)
    assert callable(covered_call)
    assert callable(cash_secured_put)
    assert callable(collar)
    assert callable(butterfly)
