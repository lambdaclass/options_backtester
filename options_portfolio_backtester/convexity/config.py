"""Configuration: instrument registry and backtest parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for a single instrument."""

    symbol: str
    options_file: str
    stocks_file: str
    target_delta: float = -0.10
    dte_min: int = 14
    dte_max: int = 60
    tail_drop: float = 0.20


@dataclass(frozen=True)
class BacktestConfig:
    """Global backtest parameters."""

    initial_capital: float = 1_000_000.0
    budget_pct: float = 0.005  # 0.5% of portfolio per month on puts
    target_delta: float = -0.10
    dte_min: int = 14
    dte_max: int = 60
    tail_drop: float = 0.20
    instruments: list[InstrumentConfig] = field(default_factory=list)


def default_config(
    options_file: str = "data/processed/options.csv",
    stocks_file: str = "data/processed/stocks.csv",
) -> BacktestConfig:
    """Default config with SPY only."""
    spy = InstrumentConfig(
        symbol="SPY",
        options_file=options_file,
        stocks_file=stocks_file,
    )
    return BacktestConfig(instruments=[spy])
