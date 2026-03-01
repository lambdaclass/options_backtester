"""Tests for convexity config."""

from options_portfolio_backtester.convexity.config import (
    BacktestConfig,
    InstrumentConfig,
    default_config,
)


class TestConfig:
    def test_instrument_defaults(self):
        inst = InstrumentConfig(symbol="SPY", options_file="o.csv", stocks_file="s.csv")
        assert inst.target_delta == -0.10
        assert inst.dte_min == 14
        assert inst.dte_max == 60
        assert inst.tail_drop == 0.20

    def test_backtest_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 1_000_000.0
        assert cfg.budget_pct == 0.005

    def test_default_config(self):
        cfg = default_config()
        assert len(cfg.instruments) == 1
        assert cfg.instruments[0].symbol == "SPY"
