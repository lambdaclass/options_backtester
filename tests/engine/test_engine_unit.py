"""Unit tests for BacktestEngine internals — repr, metadata, static methods."""

import json

from options_portfolio_backtester.engine.engine import BacktestEngine


class TestBacktestEngineRepr:
    def test_repr_basic(self):
        engine = BacktestEngine(
            allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
            initial_capital=500_000,
        )
        r = repr(engine)
        assert "BacktestEngine" in r
        assert "500000" in r
        assert "NoCosts" in r

    def test_repr_with_custom_cost_model(self):
        from options_portfolio_backtester.execution.cost_model import PerContractCommission
        engine = BacktestEngine(
            allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
            cost_model=PerContractCommission(0.65),
        )
        r = repr(engine)
        assert "PerContractCommission" in r


class TestSha256Json:
    def test_deterministic(self):
        payload = {"a": 1, "b": "hello"}
        h1 = BacktestEngine._sha256_json(payload)
        h2 = BacktestEngine._sha256_json(payload)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_inputs_different_hashes(self):
        h1 = BacktestEngine._sha256_json({"x": 1})
        h2 = BacktestEngine._sha256_json({"x": 2})
        assert h1 != h2

    def test_key_order_independent(self):
        h1 = BacktestEngine._sha256_json({"a": 1, "b": 2})
        h2 = BacktestEngine._sha256_json({"b": 2, "a": 1})
        assert h1 == h2


class TestGitSha:
    def test_returns_string(self):
        sha = BacktestEngine._git_sha()
        assert isinstance(sha, str)
        # Should be either a hex sha or "unknown"
        assert sha == "unknown" or len(sha) == 40


class TestFlatTradeLogToMultiIndex:
    def test_empty_dataframe(self):
        import pandas as pd
        engine = BacktestEngine(
            allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
        )
        result = engine._flat_trade_log_to_multiindex(pd.DataFrame())
        assert result.empty

    def test_converts_double_underscore_columns(self):
        import pandas as pd
        df = pd.DataFrame({
            "leg_1__contract": ["SPY_C_001"],
            "leg_1__cost": [500.0],
            "totals__qty": [1],
        })
        engine = BacktestEngine(
            allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
        )
        result = engine._flat_trade_log_to_multiindex(df)
        assert isinstance(result.columns, pd.MultiIndex)
        assert ("leg_1", "contract") in result.columns
        assert ("totals", "qty") in result.columns


class TestEventsDataframe:
    def test_empty_events(self):
        engine = BacktestEngine(
            allocation={"stocks": 0.9, "options": 0.1, "cash": 0.0},
        )
        df = engine.events_dataframe()
        assert list(df.columns) == ["date", "event", "status"]
        assert len(df) == 0


class TestAllocationNormalization:
    def test_normalizes_to_sum_one(self):
        engine = BacktestEngine(
            allocation={"stocks": 60, "options": 30, "cash": 10},
        )
        total = sum(engine.allocation.values())
        assert abs(total - 1.0) < 1e-10

    def test_missing_keys_default_to_zero(self):
        engine = BacktestEngine(allocation={"stocks": 1.0})
        assert engine.allocation["options"] == 0.0
        assert engine.allocation["cash"] == 0.0
