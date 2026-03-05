"""Deep strategy & risk tests — presets, multi-leg construction, portfolio, positions, Greeks.

Tests strategy construction edge cases, preset validation, risk constraint
boundary conditions, and position/portfolio accounting.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester.core.types import (
    Direction,
    OptionType,
    Order,
    Signal,
    Greeks,
    Fill,
    Stock,
    get_order,
)
from options_portfolio_backtester.data.providers import HistoricalOptionsData
from options_portfolio_backtester.data.schema import Schema, Field, Filter
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.strategy.presets import (
    strangle,
    iron_condor,
    covered_call,
    cash_secured_put,
    collar,
    butterfly,
    Strangle,
)
from options_portfolio_backtester.portfolio.risk import (
    RiskManager,
    MaxDelta,
    MaxVega,
    MaxDrawdown,
)
from options_portfolio_backtester.portfolio.portfolio import Portfolio, StockHolding
from options_portfolio_backtester.portfolio.position import OptionPosition, PositionLeg
from options_portfolio_backtester.portfolio.greeks import aggregate_greeks

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
OPTIONS_FILE = os.path.join(TEST_DIR, "options_data.csv")


@pytest.fixture
def schema():
    data = HistoricalOptionsData(OPTIONS_FILE)
    return data.schema


# ---------------------------------------------------------------------------
# Strategy preset construction
# ---------------------------------------------------------------------------


class TestStranglePreset:
    def test_has_two_legs(self, schema):
        s = strangle(schema, "SPX", Direction.BUY, (30, 60), 14)
        assert len(s.legs) == 2

    def test_leg_types(self, schema):
        s = strangle(schema, "SPX", Direction.BUY, (30, 60), 14)
        types = {leg.type for leg in s.legs}
        assert types == {OptionType.CALL, OptionType.PUT}

    def test_leg_directions_long(self, schema):
        s = strangle(schema, "SPX", Direction.BUY, (30, 60), 14)
        for leg in s.legs:
            assert leg.direction == Direction.BUY

    def test_leg_directions_short(self, schema):
        s = strangle(schema, "SPX", Direction.SELL, (30, 60), 14)
        for leg in s.legs:
            assert leg.direction == Direction.SELL

    def test_exit_thresholds_applied(self, schema):
        s = strangle(schema, "SPX", Direction.BUY, (30, 60), 14,
                      exit_thresholds=(2.0, 0.5))
        assert s.exit_thresholds == (2.0, 0.5)

    def test_default_exit_thresholds(self, schema):
        s = strangle(schema, "SPX", Direction.BUY, (30, 60), 14)
        assert s.exit_thresholds == (float("inf"), float("inf"))


class TestIronCondorPreset:
    def test_has_four_legs(self, schema):
        s = iron_condor(schema, "SPX", (30, 60), 14)
        assert len(s.legs) == 4

    def test_two_sells_two_buys(self, schema):
        s = iron_condor(schema, "SPX", (30, 60), 14)
        sells = [l for l in s.legs if l.direction == Direction.SELL]
        buys = [l for l in s.legs if l.direction == Direction.BUY]
        assert len(sells) == 2
        assert len(buys) == 2

    def test_has_both_option_types(self, schema):
        s = iron_condor(schema, "SPX", (30, 60), 14)
        types = {l.type for l in s.legs}
        assert types == {OptionType.CALL, OptionType.PUT}


class TestCoveredCallPreset:
    def test_one_sell_call_leg(self, schema):
        s = covered_call(schema, "SPX", (30, 60), 14)
        assert len(s.legs) == 1
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[0].type == OptionType.CALL

    def test_otm_pct_applied(self, schema):
        s1 = covered_call(schema, "SPX", (30, 60), 14, otm_pct=1.0)
        s2 = covered_call(schema, "SPX", (30, 60), 14, otm_pct=5.0)
        # Different OTM should produce different entry filters
        assert s1.legs[0].entry_filter.query != s2.legs[0].entry_filter.query


class TestCashSecuredPutPreset:
    def test_one_sell_put_leg(self, schema):
        s = cash_secured_put(schema, "SPX", (30, 60), 14)
        assert len(s.legs) == 1
        assert s.legs[0].direction == Direction.SELL
        assert s.legs[0].type == OptionType.PUT


class TestCollarPreset:
    def test_two_legs(self, schema):
        s = collar(schema, "SPX", (30, 60), 14)
        assert len(s.legs) == 2

    def test_short_call_long_put(self, schema):
        s = collar(schema, "SPX", (30, 60), 14)
        call_leg = [l for l in s.legs if l.type == OptionType.CALL][0]
        put_leg = [l for l in s.legs if l.type == OptionType.PUT][0]
        assert call_leg.direction == Direction.SELL
        assert put_leg.direction == Direction.BUY


class TestButterflyPreset:
    def test_three_legs(self, schema):
        s = butterfly(schema, "SPX", (30, 60), 14)
        assert len(s.legs) == 3

    def test_buy_sell_buy_pattern(self, schema):
        s = butterfly(schema, "SPX", (30, 60), 14)
        dirs = [l.direction for l in s.legs]
        assert dirs == [Direction.BUY, Direction.SELL, Direction.BUY]

    def test_call_butterfly(self, schema):
        s = butterfly(schema, "SPX", (30, 60), 14, option_type=OptionType.CALL)
        for leg in s.legs:
            assert leg.type == OptionType.CALL

    def test_put_butterfly(self, schema):
        s = butterfly(schema, "SPX", (30, 60), 14, option_type=OptionType.PUT)
        for leg in s.legs:
            assert leg.type == OptionType.PUT

    def test_lower_wing_has_asc_sort(self, schema):
        s = butterfly(schema, "SPX", (30, 60), 14)
        assert s.legs[0].entry_sort == ("strike", True)

    def test_upper_wing_has_desc_sort(self, schema):
        s = butterfly(schema, "SPX", (30, 60), 14)
        assert s.legs[2].entry_sort == ("strike", False)


class TestStrangleClassBased:
    def test_long_strangle(self, schema):
        s = Strangle(schema, "long", "SPX", (30, 60), 14)
        assert len(s.legs) == 2
        for leg in s.legs:
            assert leg.direction == Direction.BUY

    def test_short_strangle(self, schema):
        s = Strangle(schema, "short", "SPX", (30, 60), 14)
        for leg in s.legs:
            assert leg.direction == Direction.SELL

    def test_invalid_name_raises(self, schema):
        with pytest.raises(AssertionError):
            Strangle(schema, "neutral", "SPX", (30, 60), 14)


# ---------------------------------------------------------------------------
# Strategy operations
# ---------------------------------------------------------------------------


class TestStrategyOperations:
    def test_add_and_remove_leg(self, schema):
        s = Strategy(schema)
        leg = StrategyLeg("x", schema, option_type=OptionType.PUT, direction=Direction.BUY)
        leg.entry_filter = schema.underlying == "SPX"
        leg.exit_filter = schema.dte <= 30
        s.add_leg(leg)
        assert len(s.legs) == 1
        s.remove_leg(0)
        assert len(s.legs) == 0

    def test_clear_legs(self, schema):
        s = strangle(schema, "SPX", Direction.BUY, (30, 60), 14)
        assert len(s.legs) == 2
        s.clear_legs()
        assert len(s.legs) == 0

    def test_exit_thresholds_validation(self, schema):
        s = Strategy(schema)
        with pytest.raises(AssertionError):
            s.add_exit_thresholds(profit_pct=-1.0)
        with pytest.raises(AssertionError):
            s.add_exit_thresholds(loss_pct=-0.5)

    def test_filter_thresholds_series(self, schema):
        s = Strategy(schema)
        s.add_exit_thresholds(profit_pct=0.5, loss_pct=0.3)
        entry = pd.Series([-100.0, -200.0, -50.0])
        current = pd.Series([-50.0, -300.0, -25.0])
        result = s.filter_thresholds(entry, current)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool


# ---------------------------------------------------------------------------
# Strategy leg entry/exit filters
# ---------------------------------------------------------------------------


class TestStrategyLegFilters:
    def test_base_entry_filter_buy_requires_ask_gt_zero(self, schema):
        leg = StrategyLeg("x", schema, option_type=OptionType.PUT, direction=Direction.BUY)
        assert "ask > 0" in leg.entry_filter.query

    def test_base_entry_filter_sell_requires_bid_gt_zero(self, schema):
        leg = StrategyLeg("x", schema, option_type=OptionType.PUT, direction=Direction.SELL)
        assert "bid > 0" in leg.entry_filter.query

    def test_custom_entry_filter_combines_with_base(self, schema):
        leg = StrategyLeg("x", schema, option_type=OptionType.CALL, direction=Direction.BUY)
        leg.entry_filter = schema.dte >= 30
        assert "ask > 0" in leg.entry_filter.query
        assert "dte >= 30" in leg.entry_filter.query

    def test_exit_filter_includes_type(self, schema):
        leg = StrategyLeg("x", schema, option_type=OptionType.PUT, direction=Direction.BUY)
        assert "put" in leg.exit_filter.query


# ---------------------------------------------------------------------------
# Risk constraints — boundary conditions
# ---------------------------------------------------------------------------


class TestMaxDeltaConstraint:
    def test_within_limit_allowed(self):
        c = MaxDelta(limit=100.0)
        assert c.check(Greeks(delta=50), Greeks(delta=30), 1e6, 1e6) is True

    def test_at_limit_allowed(self):
        c = MaxDelta(limit=100.0)
        assert c.check(Greeks(delta=50), Greeks(delta=50), 1e6, 1e6) is True

    def test_exceeds_limit_blocked(self):
        c = MaxDelta(limit=100.0)
        assert c.check(Greeks(delta=90), Greeks(delta=20), 1e6, 1e6) is False

    def test_negative_delta(self):
        c = MaxDelta(limit=50.0)
        # -30 + -30 = -60, abs = 60 > 50
        assert c.check(Greeks(delta=-30), Greeks(delta=-30), 1e6, 1e6) is False

    def test_describe(self):
        c = MaxDelta(limit=50.0)
        assert "50.0" in c.describe()


class TestMaxVegaConstraint:
    def test_within_limit(self):
        c = MaxVega(limit=100.0)
        assert c.check(Greeks(vega=40), Greeks(vega=40), 1e6, 1e6) is True

    def test_exceeds_limit(self):
        c = MaxVega(limit=50.0)
        assert c.check(Greeks(vega=30), Greeks(vega=30), 1e6, 1e6) is False


class TestMaxDrawdownConstraint:
    def test_no_drawdown_allowed(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        assert c.check(Greeks(), Greeks(), 1e6, 1e6) is True

    def test_at_drawdown_limit(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        # dd = (1e6 - 800000) / 1e6 = 0.20 → NOT blocked (< not <=)
        assert c.check(Greeks(), Greeks(), 800_000, 1e6) is False

    def test_beyond_drawdown(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        assert c.check(Greeks(), Greeks(), 700_000, 1e6) is False

    def test_peak_is_zero(self):
        c = MaxDrawdown(max_dd_pct=0.20)
        assert c.check(Greeks(), Greeks(), 100, 0) is True


class TestRiskManagerComposite:
    def test_empty_constraints_allows_all(self):
        rm = RiskManager()
        ok, reason = rm.is_allowed(Greeks(), Greeks(), 1e6, 1e6)
        assert ok is True
        assert reason == ""

    def test_single_violation_blocks(self):
        rm = RiskManager([MaxDelta(limit=10)])
        ok, reason = rm.is_allowed(Greeks(delta=50), Greeks(delta=50), 1e6, 1e6)
        assert ok is False
        assert "MaxDelta" in reason

    def test_first_failure_reported(self):
        rm = RiskManager([MaxDelta(limit=10), MaxVega(limit=10)])
        ok, reason = rm.is_allowed(
            Greeks(delta=50, vega=50), Greeks(delta=50, vega=50), 1e6, 1e6
        )
        assert ok is False
        assert "MaxDelta" in reason  # first constraint to fail

    def test_all_pass(self):
        rm = RiskManager([MaxDelta(limit=1000), MaxVega(limit=1000)])
        ok, _ = rm.is_allowed(Greeks(delta=1, vega=1), Greeks(delta=1, vega=1), 1e6, 1e6)
        assert ok is True


# ---------------------------------------------------------------------------
# Greeks algebra
# ---------------------------------------------------------------------------


class TestGreeksAlgebra:
    def test_addition(self):
        g1 = Greeks(delta=1, gamma=2, theta=3, vega=4)
        g2 = Greeks(delta=10, gamma=20, theta=30, vega=40)
        result = g1 + g2
        assert result.delta == 11
        assert result.gamma == 22
        assert result.theta == 33
        assert result.vega == 44

    def test_scalar_multiplication(self):
        g = Greeks(delta=1, gamma=2, theta=3, vega=4)
        result = g * 3
        assert result.delta == 3
        assert result.vega == 12

    def test_rmul(self):
        g = Greeks(delta=1, gamma=2, theta=3, vega=4)
        result = 3 * g
        assert result == g * 3

    def test_negation(self):
        g = Greeks(delta=10, gamma=5, theta=-3, vega=1)
        neg = -g
        assert neg.delta == -10
        assert neg.theta == 3

    def test_as_dict(self):
        g = Greeks(delta=1, gamma=2, theta=3, vega=4)
        d = g.as_dict
        assert d["delta"] == 1
        assert len(d) == 4


# ---------------------------------------------------------------------------
# Order mapping
# ---------------------------------------------------------------------------


class TestOrderMapping:
    def test_buy_entry_bto(self):
        assert get_order(Direction.BUY, Signal.ENTRY) == Order.BTO

    def test_buy_exit_stc(self):
        assert get_order(Direction.BUY, Signal.EXIT) == Order.STC

    def test_sell_entry_sto(self):
        assert get_order(Direction.SELL, Signal.ENTRY) == Order.STO

    def test_sell_exit_btc(self):
        assert get_order(Direction.SELL, Signal.EXIT) == Order.BTC

    def test_order_inversion(self):
        assert ~Order.BTO == Order.STC
        assert ~Order.STC == Order.BTO
        assert ~Order.STO == Order.BTC
        assert ~Order.BTC == Order.STO


class TestDirectionInversion:
    def test_buy_inverts_to_sell(self):
        assert ~Direction.BUY == Direction.SELL

    def test_sell_inverts_to_buy(self):
        assert ~Direction.SELL == Direction.BUY


class TestOptionTypeInversion:
    def test_call_inverts_to_put(self):
        assert ~OptionType.CALL == OptionType.PUT

    def test_put_inverts_to_call(self):
        assert ~OptionType.PUT == OptionType.CALL


# ---------------------------------------------------------------------------
# Fill dataclass
# ---------------------------------------------------------------------------


class TestFillNotional:
    def test_buy_fill_negative_notional(self):
        f = Fill(price=5.0, quantity=10, direction=Direction.BUY)
        # BUY → sign=-1, notional = -1 * 5 * 10 * 100 = -5000
        assert f.notional == -5000.0

    def test_sell_fill_positive_notional(self):
        f = Fill(price=5.0, quantity=10, direction=Direction.SELL)
        assert f.notional == 5000.0

    def test_commission_deducted(self):
        f = Fill(price=5.0, quantity=10, direction=Direction.BUY, commission=50.0)
        assert f.notional == -5050.0

    def test_slippage_deducted(self):
        f = Fill(price=5.0, quantity=10, direction=Direction.SELL, slippage=100.0)
        assert f.notional == 4900.0


# ---------------------------------------------------------------------------
# Portfolio and Position
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_initial_cash(self):
        p = Portfolio(initial_cash=100_000)
        assert p.cash == 100_000

    def test_add_remove_option_position(self):
        p = Portfolio()
        pos = OptionPosition(position_id=0, quantity=10)
        p.add_option_position(pos)
        assert 0 in p.option_positions
        removed = p.remove_option_position(0)
        assert removed is pos
        assert 0 not in p.option_positions

    def test_remove_nonexistent_returns_none(self):
        p = Portfolio()
        assert p.remove_option_position(999) is None

    def test_stock_holdings(self):
        p = Portfolio()
        p.set_stock_holding("AAPL", 100, 150.0)
        assert p.stock_holdings["AAPL"].quantity == 100
        assert p.stocks_value({"AAPL": 160.0}) == 16_000.0

    def test_clear_stock_holdings(self):
        p = Portfolio()
        p.set_stock_holding("AAPL", 100, 150.0)
        p.clear_stock_holdings()
        assert len(p.stock_holdings) == 0

    def test_total_value(self):
        p = Portfolio(initial_cash=10_000)
        p.set_stock_holding("AAPL", 100, 150.0)
        total = p.total_value(
            stock_prices={"AAPL": 160.0},
            option_prices={},
            shares_per_contract=100,
        )
        assert total == 10_000 + 16_000

    def test_next_position_id_increments(self):
        p = Portfolio()
        assert p.next_position_id() == 0
        assert p.next_position_id() == 1
        assert p.next_position_id() == 2


class TestPositionLeg:
    def test_buy_leg_positive_value(self):
        leg = PositionLeg(
            name="leg_1", contract_id="SPX1", underlying="SPX",
            expiration=pd.Timestamp("2025-01-01"), option_type=OptionType.PUT,
            strike=100.0, entry_price=5.0, direction=Direction.BUY,
            order=Order.BTO,
        )
        # BUY: value = +1 * current_price * qty * spc
        value = leg.current_value(current_price=6.0, quantity=10, shares_per_contract=100)
        assert value == 6000.0

    def test_sell_leg_negative_value(self):
        leg = PositionLeg(
            name="leg_1", contract_id="SPX1", underlying="SPX",
            expiration=pd.Timestamp("2025-01-01"), option_type=OptionType.PUT,
            strike=100.0, entry_price=5.0, direction=Direction.SELL,
            order=Order.STO,
        )
        value = leg.current_value(current_price=6.0, quantity=10, shares_per_contract=100)
        assert value == -6000.0

    def test_exit_order(self):
        leg = PositionLeg(
            name="x", contract_id="X", underlying="SPX",
            expiration=pd.Timestamp("2025-01-01"), option_type=OptionType.CALL,
            strike=100.0, entry_price=5.0, direction=Direction.BUY, order=Order.BTO,
        )
        assert leg.exit_order == Order.STC


class TestOptionPosition:
    def test_multi_leg_value(self):
        pos = OptionPosition(position_id=0, quantity=10)
        pos.add_leg(PositionLeg(
            "call", "C1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.CALL, 100.0, 3.0, Direction.BUY, Order.BTO,
        ))
        pos.add_leg(PositionLeg(
            "put", "P1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.PUT, 100.0, 2.0, Direction.SELL, Order.STO,
        ))
        value = pos.current_value({"call": 4.0, "put": 3.0}, shares_per_contract=100)
        # call: +4*10*100=4000, put: -3*10*100=-3000
        assert value == 1000.0

    def test_greeks_aggregation(self):
        pos = OptionPosition(position_id=0, quantity=5)
        pos.add_leg(PositionLeg(
            "call", "C1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.CALL, 100.0, 3.0, Direction.BUY, Order.BTO,
        ))
        pos.add_leg(PositionLeg(
            "put", "P1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.PUT, 100.0, 2.0, Direction.SELL, Order.STO,
        ))
        greeks = pos.greeks({
            "call": Greeks(delta=0.5, gamma=0.02, theta=-0.01, vega=0.1),
            "put": Greeks(delta=-0.3, gamma=0.01, theta=-0.02, vega=0.05),
        })
        # call: BUY → sign=+1, qty=5: delta=0.5*5=2.5
        # put: SELL → sign=-1, qty=5: delta=-0.3*(-1)*5=1.5
        assert abs(greeks.delta - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# Portfolio-level Greeks aggregation
# ---------------------------------------------------------------------------


class TestAggregateGreeks:
    def test_empty_portfolio(self):
        g = aggregate_greeks({}, {})
        assert g.delta == 0.0

    def test_single_position(self):
        pos = OptionPosition(position_id=0, quantity=1)
        pos.add_leg(PositionLeg(
            "leg_1", "C1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.CALL, 100.0, 5.0, Direction.BUY, Order.BTO,
        ))
        greeks_map = {0: {"leg_1": Greeks(delta=0.5, gamma=0.02, theta=-0.01, vega=0.1)}}
        result = aggregate_greeks({0: pos}, greeks_map)
        assert abs(result.delta - 0.5) < 1e-10

    def test_multiple_positions(self):
        p1 = OptionPosition(position_id=0, quantity=10)
        p1.add_leg(PositionLeg(
            "leg_1", "C1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.CALL, 100.0, 5.0, Direction.BUY, Order.BTO,
        ))
        p2 = OptionPosition(position_id=1, quantity=5)
        p2.add_leg(PositionLeg(
            "leg_1", "P1", "SPX", pd.Timestamp("2025-01-01"),
            OptionType.PUT, 100.0, 3.0, Direction.BUY, Order.BTO,
        ))
        greeks_map = {
            0: {"leg_1": Greeks(delta=0.5)},
            1: {"leg_1": Greeks(delta=-0.3)},
        }
        result = aggregate_greeks({0: p1, 1: p2}, greeks_map)
        # p1: BUY, qty=10, delta=0.5*1*10 = 5.0
        # p2: BUY, qty=5, delta=-0.3*1*5 = -1.5
        assert abs(result.delta - 3.5) < 1e-10
