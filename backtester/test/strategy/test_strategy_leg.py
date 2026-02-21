import pandas as pd

from backtester.strategy.strategy_leg import StrategyLeg
from backtester.datahandler.schema import Schema
from backtester.enums import Type, Direction


def make_options_df():
    """Minimal options DataFrame for testing filters."""
    return pd.DataFrame({
        'type': ['call', 'put', 'call', 'put'],
        'ask': [1.5, 2.0, 0.0, 1.0],
        'bid': [1.0, 1.5, 0.5, 0.0],
    })


class TestDefaultEntryFilter:
    def test_buy_call_filters_calls_with_positive_ask(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.BUY)
        df = make_options_df()
        result = df[leg.entry_filter(df)]
        # Should match rows where type=='call' AND ask > 0 => row 0 only (row 2 has ask=0)
        assert len(result) == 1
        assert result.iloc[0]['ask'] == 1.5

    def test_sell_put_filters_puts_with_positive_bid(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.SELL)
        df = make_options_df()
        result = df[leg.entry_filter(df)]
        # Should match rows where type=='put' AND bid > 0 => row 1 only (row 3 has bid=0)
        assert len(result) == 1
        assert result.iloc[0]['bid'] == 1.5

    def test_buy_put_filters_puts_with_positive_ask(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        df = make_options_df()
        result = df[leg.entry_filter(df)]
        # type=='put' AND ask > 0 => rows 1, 3
        assert len(result) == 2

    def test_sell_call_filters_calls_with_positive_bid(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.SELL)
        df = make_options_df()
        result = df[leg.entry_filter(df)]
        # type=='call' AND bid > 0 => rows 0, 2
        assert len(result) == 2


class TestDefaultExitFilter:
    def test_exit_filter_matches_type(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.BUY)
        df = make_options_df()
        result = df[leg.exit_filter(df)]
        # Should match all calls (rows 0, 2)
        assert len(result) == 2
        assert (result['type'] == 'call').all()


class TestCustomFilter:
    def test_custom_entry_filter_is_anded_with_base(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.CALL, direction=Direction.BUY)
        # Add a custom filter: ask > 1.0
        leg.entry_filter = schema.ask > 1.0
        df = make_options_df()
        result = df[leg.entry_filter(df)]
        # Base: type=='call' AND ask > 0. Custom AND'd: ask > 1.0
        # Row 0: type=call, ask=1.5 => matches (1.5 > 0 and 1.5 > 1.0)
        # Row 2: type=call, ask=0.0 => no (ask not > 0)
        assert len(result) == 1

    def test_custom_exit_filter_is_anded_with_base(self):
        schema = Schema.options()
        leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
        # Custom exit: bid > 1.0
        leg.exit_filter = schema.bid > 1.0
        df = make_options_df()
        result = df[leg.exit_filter(df)]
        # Base: type=='put'. Custom AND'd: bid > 1.0
        # Row 1: type=put, bid=1.5 => matches
        # Row 3: type=put, bid=0.0 => no
        assert len(result) == 1
