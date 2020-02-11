from .strategy_leg import StrategyLeg
from .strategy import Strategy
from backtester.enums import Direction, Type


class Strangle(Strategy):
    def __init__(self,
                 schema,
                 name,
                 underlying,
                 dte_entry_range,
                 dte_exit,
                 otm_pct=0,
                 pct_tolerance=1,
                 exit_thresholds=(float('inf'), float('inf')),
                 shares_per_contract=100):
        assert (name.lower() == 'short' or name.lower() == 'long')
        super().__init__(schema, shares_per_contract)
        direction = Direction.SELL if name.lower() == 'short' else Direction.BUY

        leg1 = StrategyLeg(
            "leg_1",
            schema,
            option_type=Type.CALL,
            direction=direction,
        )

        otm_lower_bound = (otm_pct - pct_tolerance) / 100
        otm_upper_bound = (otm_pct + pct_tolerance) / 100

        leg1.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_entry_range[0]) & (
            schema.dte <= dte_entry_range[1]) & (schema.strike >= schema.underlying_last *
                                                 (1 + otm_lower_bound)) & (schema.strike <= schema.underlying_last *
                                                                           (1 + otm_upper_bound))
        leg1.exit_filter = (schema.dte <= dte_exit)

        leg2 = StrategyLeg("leg_2", schema, option_type=Type.PUT, direction=direction)
        leg2.entry_filter = (schema.underlying == underlying) & (schema.dte >= dte_entry_range[0]) & (
            schema.dte <= dte_entry_range[1]) & (schema.strike <= schema.underlying_last *
                                                 (1 - otm_lower_bound)) & (schema.strike >= schema.underlying_last *
                                                                           (1 - otm_upper_bound))
        leg2.exit_filter = (schema.dte <= dte_exit)

        self.add_legs([leg1, leg2])
        self.add_exit_thresholds(exit_thresholds[0], exit_thresholds[1])
