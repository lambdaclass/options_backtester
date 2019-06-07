import pandas as pd

from backtester.datahandler import Schema
from .strategy_leg import StrategyLeg
from .signal import Signal, get_order


class Strategy:
    """Options strategy class.
    Takes in a number of `legs` (option contracts), and filters that determine
    entry and exit conditions.
    """

    def __init__(self, schema):
        assert isinstance(schema, Schema)
        self.schema = schema
        self.legs = []

    def add_leg(self, leg):
        """Adds leg to the strategy"""
        assert isinstance(leg, StrategyLeg)
        assert self.schema == leg.schema
        self.legs.append(leg)
        return self

    def add_legs(self, legs):
        """Adds legs to the strategy"""
        for leg in legs:
            assert isinstance(leg, StrategyLeg)
            assert self.schema == leg.schema
        self.legs.extend(legs)
        return self

    def remove_leg(self, leg_number):
        """Removes leg from the strategy"""
        self.legs.pop(leg_number)
        return self

    def remove_legs(self):
        """Removes *all* legs from the strategy"""
        self.legs = []
        return self

    def signals(self, data):
        """Iterates over `data` and yields a tuple of
        (date, entry_signals, exit_signals) for each time step.
        """
        assert self.schema == data.schema

        for date, group in data.iter_dates():
            entry_legs = self._filter_legs(group, signal=Signal.ENTRY)
            if any(df.empty for df in entry_legs):
                entry_df = pd.DataFrame()
            else:
                entry_df = pd.concat(entry_legs, axis=1)

            exit_legs = self._filter_legs(group, signal=Signal.EXIT)
            exit_df = pd.concat(exit_legs, axis=1)
            entry_df.legs = exit_df.legs = exit_df.columns.levels[0]

            yield (date, entry_df, exit_df)

    def _filter_legs(self, data, signal=Signal.ENTRY):
        """Returns a list of `pd.DataFrame`.
        Each dataframe contains signals for each leg in the strategy.
        """
        schema = self.schema
        dfs = []
        for number, leg in enumerate(self.legs, start=1):
            flt = leg.entry_filter if signal == Signal.ENTRY else leg.exit_filter
            df = flt(data)
            price = leg.direction.value
            fields = {
                schema["contract"]: "contract",
                schema["underlying"]: "underlying",
                schema["expiration"]: "expiration",
                schema["type"]: "type",
                schema["strike"]: "strike",
                schema[price]: "price"
            }
            subset_df = df.loc[:, fields.keys()]
            subset_df.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, signal)
            subset_df["order"] = order.name
            col_index = pd.MultiIndex.from_product([["leg_{}".format(number)],
                                                    subset_df.columns])
            subset_df.columns = col_index
            dfs.append(subset_df.reset_index(drop=True))

        return dfs

    def __repr__(self):
        return "Strategy(legs={})".format(self.legs)
