from collections import namedtuple

import pandas as pd

from backtester.datahandler import Schema
from backtester.option import Direction
from .strategy_leg import StrategyLeg
from .signal import Signal, get_order, Order

Condition = namedtuple('Condition', 'fields legs tolerance')


class Strategy:
    """Options strategy class.
    Takes in a number of `legs` (option contracts), and filters that determine
    entry and exit conditions.
    """

    def __init__(self, schema):
        assert isinstance(schema, Schema)
        self.schema = schema
        self.legs = []
        self.conditions = []
        self.entries = set()
        self.exit_thresholds = []
        self.dte_on_exit = 2

    def add_leg(self, leg):
        """Adds leg to the strategy"""
        assert isinstance(leg, StrategyLeg)
        assert self.schema == leg.schema
        leg.name = "leg_{}".format(len(self.legs) + 1)
        self.legs.append(leg)
        return self

    def add_legs(self, legs):
        """Adds legs to the strategy"""
        for leg in legs:
            assert isinstance(leg, StrategyLeg)
            assert self.schema == leg.schema
            self.add_leg(leg)
        return self

    def remove_leg(self, leg_number):
        """Removes leg from the strategy"""
        self.legs.pop(leg_number)
        return self

    def clear_legs(self):
        """Removes *all* legs from the strategy"""
        self.legs = []
        return self

    def add_condition(self, fields, legs=None, tolerance=0.0):
        """Adds a condition that all legs in `legs` should have the same value for `fields`"""
        assert all((f in self.schema for f in fields))
        if legs:
            assert all(legs, lambda l: l in self.legs)
        else:
            legs = self.legs

        self.conditions.append(Condition(fields, legs, tolerance))

    def register_entry(self, contract, price):
        """Allows the Backtester to register entries in order to allow exiting on
        given profit/loss levels"""
        self.entries.add(contract)

    def signals(self, data, bt):
        """Iterates over `data` and yields a tuple of
        `(date, entry_signals, exit_signals)` for each time step.
        """
        assert self.schema == data.schema

        for date, group in data.iter_dates():
            entry_legs = self._filter_legs(group, signal=Signal.ENTRY)

            if any(df.empty for df in entry_legs):
                entry_df = pd.DataFrame()
            else:
                entry_df = pd.concat(entry_legs, axis=1)

            exit_df = self._filter_exits(group, bt.inventory)

            yield (date, entry_df, exit_df)

    def _filter_legs(self, data, signal=Signal.ENTRY):
        """Returns a list of `pd.DataFrame`.
        Each dataframe contains signals for each leg in the strategy.
        """
        schema = self.schema
        dfs = []
        for leg in self.legs:
            if signal == Signal.ENTRY:
                flt = leg.entry_filter
                cost = leg.direction.value
            else:
                flt = leg.exit_filter
                cost = (~leg.direction).value

            df = flt(data)
            fields = {
                schema["contract"]: "contract",
                schema["underlying"]: "underlying",
                schema["expiration"]: "expiration",
                schema["type"]: "type",
                schema["strike"]: "strike",
                schema[cost]: "cost"
            }
            subset_df = df.loc[:, fields.keys()]
            subset_df.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, signal)
            subset_df["order"] = order.name

            # Change sign of cost for SELL orders
            if leg.direction == Direction.SELL:
                subset_df["cost"] = -subset_df["cost"]

            dfs.append(subset_df.reset_index(drop=True))

        return self._apply_conditions(dfs)

    def _filter_exits(self, data, inventory):
        exits = []
        for _, row in inventory.iterrows():
            old_price = 0
            current_price = 0
            contracts = set()
            is_empty = False
            filters_exit = False
            for leg in self.legs:
                contract = row[(leg.name, 'contract')]
                order = get_order(leg.direction, Signal.EXIT).name
                old_price += row[(leg.name, 'cost')]
                option = data[data['optionroot'] == contract]

                # This was originally to skip (and then remove) entries that are past their expiration and therefore
                # don't have a corresponding exit anymore (i.e, option is empty). It doesn't work, however, because
                # option might just be empty because of missing data in the middle. Moreover, even if the entry is
                # past its expiration the current code will still execute the other exit legs associated with it,
                # which is inaccurate. This last point can only be truly resolved by not executing the entry
                # in the first place.
                if option.empty:
                    is_empty = True
                    contracts.add((contract, order, 0))
                    continue
                #
                if order == Order.BTC.name:
                    ask = option['ask'].values[0]
                    current_price -= ask
                    contracts.add((contract, order, -ask))
                else:
                    bid = option['bid'].values[0]
                    current_price += bid
                    contracts.add((contract, order, bid))
                flt = leg.exit_filter
                option = flt(option)
                if not option.empty:
                    filters_exit = True
            if is_empty or filters_exit or self._is_past_threshold(
                    current_price, old_price):
                exits.append((contracts, current_price))
        return exits

    def _apply_conditions(self, dfs):
        """Applies conditions on the specified legs."""

        for condition in self.conditions:
            condition_idx = None
            for df in dfs:
                df.set_index(condition.fields, inplace=True)
                if condition_idx is not None:
                    condition_idx = condition_idx.intersection(df.index)
                else:
                    condition_idx = df.index

            for i in range(len(dfs)):
                dfs[i] = dfs[i].loc[condition_idx]
                dfs[i].reset_index(inplace=True)

        for i in range(len(dfs)):
            dfs[i].columns = pd.MultiIndex.from_product(
                [["leg_{}".format(i + 1)], dfs[i].columns])

        return dfs

    def _is_past_threshold(self, current_price, old_price):
        current_abs = abs(current_price)
        old_abs = abs(old_price)
        return (current_abs <= self.exit_thresholds[0] * old_abs) or (
            current_abs >= self.exit_thresholds[1] * old_abs)

    def __repr__(self):
        return "Strategy(legs={}, conditions={})".format(
            self.legs, self.conditions)
