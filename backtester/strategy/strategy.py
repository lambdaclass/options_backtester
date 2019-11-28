from collections import namedtuple

import pandas as pd

from backtester.datahandler import Schema
from backtester.option import Direction
from .strategy_leg import StrategyLeg
from .signal import Signal, get_order

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

            # exit_legs = self._filter_legs(group, signal=Signal.EXIT)
            # exit_df = pd.concat(exit_legs, axis=1)
            exit_df = self._filter_exits(data, bt.inventory,
                                         ['leg_1', 'leg_2'])
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

    def _filter_exits(self, data, inventory, legs):
        exits = []
        for index, row in inventory.iterrows():
            old_price = 0
            current_price = 0
            contracts = set()
            for leg in legs:
                contract = row[leg]['contract']
                order = get_order(~leg.direction, Signal.EXIT)
                contracts.add((contract, order))
                old_price += row[leg]['price']
                option = data[data['optionroot'] == contract]
                if order[0] == 'B':
                    current_price -= option['ask']
                else:
                    current_price += option['bid']
            if (current_price <= 0.8 * old_price) & (current_price >=
                                                     1.2 * old_price):
                exits.append((contracts, current_price))
            else:
                # Filter the data according to the exit filters and append to exits the contracts that need to exit
                pass
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

    def __repr__(self):
        return "Strategy(legs={}, conditions={})".format(
            self.legs, self.conditions)
