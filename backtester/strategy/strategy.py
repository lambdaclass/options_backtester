from collections import namedtuple

import pandas as pd
import numpy as np

from backtester.datahandler import Schema
from backtester.option import Direction
from .strategy_leg import StrategyLeg
from .signal import Signal, get_order, Order

Condition = namedtuple('Condition', 'fields legs tolerance')


class Strategy:
    """Options strategy class.
    Takes in a number of `StrategyLeg`'s (option contracts), and filters that determine
    entry and exit conditions.
    """

    def __init__(self, schema):
        assert isinstance(schema, Schema)
        self.schema = schema
        self.legs = []
        self.conditions = []
        self.exit_thresholds = (0.0, 0.0)

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
        return self

    def add_exit_thresholds(self, profit_pct=0.0, loss_pct=0.0):
        """Adds maximum profit/loss thresholds.

        Args:
            profit_pct (float, optional):   Max profit level. Defaults to 0.0
            loss_pct (float, optional):     Max loss level. Defaults to 0.0
        """
        self.exit_thresholds = (profit_pct, loss_pct)

    def filter_entries(self, options):
        """Returns the entry signals chosen by the strategy for the given
        (daily) options.

        Args:
            options (pd.DataFrame): DataFrame of (daily) options
        Returns:
            pd.DataFrame:           Entry signals
        """
        return self._filter_legs(options, Signal.ENTRY)

    def filter_exits(self, options, inventory):
        """Returns the exit signals chosen by the strategy for the given
        (daily) options.

        Args:
            options (pd.DataFrame):     DataFrame of (daily) options
            inventory (pd.DataFrame):   Inventory of current positions
        Returns:
            pd.DataFrame:               Exit signals
        """

        underlying_col, spot_col = self.schema['underlying'], self.schema[
            'underlying_last']
        underlying_symbols = options.loc[:, (
            underlying_col, spot_col)].drop_duplicates(underlying_col)
        spot_prices = underlying_symbols.set_index(underlying_col).to_dict()

        leg_costs = [
            self._exit_costs(~l.direction, inventory[l.name], options,
                             spot_prices) for l in self.legs
        ]

        total_costs = sum((l['current_cost'] for l in leg_costs))
        threshold_exits = self._filter_thresholds(inventory['cost'],
                                                  total_costs)

        # Only check exits for options in inventory
        subset = options[self.schema['contract']].isin(inventory['contract'])
        options_in_inventory = options[subset]

        exit_df = self._filter_legs(options_in_inventory, Signal.EXIT)
        return total_costs & threshold_exits

    def _filter_legs(self, options, signal):
        """Returns a hierarchically indexed `pd.DataFrame` containing signals for each
        leg in the strategy.

        Args:
            options (pd.DataFrame): DataFrame of (daily) options
            signal (Signal):        Either `Signal.ENTRY` or `Signal.EXIT`

        Returns:
            pd.DataFrame:           DataFrame of signals, with `pd.MultiIndex` columns
        """

        dfs = []
        for leg in self.legs:
            if signal == Signal.ENTRY:
                flt = leg.entry_filter
                cost_field = leg.direction.value
            else:
                flt = leg.exit_filter
                cost_field = (~leg.direction).value

            df = flt(options)
            fields = self._signal_fields(cost_field)
            subset_df = df.loc[:, fields.keys()]
            subset_df.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, signal)
            subset_df["order"] = order.name

            # Change sign of cost for SELL orders
            if leg.direction == Direction.SELL:
                subset_df["cost"] = -subset_df["cost"]

            dfs.append(subset_df.reset_index(drop=True))

        return self._apply_conditions(dfs)

    def _signal_fields(self, cost_field):
        fields = {
            self.schema['contract']: 'contract',
            self.schema['underlying']: 'underlying',
            self.schema['expiration']: 'expiration',
            self.schema['type']: 'type',
            self.schema['strike']: 'strike',
            self.schema[cost_field]: 'cost'
        }

        return fields

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

    def _exit_costs(self, direction, inventory_leg, options, spot_prices):
        """Returns the exit cost (positive for STC orders) for the given inventory leg.

        Args:
            direction (option.Direction):   Direction of the leg for `Signal.EXIT`
            inventory_leg (pd.DataFrame):   DataFrame of contracts in the inventory leg
            options (pd.DataFrame):         Options in the current time step
            spot_prices (dict):             Dictionary mapping underlying symbols to their spot prices

        Returns:
            pd.DataFrame:                   DataFrame with a `current_cost` column with the
            (possibly imputed) cost for the contracts in `inventory_leg`
        """

        options_cost = options[[
            self.schema['contract'], self.schema[direction.value]
        ]]

        # FIXME: Leaky abstraction (inventory schema)
        leg_cost = inventory_leg[['underlying', 'contract', 'cost'
                                  ]].merge(options_cost,
                                           how='left',
                                           left_on='contract',
                                           right_on=self.schema['contract'])

        def calculate_cost(row):
            price = row[self.schema[direction.value]]
            if pd.isna(price):
                # Impute contract price from the difference between spot and strike
                imputed = spot_prices[row['underlying']] - row['strike']
                if row['type'] == 'put':
                    imputed = -imputed

                price = max(imputed, 0)

            return price

        leg_cost['current_cost'] = leg_cost.apply(calculate_cost, axis=1)

        # Change sign of cost for SELL orders
        if direction == Direction.SELL:
            leg_cost['current_cost'] = -leg_cost['current_cost']

        return leg_cost

    def _filter_thresholds(self, entry_cost, current_cost):
        """Returns a `pd.Series` of booleans indicating where profit (loss) levels
        exceed the given thresholds.

        Args:
            entry_cost (pd.Series):     Total _entry_ cost of inventory row
            current_cost (pd.Series):   Present cost of inventory row

        Returns:
            pd.Series:                  Indicator series with `True` for every row that
            exceeds the specified profit (loss) thresholds
        """

        profit_pct, loss_pct = self.exit_thresholds

        excess_return = (current_cost / entry_cost + 1) * -np.sign(entry_cost)
        return (excess_return >= profit_pct) | (excess_return <= -loss_pct)

    def __repr__(self):
        return "Strategy(legs={}, conditions={})".format(
            self.legs, self.conditions)
