import math
from collections import namedtuple
from functools import reduce

import pandas as pd
import numpy as np

from backtester.datahandler import Schema
from backtester.option import Direction
from .strategy_leg import StrategyLeg
from .signal import Signal, get_order

Condition = namedtuple('Condition', 'fields legs tolerance')


class Strategy:
    """Options strategy class.
    Takes in a number of `StrategyLeg`'s (option contracts), and filters that determine
    entry and exit conditions.
    """
    def __init__(self, schema, shares_per_contract=100, initial_capital=1_000_000):
        assert isinstance(schema, Schema)
        self.schema = schema
        self._shares_per_contract = shares_per_contract
        self.initial_capital = initial_capital
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

    def add_exit_thresholds(self, profit_pct=math.inf, loss_pct=math.inf):
        """Adds maximum profit/loss thresholds. Both **must** be >= 0.0

        Args:
            profit_pct (float, optional):   Max profit level. Defaults to math.inf
            loss_pct (float, optional):     Max loss level. Defaults to math.inf
        """
        assert profit_pct >= 0
        assert loss_pct >= 0
        self.exit_thresholds = (profit_pct, loss_pct)

    def filter_entries(self, options, inventory, date):
        """Returns the entry signals chosen by the strategy for the given
        (daily) options.

        Args:
            options (pd.DataFrame): DataFrame of (daily) options
            inventory (pd.DataFrame):   Inventory of current positions
        Returns:
            pd.DataFrame:           Entry signals
        """

        # Remove contracts already in inventory
        inventory_contracts = pd.concat([inventory[leg.name]['contract'] for leg in self.legs])
        subset_options = options[~options[self.schema['contract']].isin(inventory_contracts)]

        return self._filter_legs(subset_options, Signal.ENTRY, date)

    def filter_exits(self, options, inventory, date):
        """Returns the exit signals chosen by the strategy for the given
        (daily) options.

        Args:
            options (pd.DataFrame):     DataFrame of (daily) options
            inventory (pd.DataFrame):   Inventory of current positions
        Returns:
            pd.DataFrame:               Exit signals
        """

        leg_candidates = [self._exit_candidates(l.direction, inventory[l.name], options) for l in self.legs]
        total_costs = sum([l['cost'] for l in leg_candidates])
        threshold_exits = self._filter_thresholds(inventory['totals']['cost'], total_costs)

        filter_mask = []
        for i, leg in enumerate(self.legs):
            flt = leg.exit_filter
            filter_mask.append(flt(leg_candidates[i]))
            fields = self._signal_fields((~leg.direction).value)
            leg_candidates[i] = leg_candidates[i].loc[:, fields.values()]
            leg_candidates[i].columns = pd.MultiIndex.from_product([["leg_{}".format(i + 1)],
                                                                    leg_candidates[i].columns])

        qtys = inventory['totals']['qty']
        dates = [date] * len(inventory)
        totals = pd.DataFrame.from_dict({"cost": total_costs, "qty": qtys, "date": dates})
        totals.columns = pd.MultiIndex.from_product([["totals"], totals.columns])
        leg_candidates.append(totals)
        filter_mask = reduce(lambda x, y: x | y, filter_mask)
        exits_mask = threshold_exits | filter_mask

        exits = pd.concat([l[exits_mask] for l in leg_candidates], axis=1)
        total_costs = total_costs[exits_mask] * exits['totals']['qty']

        return (exits, exits_mask, total_costs)

    def _filter_legs(self, options, signal, date):
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

            df = options[flt(options)]
            fields = self._signal_fields(cost_field)
            subset_df = df.reindex(columns=fields.keys())
            subset_df.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, signal)
            subset_df['order'] = order

            # Change sign of cost for SELL orders
            if leg.direction == Direction.SELL:
                subset_df['cost'] = -subset_df['cost']

            subset_df['cost'] *= self._shares_per_contract

            dfs.append(subset_df.reset_index(drop=True))

        return self._apply_conditions(dfs, date)

    def _signal_fields(self, cost_field):
        fields = {
            self.schema['contract']: 'contract',
            self.schema['underlying']: 'underlying',
            self.schema['expiration']: 'expiration',
            self.schema['type']: 'type',
            self.schema['strike']: 'strike',
            self.schema[cost_field]: 'cost',
            'order': 'order'
        }

        return fields

    def _apply_conditions(self, dfs, date):
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

        if any(df.empty for df in dfs):
            return pd.DataFrame()

        cost = sum(leg["cost"] for leg in dfs)
        # Put qty of contracts to buy/sell in ['totals']['qty']
        qty = self.initial_capital // cost
        qty = np.abs(qty)
        totals = pd.DataFrame.from_dict({"cost": cost, "qty": qty, "date": date})
        totals.columns = pd.MultiIndex.from_product([["totals"], totals.columns])

        for i in range(len(dfs)):
            dfs[i].columns = pd.MultiIndex.from_product([["leg_{}".format(i + 1)], dfs[i].columns])

        dfs.append(totals)

        return pd.concat(dfs, axis=1)

    def _exit_candidates(self, direction, inventory_leg, options):
        """Returns the exit candidates for the given inventory leg with their order and cost (positive for STC orders).

        Args:
            direction (option.Direction):   Direction of the leg for `Signal.EXIT`
            inventory_leg (pd.DataFrame):   DataFrame of contracts in the inventory leg
            options (pd.DataFrame):         Options in the current time step

        Returns:
            pd.DataFrame:                   DataFrame with the cost for the contracts in `inventory_leg`
        """

        # FIXME: Leaky abstraction (inventory schema)
        # This is a left join to ensure that the result has the same length as the inventory. If the contract isn't in
        # the daily data the values will all be NaN and the filters should all yield False.
        fields = self._signal_fields((~direction).value)
        options = options.rename(columns=fields)
        candidates = inventory_leg[['contract']].merge(options, how='left', on='contract')

        order = get_order(direction, Signal.EXIT)
        candidates['order'] = order

        # Change sign of cost for SELL orders
        if ~direction == Direction.SELL:
            candidates['cost'] = -candidates['cost']

        candidates['cost'] *= self._shares_per_contract

        return candidates

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
        return "Strategy(legs={}, conditions={})".format(self.legs, self.conditions)
