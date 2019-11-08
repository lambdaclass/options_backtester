import pandas as pd


class Strangle:
    def __init__(self,
                 underlying,
                 strike,
                 dte,
                 strike_diff,
                 shares_per_contract=100,
                 capital=1000000.0):
        self.underlying = underlying
        self.strike = strike
        self.dte = dte
        self.strike_diff = strike_diff
        self.inventory = set()
        self.shares_per_contract = shares_per_contract
        self.capital = capital

    def execute_entry(self, date, group):
        calls = group.loc[(group.type == 'call')
                          & (group.strike >= self.strike[0]) &
                          (group.strike <= self.strike[1]) &
                          (group.dte >= self.dte[0])
                          & (group.dte <= self.dte[1])]
        puts = group.loc[group.type == 'put']
        merge = calls.merge(puts, on=['dte'], suffixes=('_call', '_put'))
        merge['ask_sum'] = merge['ask_call'] + merge['ask_put']
        merge['strike_diff'] = abs(merge['strike_call'] - merge['strike_put'])
        merge_strangle = merge.loc[merge['strike_diff'] <= self.strike_diff]
        if merge_strangle.empty:
            return
        entry_index = merge_strangle['ask_sum'].idxmin()
        entry = merge_strangle.loc[entry_index]
        cost = sum([entry['ask_sum'] * self.shares_per_contract])
        if cost <= self.capital:
            self.capital -= cost
            self.inventory.add((entry.optionroot_call, entry.dte))
            self.inventory.add((entry.optionroot_put, entry.dte))
            self._update_trade_log(date, entry.optionroot_call,
                                   entry.type_call,
                                   -entry.ask_call * self.shares_per_contract)
            self._update_trade_log(date, entry.optionroot_put, entry.type_put,
                                   -entry.ask_put * self.shares_per_contract)

    def execute_exits(self, inventory, date, group):
        exits = []
        remove_set = set()
        for entry in inventory:
            exit = group.loc[(group.optionroot == entry[0]) & (group.dte == 1)]
            if not exit.empty:
                exits.append(exit)
                remove_set.add(entry)
        for exit in exits:
            profit = exit.bid.values[0] * self.shares_per_contract
            contract = exit.optionroot.values[0]
            type_ = exit.type.values[0]
            self.capital += profit
            self._update_trade_log(date, contract, type_, profit)
        self.inventory.difference_update(remove_set)

    def run(self, data):
        self.trade_log = pd.DataFrame(
            columns=["date", "contract", "type", "profit", "capital"])

        for date, group in self._iter_dates(data):
            self.execute_entry(date, group)
            self.execute_exits(self.inventory, date, group)

        return self.trade_log

    def _update_trade_log(self, date, contract, type_, profit):
        """Adds entry for the given order to `self.trade_log`."""
        self.trade_log.loc[len(
            self.trade_log)] = [date, contract, type_, profit, self.capital]

    def _iter_dates(self, data):
        """Returns `pd.DataFrameGroupBy` with the given underlying and with contracts grouped by date"""
        df = data._data.loc[data._data.underlying == self.underlying]
        return df.groupby(data.schema["date"])
