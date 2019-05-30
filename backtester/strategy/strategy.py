from ..option import OptionContract
from ..datahandler import Filter


class Strategy:
    """Options strategy class.
    Takes in a number of `legs` (option contracts), and filters that determine
    entry and exit conditions.
    """

    def __init__(self, data, entry_filter, exit_filter, legs=[]):
        assert all((isinstance(leg, OptionContract) for leg in legs))
        assert isinstance(entry_filter, Filter)
        assert isinstance(exit_filter, Filter)

        self.data = data
        self.entry = entry_filter
        self.exit = exit_filter
        self.legs = legs

    def add_leg(self, leg):
        """Adds leg to the strategy"""
        self.legs.append(leg)
        return self

    def remove_leg(self, leg_number):
        """Removes leg to the strategy"""
        self.legs.pop(leg_number)
        return self

    def run(self, data):
        """Returns a dataframe of trades executed as a result of
        runnning the strategy on the data.
        """
        entry_query = self.entry(self._data)
        exit_query = self.exit(self._data)

        entry_df = data.query(entry_query)
        exit_df = data.query(exit_query)

        return entry_df.merge(exit_df,
                              on="optionroot",
                              suffixes=("_entry", "_exit"))

    def __repr__(self):
        return "Strategy(entry_filter={}, exit_filter={}, legs={})".format(
            self.entry, self.exit, self.legs)
