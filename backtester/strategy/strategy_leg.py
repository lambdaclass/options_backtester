from backtester.enums import Type, Direction


class StrategyLeg:
    """Strategy Leg data class"""
    def __init__(self, name, schema, option_type=Type.CALL, direction=Direction.BUY):
        self.name = name
        self.schema = schema
        self.type = option_type
        self.direction = direction

        self._entry_filter = self._base_entry_filter()
        self._exit_filter = self._base_exit_filter()

    @property
    def entry_filter(self):
        """Returns the entry filter"""
        return self._entry_filter

    @entry_filter.setter
    def entry_filter(self, flt):
        """Sets the entry filter"""
        self._entry_filter = self._base_entry_filter() & flt

    @property
    def exit_filter(self):
        """Returns the exit filter"""
        return self._exit_filter

    @exit_filter.setter
    def exit_filter(self, flt):
        """Sets the exit filter"""
        self._exit_filter = self._base_exit_filter() & flt

    def _base_entry_filter(self):
        if self.direction == Direction.BUY:
            return (self.schema.type == self.type.value) & (self.schema.ask > 0)
        else:
            return (self.schema.type == self.type.value) & (self.schema.bid > 0)

    def _base_exit_filter(self):
        return self.schema.type == self.type.value

    def __repr__(self):
        return "StrategyLeg(name={}, type={}, direction={}, entry_filter={}, exit_filter={})".format(
            self.name, self.type, self.direction, self._entry_filter, self._exit_filter)
