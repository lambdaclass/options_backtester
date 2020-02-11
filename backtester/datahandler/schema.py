class Schema:
    """Data schema class.
    Used provide uniform access to fields in the data set.
    """

    stock_columns = [
        "symbol", "date", "open", "close", "high", "low", "volume", "adjClose", "adjHigh", "adjLow", "adjOpen",
        "adjVolume", "divCash", "splitFactor"
    ]

    option_columns = [
        "underlying", "underlying_last", "date", "contract", "type", "expiration", "strike", "bid", "ask", "volume",
        "open_interest"
    ]

    def stocks():
        """Builder method that returns a `Schema` with default mappings for stocks"""
        mappings = {key: key for key in Schema.stock_columns}
        return Schema(mappings)

    def options():
        """Builder method that returns a `Schema` with default mappings for options"""
        mappings = {key: key for key in Schema.option_columns}
        return Schema(mappings)

    def __init__(self, mappings):
        assert all((key in mappings for key in Schema.stock_columns)) or all(
            (key in mappings for key in Schema.option_columns))

        self._mappings = mappings

    def update(self, mappings):
        """Update schema according to given `mappings`"""
        self._mappings.update(mappings)
        return self

    def __contains__(self, key):
        """Returns True if key is in schema"""
        return key in self._mappings.keys()

    def __getattr__(self, key):
        """Returns Field object used to build Filters"""
        return Field(key, self._mappings[key])

    def __setitem__(self, key, value):
        self._mappings[key] = value

    def __getitem__(self, key):
        """Returns mapping of given `key`"""
        return self._mappings[key]

    def __iter__(self):
        return iter(self._mappings.items())

    def __repr__(self):
        return "Schema({})".format([Field(k, m) for k, m in self._mappings.items()])

    def __eq__(self, other):
        return self._mappings == other._mappings


class Field:
    """Encapsulates data fields to build filters used by strategies"""

    __slots__ = ("name", "mapping")

    def __init__(self, name, mapping):
        self.name = name
        self.mapping = mapping

    def _create_filter(self, op, other):
        if isinstance(other, Field):

            query = Field._format_query(self.mapping, op, other.mapping)
        else:
            query = Field._format_query(self.mapping, op, other)
        return Filter(query)

    def _combine_fields(self, op, other, invert=False):
        if isinstance(other, Field):
            name = Field._format_query(self.name, op, other.name, invert)
            mapping = Field._format_query(self.mapping, op, other.mapping, invert)
        elif isinstance(other, (int, float)):
            name = Field._format_query(self.name, op, other, invert)
            mapping = Field._format_query(self.mapping, op, other, invert)
        else:
            raise TypeError

        return Field(name, mapping)

    def _format_query(left, op, right, invert=False):
        if invert:
            left, right = right, left
        query = "{left} {op} {right}".format(left=left, op=op, right=right)
        return query

    def __add__(self, value):
        return self._combine_fields("+", value)

    def __radd__(self, value):
        return self._combine_fields("+", value, invert=True)

    def __sub__(self, value):
        return self._combine_fields("-", value)

    def __rsub__(self, value):
        return self._combine_fields("-", value, invert=True)

    def __mul__(self, value):
        return self._combine_fields("*", value)

    def __rmul__(self, value):
        return self._combine_fields("*", value, invert=True)

    def __truediv__(self, value):
        return self._combine_fields("/", value)

    def __rtruediv__(self, value):
        return self._combine_fields("/", value, invert=True)

    def __lt__(self, value):
        return self._create_filter("<", value)

    def __le__(self, value):
        return self._create_filter("<=", value)

    def __gt__(self, value):
        return self._create_filter(">", value)

    def __ge__(self, value):
        return self._create_filter(">=", value)

    def __eq__(self, value):
        if isinstance(value, str):
            value = "'{}'".format(value)
        return self._create_filter("==", value)

    def __ne__(self, value):
        return self._create_filter("!=", value)

    def __repr__(self):
        return "Field(name='{}', mapping='{}')".format(self.name, self.mapping)


class Filter:
    """This class determines entry/exit conditions for strategies"""

    __slots__ = ("query")

    def __init__(self, query):
        self.query = query

    def __and__(self, other):
        """Returns logical *and* between `self` and `other`"""
        assert isinstance(other, Filter)
        new_query = "({}) & ({})".format(self.query, other.query)
        return Filter(query=new_query)

    def __or__(self, other):
        """Returns logical *or* between `self` and `other`"""
        assert isinstance(other, Filter)
        new_query = "(({}) | ({}))".format(self.query, other.query)
        return Filter(query=new_query)

    def __invert__(self):
        """Negates filter"""
        return Filter("!({})".format(self.query))

    def __call__(self, data):
        """Returns dataframe of filtered data"""
        return data.eval(self.query)

    def __repr__(self):
        return "Filter(query='{}')".format(self.query)
