class Schema:
    """Data schema class (used to run validations)"""

    columns = [
        "underlying", "underlying_last", "date", "contract", "type",
        "expiration", "strike", "bid", "ask", "volume", "open_interest"
    ]

    def canonical():
        """Builder method that returns a `Schema` with default mappings"""
        mappings = {key: key for key in Schema.columns}
        return Schema(mappings)

    def __init__(self, mappings):
        assert all((key in mappings for key in Schema.columns))

        self._mappings = mappings

    def update(self, mappings):
        self._mappings.update(mappings)

    def __setitem__(self, key, value):
        self._mappings[key] = value

    def __getitem__(self, key):
        return Field(key, self._mappings[key])

    def __iter__(self):
        return iter(self._mappings.items())

    def __repr__(self):
        return "Schema({})".format(
            [Field(k, m) for k, m in self._mappings.items()])


class Field:
    """Encapsulates data fields to build filters used by strategies"""

    def __init__(self, name, mapping):
        self.name = name
        self.mapping = mapping

    def _create_filter(self, operator, value):
        query = "{field} {op} {value}".format(field=self.mapping,
                                              op=operator,
                                              value=value)
        return Filter(query)

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
        """Returns filtered dataframe"""
        return data.query(self.query)

    def __repr__(self):
        return "Filter(query='{}')".format(self.query)
