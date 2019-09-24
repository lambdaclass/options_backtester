# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from pprint import pformat
from scrapy import Item, Field


class DataItem(Item):
    start_date = Field()
    end_date = Field()
    symbol = Field()
    symbol_path = Field()
    filename = Field()
    data = Field()

    # Used to cleanup old files
    old_files = Field()

    def __repr__(self):
        # Avoids writing `data` field to log

        return pformat({
            'symbol': self['symbol'],
            'start_date': self['start_date'],
            'end_date': self['end_date'],
            'symbol_path': self['symbol_path'],
            'filename': self['filename']
        })
