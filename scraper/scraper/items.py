# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class DataItem(Item):
    start_date = Field()
    end_date = Field()
    symbol = Field()
    symbol_path = Field()
    filename = Field()
    data = Field()
