# -*- coding: utf-8 -*-
from datetime import date, datetime
import os
from urllib.parse import urlencode

import scrapy
from scrapy.exceptions import DropItem
from scrapy.loader import ItemLoader
from scrapy.http import Request

from scraper import utils
from scraper.items import DataItem


class TiingoSpider(scrapy.Spider):
    name = 'tiingo'
    allowed_domains = ['tiingo.com']
    spider_path = utils.create_spider_path(name)

    custom_settings = {
        'ITEM_PIPELINES': {
            'scraper.pipelines.ValidateAndMergeData': 300,
            'scraper.pipelines.SaveDataPipeline': 500,
            'scraper.pipelines.CleanupFiles': 600
        },
        'SPIDER_DATA_PATH':
        spider_path,
        'FEED_URI':
        os.path.join(
            spider_path, 'tiingo_feed',
            '{}_feed_{}.csv'.format(name,
                                    datetime.now().strftime('%Y%m%d%H%M%S')))
    }

    def __init__(self, symbols_file=None, api_key=None, *args, **kwargs):
        super(TiingoSpider, self).__init__(*args, **kwargs)
        self.api_key = api_key or os.environ['TIINGO_API_KEY']

        with open(symbols_file, 'r') as f:
            self.symbols = [symbol.rstrip('\n').upper() for symbol in f]

        # Fetch data from 1990-01-01 to current day
        self.start_date = date(1990, 1, 1).strftime('%Y-%m-%d')
        self.end_date = date.today().strftime('%Y-%m-%d')

    def start_requests(self):
        url = 'https://api.tiingo.com/tiingo/daily/{symbol}/prices?'

        params = {
            'startDate': self.start_date,
            'endDate': self.end_date,
            'format': 'csv'
        }

        headers = {
            'Content-Type': 'text/csv',
            'Authorization': 'Token ' + self.api_key
        }

        return [
            Request(url.format(symbol=symbol) + urlencode(params),
                    headers=headers,
                    callback=self.parse_response,
                    meta={'loader': self._build_loader(symbol)})
            for symbol in self.symbols
        ]

    def parse_response(self, response):
        loader = response.meta['loader']

        if response.status != 200:
            symbol, = loader.get_collected_values('symbol')
            self.logger.error(
                'Symbol %s not found. API returned status code %i', symbol,
                response.status)
            raise DropItem()

        loader.add_value('end_date', datetime.now().isoformat())
        loader.add_value('data', response.text)

        return loader.load_item()

    def _build_loader(self, symbol):
        loader = ItemLoader(item=DataItem())
        loader.add_value('symbol', symbol)
        loader.add_value('symbol_path', symbol)
        loader.add_value('start_date', datetime.now().isoformat())
        loader.add_value('filename',
                         symbol + date.today().strftime('%Y%m%d') + '.csv')

        return loader
