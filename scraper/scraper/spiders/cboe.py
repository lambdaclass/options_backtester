# -*- coding: utf-8 -*-
from datetime import date, datetime
import os

import scrapy
from scrapy.exceptions import DropItem
from scrapy.loader import ItemLoader
from scrapy.http import Request, FormRequest

from scraper import utils
from scraper.items import DataItem


class CBOESpider(scrapy.Spider):
    name = 'cboe'
    allowed_domains = ['cboe.com']
    spider_path = utils.create_spider_path(name)

    custom_settings = {
        'SPIDER_DATA_PATH':
        spider_path,
        'FEED_URI':
        os.path.join(
            spider_path, 'cboe_feed',
            '{}_feed_{}.csv'.format(name,
                                    datetime.now().strftime('%Y%m%d%H%M%S'))),
        'FEED_FORMAT':
        'csv',
        'FEED_EXPORT_ENCODING':
        'utf-8',
        'FEED_EXPORT_FIELDS':
        ['symbol', 'start_date', 'end_date', 'symbol_path', 'filename']
    }

    def __init__(self, symbols_file=None, *args, **kwargs):
        super(CBOESpider, self).__init__(*args, **kwargs)

        with open(symbols_file, 'r') as f:
            self.symbols = [symbol.upper() for symbol in f]

    def start_requests(self):
        return [
            Request('http://www.cboe.com/delayedquote/quote-table-download',
                    callback=self.parse_form)
        ]

    def parse_form(self, response):
        for symbol in self.symbols:
            loader = ItemLoader(item=DataItem())
            loader.add_value('symbol', symbol)
            loader.add_value('symbol_path', symbol + '_daily')
            loader.add_value('start_date', datetime.now().isoformat())
            loader.add_value('filename',
                             symbol + date.today().strftime('%Y%m%d') + '.csv')

            yield FormRequest.from_response(
                response,
                formdata={'ctl00$ContentTop$C005$txtTicker': symbol},
                callback=self.fetch_data,
                dont_filter=True,
                meta={'loader': loader})

    def fetch_data(self, response):
        loader = response.meta['loader']

        content_type = response.headers.get('Content-Type')
        if content_type.startswith(b'text/html'):
            symbol, = loader.get_collected_values('symbol')
            error = response.selector.xpath('//*[@id="lblError"]/text()').get()
            self.logger.error('%s - %s', symbol, error)
            raise DropItem()

        loader.add_value('end_date', datetime.now().isoformat())
        loader.add_value('data', response.text)

        return loader.load_item()
