# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import os


class SaveDataPipeline():
    def process_item(self, item, spider):

        symbol, = item['symbol']
        symbol_path, = item['symbol_path']
        filename, = item['filename']
        spider_path = spider.settings['SPIDER_DATA_PATH']

        symbol_path = os.path.join(spider_path, symbol_path)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)

        with open(os.path.join(symbol_path, filename), 'w') as file:
            data, = item['data']
            file.write(data)

        return item
