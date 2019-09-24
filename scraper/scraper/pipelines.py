# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import io
import os

import pandas as pd
from scrapy.exceptions import DropItem

from scraper.utils import file_hash_matches_data
from datetime import date


class SaveDataPipeline:
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


class ValidateAndMergeData:
    """Tiingo data validation and merging pipline"""

    expected_columns = {
        'date', 'close', 'high', 'low', 'open', 'volume', 'adjClose',
        'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'
    }

    def process_item(self, item, spider):
        spider_path = spider.settings['SPIDER_DATA_PATH']
        symbol, = item['symbol']
        symbol_path, = item['symbol_path']
        filename, = item['filename']
        item.setdefault('old_files', [])

        data, = item['data']
        line, _ = data.split('\n', maxsplit=1)
        columns = set(line.split(','))

        if not ValidateAndMergeData.expected_columns == columns:
            spider.logger.error(
                'Invalid columns.\nExpected: %s\nReceived: %s',
                ' | '.join(ValidateAndMergeData.expected_columns),
                ' | '.join(columns))
            raise DropItem()

        symbol_dir = os.path.join(spider_path, symbol_path)
        if os.path.exists(symbol_dir) and os.listdir(symbol_dir):
            files = os.listdir(symbol_dir)
            previous_file = sorted(files)[-1]
            file_path = os.path.join(symbol_dir, previous_file)

            if previous_file == filename:
                if file_hash_matches_data(file_path, data):
                    spider.logger.debug(
                        'File {} is already downloaded'.format(filename))
                    raise DropItem()
                else:
                    os.rename(file_path,
                              os.path.join(symbol_dir, previous_file + '.old'))
                    files.remove(previous_file)
                    previous_file += '.old'
                    files.append(previous_file)

            item['old_files'] = files
            previous_df = pd.read_csv(file_path, index_col='date')
            symbol_df = pd.read_csv(io.StringIO(data), index_col='date')
            diffs = previous_df.index.difference(symbol_df.index)

            if not diffs.empty:
                msg = 'Merged new data for symbol {} with previous file {}'.format(
                    symbol, previous_file)
                spider.logger.warning(msg)
                merged_df = pd.concat([symbol_df, previous_df.loc[diffs]])
                merged_df.sort_index(inplace=True)
                item['data'] = [merged_df.to_csv()]

        return item


class CleanupFiles:
    """Remove old files"""
    def process_item(self, item, spider):
        spider_path = spider.settings['SPIDER_DATA_PATH']
        symbol_path, = item['symbol_path']
        old_files = item['old_files']

        for file in old_files:
            spider.logger.debug('Removed {}'.format(file))
            file_path = os.path.join(spider_path, symbol_path, file)
            os.remove(file_path)

        return item


class FormatData:
    """CBOE data formatting"""
    def process_item(self, item, spider):
        item_data, = item['data']
        symbol, = item['symbol']
        string_data = io.StringIO(item_data)
        first_line = string_data.readline()
        spot_price = float(first_line.split(",")[-2])
        quote_date = date.today().strftime("%m/%d/%Y")

        data = pd.read_csv(string_data, skiprows=1)
        call_columns = [
            "Calls", "Expiration Date", "Strike", "Last Sale", "Net", "Bid",
            "Ask", "Vol", "Open Int", "IV", "Delta", "Gamma"
        ]
        calls = data[call_columns]

        put_columns = [
            "Puts", "Expiration Date", "Strike", "Last Sale.1", "Net.1",
            "Bid.1", "Ask.1", "Vol.1", "Open Int.1", "IV.1", "Delta.1",
            "Gamma.1"
        ]
        puts = data[put_columns]

        renamed_columns = [
            "optionroot", "expiration", "strike", "last", "net", "bid", "ask",
            "volume", "openinterest", "impliedvol", "delta", "gamma"
        ]
        calls.columns = renamed_columns
        calls.insert(loc=1, column="type", value="call")
        puts.columns = renamed_columns
        puts.insert(loc=1, column="type", value="put")

        merged = pd.concat([calls, puts])
        merged.insert(loc=0, column="underlying", value=symbol)
        merged.insert(loc=1, column="underlying_last", value=spot_price)
        merged.insert(loc=2, column="exchange", value="CBOE")
        merged.insert(loc=6, column="quotedate", value=quote_date)

        spider.logger.warning(merged)
        item['data'] = [merged.to_csv(index=False)]

        return item
