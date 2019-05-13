import argparse
import os
import logging
from .backtester import run
from .utils import get_data_dir

parser = argparse.ArgumentParser(prog="backtester.py")
parser.add_argument(
    "-t", "--symbols", nargs="+", help="Symbols to fetch", required=True)
parser.add_argument("-s", "--scraper", choices=["cboe"])
args = parser.parse_args()

data_dir = get_data_dir()
spx_data = os.path.join(data_dir, "SPX_2008-2018.csv")
run(spx_data)
