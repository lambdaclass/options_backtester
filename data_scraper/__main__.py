import logging.config
import os
import argparse

from data_scraper import cboe, tiingo, backup

parser = argparse.ArgumentParser(prog="data_scraper.py")
parser.add_argument("-t", "--symbols", nargs="+", help="Symbols to fetch")
parser.add_argument(
    "-s",
    "--scraper",
    choices=["cboe", "tiingo"],
    default="cboe",
    help="Scraper to use")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable logging")
parser.add_argument(
    "-a",
    "--aggregate",
    action="store_true",
    help="Aggregate daily data files")
parser.add_argument(
    "-b", "--backup", action="store_true", help="Backup files in S3 bucket")

args = parser.parse_args()
module_dir = os.path.join(os.getcwd(), os.path.dirname(__file__))

if args.verbose:
    config_file = os.path.realpath(os.path.join(module_dir, "logconfig.ini"))
    logging.config.fileConfig(fname=config_file)

if args.aggregate:
    if args.symbols:
        cboe.aggregate_monthly_data(args.symbols)
    else:
        cboe.aggregate_monthly_data()
elif args.backup:
    backup.backup_data()
else:
    if args.scraper == "tiingo":
        scraper = tiingo
    else:
        scraper = cboe

    if args.symbols:
        scraper.fetch_data(args.symbols)
    else:
        scraper.fetch_data()
