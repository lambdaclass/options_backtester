import logging
import os
from datetime import date
from io import StringIO
from itertools import groupby

from bs4 import BeautifulSoup
import requests
import pandas as pd

from . import utils, validation
from .notifications import slack_notification, Status

logger = logging.getLogger(__name__)

url = "http://www.cboe.com/delayedquote/quote-table-download"


def fetch_data(symbols=None):
    """Fetches options data for a given list of symbols"""
    symbols = symbols or _get_all_listed_symbols()
    options = utils.get_module_config("cboe")
    mute_notifications = options.get("mute_notifications", [])

    try:
        form_data = _form_data()
    except requests.ConnectionError as ce:
        msg = "Connection error trying to reach {}".format(url)
        logger.error(msg)
        slack_notification(msg, __name__)
        raise ce
    except Exception as e:
        msg = "Error parsing response"
        logger.error(msg, exc_info=True)
        slack_notification(msg, __name__)
        raise e

    headers = {"Referer": url}
    file_url = "http://www.cboe.com/delayedquote/quotedata.dat"

    symbols = [symbol.upper() for symbol in symbols]
    done, failed = [], []

    for symbol in symbols:
        form_data["ctl00$ContentTop$C005$txtTicker"] = symbol
        try:
            response = requests.post(url,
                                     data=form_data,
                                     headers=headers,
                                     allow_redirects=False)
            symbol_req = requests.get(file_url,
                                      cookies=response.cookies,
                                      headers=headers)
            symbol_data = symbol_req.text
            if symbol_data == "" or symbol_data.startswith(" <!DOCTYPE"):
                raise Exception
        except Exception:
            failed.append(symbol)
            msg = "Error fetching symbol {} data".format(symbol)
            logger.error(msg, exc_info=True)
            if symbol not in mute_notifications:
                slack_notification(msg, __name__)
        else:
            _save_data(symbol, symbol_data)
            done.append(symbol)

    if len(done) > 0:
        msg = "Successfully scraped symbols: " + ", ".join(done)
        slack_notification(msg, __name__, status=Status.Success)
    if len(failed) > 0:
        msg = "Failed to scrape symbols: " + ", ".join(failed)
        slack_notification(msg, __name__, status=Status.Warning)


def aggregate_monthly_data(symbols=None):
    """Aggregate daily snapshots into monthly files and validate data"""
    symbols = symbols or _get_all_listed_symbols()

    save_data_path = utils.get_save_data_path()
    scraper_dir = os.path.join(save_data_path, "cboe")

    symbols = [symbol.upper() for symbol in symbols]

    for symbol in symbols:
        daily_dir = os.path.join(scraper_dir, symbol + "_daily")
        if not os.path.exists(daily_dir):
            msg = "Error aggregating data. Dir {} not found.".format(daily_dir)
            logger.error(msg)
            slack_notification(msg, __name__)
            continue

        monthly_dir = os.path.join(scraper_dir, symbol)

        symbol_files = [
            file for file in os.listdir(daily_dir) if file.endswith(".csv")
        ]

        for month, files in groupby(symbol_files, _monthly_grouper):
            file_names = list(files)
            daily_files = [
                os.path.join(daily_dir, name) for name in file_names
            ]
            try:
                symbol_df = concatenate_files(daily_files)
            except Exception:
                msg = "Error concatenating daily files for period " + month
                logger.error(msg, exc_info=True)
                slack_notification(msg, __name__)
                continue

            date_range = pd.to_datetime(symbol_df["quotedate"].unique())
            if not validation.validate_dates_in_month(symbol, date_range):
                today = pd.Timestamp.today()
                first_date = date_range[0]
                if first_date.year != today.year or first_date.month != today.month:
                    msg = "Some trading dates where missing for symbol {}".format(
                        symbol)
                    slack_notification(msg, __name__)
                continue

            if not os.path.exists(monthly_dir):
                os.makedirs(monthly_dir)
                logger.debug("Symbol dir %s created", monthly_dir)

            file_name = _monthly_filename(file_names)
            monthly_file = os.path.join(monthly_dir, file_name)
            symbol_df.to_csv(monthly_file, index=False)

            if not validation.validate_aggregate_file(monthly_file,
                                                      daily_files):
                utils.remove_file(monthly_file)
                msg = "Data in {} differs from the daily files".format(
                    monthly_file)
                logger.error(msg)
                slack_notification(msg, __name__)
                continue

            logger.debug("Saved monthly data %s", monthly_file)

            for file in daily_files:
                utils.remove_file(file, logger)


def _get_all_listed_symbols():
    """Returns array of all listed symbols.
    http://www.cboe.com/publish/scheduledtask/mktdata/cboesymboldir2.csv
    """
    current_dir = os.path.join(os.getcwd(), os.path.dirname(__file__))
    symbols_file = os.path.realpath(
        os.path.join(current_dir, "cboesymboldir2.csv"))
    symbols_df = pd.read_csv(symbols_file, skiprows=1)
    return symbols_df["Stock Symbol"].array


def concatenate_files(files):
    """Returns a dataframe of the concatenated data from `files`."""
    df_generator = (pd.read_csv(file) for file in sorted(files))
    return pd.concat(df_generator, ignore_index=True)


def _form_data():
    """Return validation form data"""
    homepage = requests.get(url)
    soup = BeautifulSoup(homepage.content, "lxml")
    data = {
        "__VIEWSTATE": soup.select_one("#__VIEWSTATE")["value"],
        "__EVENTVALIDATION": soup.select_one("#__EVENTVALIDATION")["value"]
    }
    return data


def _save_data(symbol, symbol_data):
    """Saves the contents of `symbol_data` to
    `$SAVE_DATA_PATH/cboe/{symbol}_daily/{symbol}_{%date}.csv`
    """
    filename = date.today().strftime(symbol + "_%Y%m%d.csv")

    save_data_path = utils.get_save_data_path()
    symbol_dir = os.path.join(save_data_path, "cboe", symbol + "_daily")

    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)
        logger.debug("Symbol dir %s created", symbol_dir)
    file_path = os.path.join(symbol_dir, filename)

    if os.path.exists(file_path) and validation.file_hash_matches_data(
            file_path, symbol_data):
        logger.debug("File %s already downloaded", file_path)
    else:
        daily_df = _wrangle_data(symbol, symbol_data)
        daily_df.to_csv(file_path, index=False)
        logger.debug("Saved daily symbol data as %s", file_path)


def _wrangle_data(symbol, symbol_data):
    """Returns a properly formated (_tidy_) dataframe"""
    string_data = StringIO(symbol_data)
    first_line = string_data.readline()
    spot_price = float(first_line.split(",")[-2])
    quote_date = date.today().strftime("%m/%d/%Y")

    data = pd.read_csv(string_data, skiprows=1)
    call_columns = [
        "Calls", "Expiration Date", "Strike", "Last Sale", "Net", "Bid", "Ask",
        "Vol", "Open Int", "IV", "Delta", "Gamma"
    ]
    calls = data[call_columns]

    put_columns = [
        "Puts", "Expiration Date", "Strike", "Last Sale.1", "Net.1", "Bid.1",
        "Ask.1", "Vol.1", "Open Int.1", "IV.1", "Delta.1", "Gamma.1"
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

    return merged


def _monthly_grouper(filename):
    """Returns `{year}{month}` string. Used to group files by month."""
    basename = filename.split(".")[0]
    file_date = basename.split("_")[1]
    return file_date[:-2]


def _monthly_filename(filenames):
    """Returns filename of monthly aggregate file in the form
    `{symbol}_{start_date}_to_{end_date}.csv`
    """
    sorted_files = list(sorted(filenames))
    first_file = sorted_files[0]
    last_file = sorted_files[-1]
    last_day = last_file.split(".")[0][-8:]  # Get only the date
    file_name = first_file.split(".")[0] + "_to_" + last_day + ".csv"
    return file_name
