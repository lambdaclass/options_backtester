import logging
import hashlib

import pandas as pd
import pandas_market_calendars as mcal

from . import cboe
from .notifications import slack_notification

logger = logging.getLogger(__name__)


def file_hash_matches_data(file_path, data):
    file_hash = file_md5(file_path)
    data_md5 = hashlib.md5(data.encode()).hexdigest()
    return file_hash == data_md5


def file_md5(file, chunk_size=4096):
    md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest()


def validate_dates_in_month(symbol, date_range):
    """Compares `date_range` (month) with NYSE trading calendar.
    Returns `True` if there are no missing days.
    """
    # NYSE and CBOE have the same trading calendar
    # https://www.nyse.com/markets/hours-calendars
    # http://cfe.cboe.com/about-cfe/holiday-calendar
    nyse = mcal.get_calendar("NYSE")
    first_date = date_range[0]
    period = pd.Period(year=first_date.year, month=first_date.month, freq="M")
    trading_days = nyse.valid_days(start_date=period.start_time,
                                   end_date=period.end_time)

    # Remove timezone info
    trading_days = trading_days.tz_convert(tz=None)
    missing_days = trading_days.difference(date_range)
    if not missing_days.empty:
        logger.error("Error validating monthly dates. Missing: %s",
                     missing_days)
    return missing_days.empty


def validate_historical_dates(symbol, date_range):
    """Compares `date_range` (any time range) with trading calendar.
    Returns `True` if there are no missing days.
    """
    nyse = mcal.get_calendar("NYSE")
    start_date = date_range.min()
    end_date = date_range.max()
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

    # Remove timezone info
    trading_days = trading_days.tz_convert(tz=None)
    date_range = date_range.dt.tz_convert(tz=None)
    missing_days = trading_days.difference(date_range)

    if not missing_days.empty:
        logger.error("Error validating historical dates. Missing: %s",
                     missing_days)

    return missing_days.empty


def validate_columns(expected, received):
    """Verify that the `received` columns scraped are equal to `expected`"""
    valid = all(expected == received)

    if not valid:
        expected_cols = ", ".join(expected)
        received_cols = ", ".join(received)
        msg = """Columns expected differ from those received.
            Expected: {}
            Received: {}""".format(expected_cols, received_cols)
        logger.error(msg)
        slack_notification(msg, __name__)

    return valid


def validate_aggregate_file(aggregate_file, daily_files):
    """Compares `aggregate_file` with the data from `daily_files`."""
    aggregate_df = pd.read_csv(aggregate_file)
    recreated_df = cboe.concatenate_files(daily_files)

    return aggregate_df.equals(recreated_df)
