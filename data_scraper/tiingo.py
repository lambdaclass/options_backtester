import logging
import os
from datetime import date

import pandas as pd
import pandas_datareader as pdr
import tenacity

from . import utils, validation
from .notifications import send_report

logger = logging.getLogger(__name__)

# Default symbols to fetch
assets = [
    "VTSMX", "VFINX", "VIVAX", "VIGRX", "VIMSX", "VMVIX", "VMGIX", "NAESX",
    "VISVX", "VISGX", "BRSIX", "VGTSX", "VTMGX", "VFSVX", "EFV", "VEURX",
    "VPACX", "VEIEX", "VFISX", "VFITX", "IEF", "VUSTX", "VBMFX", "VIPSX",
    "PIGLX", "PGBIX", "VFSTX", "LQD", "VWESX", "VWEHX", "VWSTX", "VWITX",
    "VWLTX", "VGSIX", "GLD", "PSAU", "GSG"
]


def fetch_data(symbols=assets):
    """Fetches historical data for given symbols from Tiingo"""
    api_key = utils.get_environment_var("TIINGO_API_KEY")
    symbols = [symbol.upper() for symbol in symbols]
    done = 0
    failed = []

    for symbol in symbols:
        try:
            symbol_data = pdr.get_data_tiingo(symbol, api_key=api_key)
        except ConnectionError as ce:
            msg = "Unable to connect to api.tiingo.com when fetching symbol {}".format(
                symbol)
            logger.error(msg, exc_info=True)
            raise ce
        except TypeError:
            # pandas_datareader raises TypeError when fetching invalid symbol
            msg = "Attempted to fetch invalid symbol {}".format(symbol)
            logger.error(msg, exc_info=True)
        except Exception:
            msg = "Error fetching symbol {}".format(symbol)
            logger.error(msg, exc_info=True)
        else:
            _save_data(symbol, symbol_data.reset_index())
            done += 1
    retry_failure(failed, done)
    send_report(done, failed, __name__)


# if a symbol failes to scrape try again exponentialy
@tenacity.retry(wait=tenacity.wait_exponential(multiplier=300),
                stop=tenacity.stop_after_attempt(10),
                retry=tenacity.retry_if_exception_type(IOError))
def retry_failure(failed, done):
    api_key = utils.get_environment_var("TIINGO_API_KEY")
    for symbol in failed:
        try:
            symbol_data = pdr.get_data_tiingo(symbol, api_key=api_key)
        except ConnectionError as ce:
            msg = "Unable to connect to api.tiingo.com when fetching symbol {}".format(
                symbol)
            logger.error(msg, exc_info=True)
            raise ce
        except TypeError:
            # pandas_datareader raises TypeError when fetching invalid symbol
            failed.append(symbol)
            msg = "Attempted to fetch invalid symbol {}".format(symbol)
            logger.error(msg, exc_info=True)
        except Exception:
            msg = "Error fetching symbol {}".format(symbol)
            logger.error(msg, exc_info=True)
        else:
            _save_data(symbol, symbol_data)
            done += 1
            failed.remove(symbol)


def _save_data(symbol, symbol_df):
    """Saves the contents of `symbol_df` to
    `$SAVE_DATA_PATH/tiingo/{symbol}/{symbol}_{%date}.csv`"""
    filename = date.today().strftime(symbol + "_%Y%m%d.csv")

    save_data_path = utils.get_save_data_path()
    symbol_dir = os.path.join(save_data_path, "tiingo", symbol)

    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)
        logger.debug("Symbol dir %s created", symbol_dir)
    file_path = os.path.join(symbol_dir, filename)

    if os.path.exists(file_path) and validation.file_hash_matches_data(
            file_path, symbol_df.to_csv()):
        logger.debug("File %s already downloaded", file_path)
    else:
        expected_columns = [
            "symbol", "date", "close", "high", "low", "open", "volume",
            "adjClose", "adjHigh", "adjLow", "adjOpen", "adjVolume", "divCash",
            "splitFactor"
        ]

        if validation.validate_historical_dates(
                symbol, symbol_df["date"]) and validation.validate_columns(
                    expected_columns, symbol_df.columns):
            merged_df = _merge(symbol, symbol_df)
            pattern = symbol + "_*"
            utils.remove_files(symbol_dir, pattern, logger)

            merged_df.to_csv(file_path, index=False)
            logger.debug("Saved symbol data as %s", file_path)


def _merge(symbol, symbol_df):
    """Merge `symbol_df` with previous data file."""
    save_data_path = utils.get_save_data_path()
    symbol_dir = os.path.join(save_data_path, "tiingo", symbol)
    files = os.listdir(symbol_dir)
    if len(files) == 0:
        return symbol_df

    last_file = sorted(files)[-1]
    old_df = pd.read_csv(os.path.join(symbol_dir, last_file),
                         parse_dates=["date"],
                         index_col="date")
    symbol_df.index = symbol_df["date"]
    sequence = [
        "symbol", "date", "adjClose", "adjHigh", "adjLow", "adjOpen",
        "adjVolume", "close", "divCash", "high", "low", "open", "splitFactor",
        "volume"
    ]
    symbol_df = symbol_df.reindex(columns=sequence)
    diffs = old_df.index.difference(symbol_df.index)

    if diffs.empty:
        return symbol_df
    else:
        msg = """Old data included dates not present in scraped file for symbol {}
            Merged new data with previous file.""".format(symbol)
        logger.error(msg)
        merged_df = pd.concat([symbol_df, old_df.loc[diffs]])
        merged_df.sort_index(inplace=True)
        return merged_df.reset_index()
