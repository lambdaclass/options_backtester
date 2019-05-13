import os
from datetime import date, timedelta
import pandas as pd
import pandas_datareader.data as web
from backtester.utils import get_data_dir


def create_test_data(data_dir, filename="SPX_2008-2018.csv"):
    """Create test data set with 10 years of SPX"""

    spx_dir = os.path.join(data_dir, "allspx")
    test_file = os.path.join(data_dir, filename)

    with open(test_file, "w+") as f:
        f.write("date,price\n")

    for year in range(2008, 2019):
        filename = "SPX_{}.csv".format(year)
        year_df = pd.read_csv(os.path.join(spx_dir, filename))
        grouped = year_df.groupby("quotedate").first()
        grouped.to_csv(
            test_file, mode="a", columns=["underlying_last"], header=False)


def create_synthetic_data(data_dir, filename="synthetic_data.csv"):
    """Create an synthetic data set with known statistics.
    Price goes from 1 to 2000.
    Mean = 1000.5
    % Price = 1999"""

    synth_file = os.path.join(data_dir, filename)

    day = date(1970, 1, 1)
    with open(synth_file, "w+") as f:
        f.write("date,price\n")
        for i in range(1, 2001):
            line = "{},{}\n".format(day.strftime("%m/%d/%Y"), i)
            f.write(line)
            day += timedelta(days=1)


def fetch_balanced_data(data_dir,
                        filename="balanced_2015.csv",
                        start=None,
                        end=None):
    """Downloads daily data from `start` til `end` from IEX.

    Symbols
    -------
    VOO: VANGUARD IX FUN/S&P 500 ETF
    GLD: SPDR Gold Trust
    VNQ: VANGUARD IX FUN/RL EST IX FD ETF
    VNQI: VANGUARD INTL E/GLB EX-US RL EST IX
    TLT: iShares Barclays 20+ Yr Treas.Bond
    TIP: iShares TIPS Bond ETF
    BNDX: VANGUARD CHARLO/TOTAL INTL BD ETF
    EEM: iShares MSCI Emerging Markets Indx
    RJI: Rogers International Commodity Index
    """

    if not start or not end:
        start = date(2015, 1, 1)
        end = date(2015, 12, 31)

    symbols = ["VOO", "GLD", "VNQ", "VNQI", "TLT", "TIP", "BNDX", "EEM", "RJI"]

    # Write headers
    full_path = os.path.join(data_dir, filename)
    with open(full_path, "w+") as f:
        f.write("date,symbol,open,high,low,close,volume\n")

    columns = ["symbol", "open", "high", "low", "close", "volume"]
    for symbol in symbols:
        data = web.DataReader(symbol, "iex", start, end)
        data["symbol"] = symbol
        data.to_csv(
            full_path,
            mode="a",
            index_label="date",
            columns=columns,
            header=False)


if __name__ == "__main__":
    data_dir = get_data_dir()
    create_test_data(data_dir)
    create_synthetic_data(data_dir)
    fetch_balanced_data(data_dir)
