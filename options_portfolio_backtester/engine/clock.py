"""Trading clock — date iteration and rebalance scheduling."""

from __future__ import annotations

from typing import Generator

import pandas as pd


class TradingClock:
    """Generates (date, stocks_df, options_df) tuples for the backtest loop.

    Handles daily/monthly iteration and rebalance scheduling.
    """

    def __init__(
        self,
        stocks_data: pd.DataFrame,
        options_data: pd.DataFrame,
        stocks_date_col: str = "date",
        options_date_col: str = "quotedate",
        monthly: bool = False,
    ) -> None:
        self.stocks_data = stocks_data
        self.options_data = options_data
        self.stocks_date_col = stocks_date_col
        self.options_date_col = options_date_col
        self.monthly = monthly

    def iter_dates(self) -> Generator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame], None, None]:
        """Iterate over trading dates, yielding (date, stocks, options) per step."""
        if self.monthly:
            stocks_iter = self._monthly_iter(self.stocks_data, self.stocks_date_col)
            options_iter = self._monthly_iter(self.options_data, self.options_date_col)
        else:
            stocks_iter = self.stocks_data.groupby(self.stocks_date_col)
            options_iter = self.options_data.groupby(self.options_date_col)

        for (date, stocks), (_, options) in zip(stocks_iter, options_iter):
            yield date, stocks, options

    def rebalance_dates(self, freq: int) -> pd.DatetimeIndex:
        """Compute rebalance dates using business-month-start frequency.

        Args:
            freq: Number of business months between rebalances.

        Returns:
            DatetimeIndex of rebalance dates present in the data.
        """
        if freq <= 0:
            return pd.DatetimeIndex([])

        dates = pd.DataFrame(
            self.options_data[[self.options_date_col, "volume"]]
        ).drop_duplicates(self.options_date_col).set_index(self.options_date_col)

        return pd.to_datetime(
            dates.groupby(pd.Grouper(freq=f"{freq}BMS"))
            .apply(lambda x: x.index.min())
            .values
        )

    @staticmethod
    def _monthly_iter(data: pd.DataFrame, date_col: str):
        return (
            data.groupby(pd.Grouper(key=date_col, freq="MS"))
            .apply(lambda g: g[g[date_col] == g[date_col].min()])
            .reset_index(drop=True)
            .groupby(date_col)
        )

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.options_data[self.options_date_col].unique())
