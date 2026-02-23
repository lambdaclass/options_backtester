"""BacktestEngine — thin orchestrator composing all framework components.

Replaces the monolithic Backtest class with a clean composition of:
- Data providers (stocks, options)
- Strategy (legs, filters, thresholds)
- Execution (cost model, fill model, sizer, signal selector)
- Portfolio (positions, cash, holdings)
- Risk management (constraints)
- Analytics (trade log, balance sheet)
"""

from __future__ import annotations

from functools import reduce
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import pyprind

from options_backtester.core.types import (
    Direction, OptionType, Order, Signal, Greeks, StockAllocation,
    get_order,
)
from options_backtester.execution.cost_model import TransactionCostModel, NoCosts
from options_backtester.execution.fill_model import FillModel, MarketAtBidAsk
from options_backtester.execution.sizer import PositionSizer, CapitalBased
from options_backtester.execution.signal_selector import SignalSelector, FirstMatch
from options_backtester.portfolio.risk import RiskManager
from options_backtester.portfolio.portfolio import Portfolio, StockHolding
from options_backtester.portfolio.position import OptionPosition, PositionLeg
from options_backtester.engine._dispatch import use_rust, rust

# We still use the original data handler types for backward compat in the engine
from backtester.datahandler.historical_options_data import HistoricalOptionsData
from backtester.datahandler.tiingo_data import TiingoData
from backtester.datahandler.schema import Schema
from backtester.strategy.strategy import Strategy
from backtester.strategy.strategy_leg import StrategyLeg
from backtester.enums import (
    Stock,
    Type as LegacyType,
    Direction as LegacyDirection,
    Signal as LegacySignal,
    Order as LegacyOrder,
    get_order as legacy_get_order,
)


class BacktestEngine:
    """New framework engine — orchestrates backtest with pluggable components.

    This engine uses the ORIGINAL data structures (MultiIndex DataFrames for
    inventory) internally to maintain backward compatibility with existing tests
    and notebooks, but exposes pluggable execution components (cost model, fill
    model, sizer, signal selector, risk manager).

    For a fully new-style engine with Portfolio dataclass, see Phase 7+.
    """

    def __init__(
        self,
        allocation: dict[str, float],
        initial_capital: int = 1_000_000,
        shares_per_contract: int = 100,
        cost_model: TransactionCostModel | None = None,
        fill_model: FillModel | None = None,
        sizer: PositionSizer | None = None,
        signal_selector: SignalSelector | None = None,
        risk_manager: RiskManager | None = None,
        stop_if_broke: bool = False,
    ) -> None:
        assets = ("stocks", "options", "cash")
        total_allocation = sum(allocation.get(a, 0.0) for a in assets)

        self.allocation: dict[str, float] = {}
        for asset in assets:
            self.allocation[asset] = allocation.get(asset, 0.0) / total_allocation

        self.initial_capital = initial_capital
        self.shares_per_contract = shares_per_contract
        self.cost_model = cost_model or NoCosts()
        self.fill_model = fill_model or MarketAtBidAsk()
        self.sizer = sizer or CapitalBased()
        self.signal_selector = signal_selector or FirstMatch()
        self.risk_manager = risk_manager or RiskManager()
        self.stop_if_broke = stop_if_broke

        self.options_budget: Union[Callable[[pd.Timestamp, float], float], float, None] = None
        self._stocks: list[Stock] = []
        self._options_strategy: Strategy | None = None
        self._stocks_data: TiingoData | None = None
        self._options_data: HistoricalOptionsData | None = None

    # -- Properties (same API as original Backtest) --

    @property
    def stocks(self) -> list[Stock]:
        return self._stocks

    @stocks.setter
    def stocks(self, stocks: list[Stock]) -> None:
        assert np.isclose(sum(s.percentage for s in stocks), 1.0, atol=1e-6)
        self._stocks = list(stocks)

    @property
    def options_strategy(self) -> Strategy | None:
        return self._options_strategy

    @options_strategy.setter
    def options_strategy(self, strat: Strategy) -> None:
        self._options_strategy = strat

    @property
    def stocks_data(self) -> TiingoData | None:
        return self._stocks_data

    @stocks_data.setter
    def stocks_data(self, data: TiingoData) -> None:
        self._stocks_schema = data.schema
        self._stocks_data = data

    @property
    def options_data(self) -> HistoricalOptionsData | None:
        return self._options_data

    @options_data.setter
    def options_data(self, data: HistoricalOptionsData) -> None:
        self._options_schema = data.schema
        self._options_data = data

    # -- Main entry point --

    def run(self, rebalance_freq: int = 0, monthly: bool = False,
            sma_days: int | None = None) -> pd.DataFrame:
        """Run the backtest. Returns the trade log DataFrame."""
        assert self._stocks_data, "Stock data not set"
        assert all(
            stock.symbol in self._stocks_data["symbol"].values
            for stock in self._stocks
        ), "Ensure all stocks in portfolio are present in the data"
        assert self._options_data, "Options data not set"
        assert self._options_strategy, "Options Strategy not set"
        assert self._options_data.schema == self._options_strategy.schema

        option_dates = self._options_data["date"].unique()
        stock_dates = self.stocks_data["date"].unique()
        assert np.array_equal(stock_dates, option_dates)

        # Dispatch to Rust full-loop if explicitly opted in via _use_rust_loop
        if (getattr(self, '_use_rust_loop', False)
                and use_rust() and not monthly and not sma_days
                and isinstance(self.cost_model, NoCosts)
                and isinstance(self.fill_model, MarketAtBidAsk)
                and isinstance(self.signal_selector, FirstMatch)
                and not self.risk_manager.constraints
                and self.options_budget is None):
            return self._run_rust(rebalance_freq)

        self._initialize_inventories()
        self.current_cash: float = self.initial_capital
        self._trade_log_parts: list[pd.DataFrame] = []
        initial_balance = pd.DataFrame(
            {"total capital": self.current_cash, "cash": self.current_cash},
            index=[self.stocks_data.start_date - pd.Timedelta(1, unit="day")],
        )
        self._balance_parts: list[pd.DataFrame] = [initial_balance]
        self._peak_value: float = self.initial_capital

        if sma_days:
            self.stocks_data.sma(sma_days)

        dates = (
            pd.DataFrame(self.options_data._data[["quotedate", "volume"]])
            .drop_duplicates("quotedate")
            .set_index("quotedate")
        )
        rebalancing_days: pd.DatetimeIndex = (
            pd.to_datetime(
                dates.groupby(pd.Grouper(freq=f"{rebalance_freq}BMS"))
                .apply(lambda x: x.index.min())
                .values
            )
            if rebalance_freq
            else []
        )

        data_iterator = self._data_iterator(monthly)
        bar = pyprind.ProgBar(len(stock_dates), bar_char="█")

        for date, stocks, options in data_iterator:
            if date in rebalancing_days:
                loc = rebalancing_days.get_loc(date)
                previous_rb_date = rebalancing_days[loc - 1] if loc != 0 else date
                self._update_balance(previous_rb_date, date)
                self._rebalance_portfolio(date, stocks, options, sma_days)

                # Track peak for drawdown risk checks
                total = self._total_capital_estimate(stocks, options)
                self._peak_value = max(self._peak_value, total)

                # stop_if_broke: halt if cash goes negative
                if self.stop_if_broke and self.current_cash < 0:
                    break

            bar.update()

        # Final balance update
        if len(rebalancing_days) > 0:
            self._update_balance(rebalancing_days[-1], self.stocks_data.end_date)

        # Assemble trade_log and balance
        self.trade_log: pd.DataFrame = (
            pd.concat(self._trade_log_parts, ignore_index=True)
            if self._trade_log_parts
            else pd.DataFrame()
        )
        self.balance: pd.DataFrame = pd.concat(self._balance_parts, sort=False)
        for col in self.balance.columns:
            self.balance[col] = pd.to_numeric(self.balance[col], errors="coerce")

        self.balance["options capital"] = (
            self.balance["calls capital"] + self.balance["puts capital"]
        )
        self.balance["stocks capital"] = sum(
            self.balance[stock.symbol] for stock in self._stocks
        )
        first_idx = self.balance.index[0]
        self.balance.loc[first_idx, "stocks capital"] = 0
        self.balance.loc[first_idx, "options capital"] = 0
        self.balance["total capital"] = (
            self.balance["cash"]
            + self.balance["stocks capital"]
            + self.balance["options capital"]
        )
        self.balance["% change"] = self.balance["total capital"].pct_change()
        self.balance["accumulated return"] = (
            1.0 + self.balance["% change"]
        ).cumprod()

        return self.trade_log

    def _run_rust(self, rebalance_freq: int) -> pd.DataFrame:
        """Run the backtest using the Rust full-loop implementation."""
        import math
        import polars as pl

        strategy = self._options_strategy

        # Compute rebalance dates in Python (same BMS logic as the Python loop)
        # to guarantee date parity with the Python path.
        dates_df = (
            pd.DataFrame(self.options_data._data[["quotedate", "volume"]])
            .drop_duplicates("quotedate")
            .set_index("quotedate")
        )
        if rebalance_freq:
            rebalancing_days = pd.to_datetime(
                dates_df.groupby(pd.Grouper(freq=f"{rebalance_freq}BMS"))
                .apply(lambda x: x.index.min())
                .values
            )
            rb_date_strs = [str(d) for d in rebalancing_days]
        else:
            rb_date_strs = []

        config = {
            "allocation": self.allocation,
            "initial_capital": float(self.initial_capital),
            "shares_per_contract": self.shares_per_contract,
            "rebalance_dates": rb_date_strs,
            "legs": [{
                "name": leg.name,
                "entry_filter": leg.entry_filter.query,
                "exit_filter": leg.exit_filter.query,
                "direction": leg.direction.value,
                "type": leg.type.value,
                "entry_sort_col": leg.entry_sort[0] if leg.entry_sort else None,
                "entry_sort_asc": leg.entry_sort[1] if leg.entry_sort else True,
            } for leg in strategy.legs],
            "profit_pct": (
                strategy.exit_thresholds[0]
                if strategy.exit_thresholds[0] != math.inf else None
            ),
            "loss_pct": (
                strategy.exit_thresholds[1]
                if strategy.exit_thresholds[1] != math.inf else None
            ),
            "stocks": [(s.symbol, s.percentage) for s in self._stocks],
        }

        schema_mapping = {
            "contract": self._options_schema["contract"],
            "date": self._options_schema["date"],
            "stocks_date": self._stocks_schema["date"],
            "stocks_symbol": self._stocks_schema["symbol"],
            "stocks_price": self._stocks_schema["adjClose"],
            "underlying": self._options_schema["underlying"],
            "expiration": self._options_schema["expiration"],
            "type": self._options_schema["type"],
            "strike": self._options_schema["strike"],
        }

        opts_pl = pl.from_pandas(self._options_data._data)
        stocks_pl = pl.from_pandas(self._stocks_data._data)

        balance_pl, trade_log_pl, stats = rust.run_backtest_py(
            opts_pl, stocks_pl, config, schema_mapping,
        )

        # Convert trade log from flat columns to MultiIndex
        trade_log_pd = trade_log_pl.to_pandas()
        self.trade_log = self._flat_trade_log_to_multiindex(trade_log_pd)

        # Convert balance
        self.balance = balance_pl.to_pandas()
        if "date" in self.balance.columns:
            self.balance.set_index("date", inplace=True)

        # Add derived columns matching Python output
        self.balance["options capital"] = (
            self.balance.get("calls capital", 0) + self.balance.get("puts capital", 0)
        )
        stock_cols = [s.symbol for s in self._stocks if s.symbol in self.balance.columns]
        self.balance["stocks capital"] = sum(
            self.balance[c] for c in stock_cols
        ) if stock_cols else 0
        self.balance["total capital"] = (
            self.balance["cash"]
            + self.balance["stocks capital"]
            + self.balance["options capital"]
        )
        self.balance["% change"] = self.balance["total capital"].pct_change()
        self.balance["accumulated return"] = (1.0 + self.balance["% change"]).cumprod()

        self.current_cash = stats.get("final_cash", 0.0)
        self._initialize_inventories()
        self._portfolio = Portfolio(initial_cash=self.current_cash)

        return self.trade_log

    def _flat_trade_log_to_multiindex(self, flat_df: pd.DataFrame) -> pd.DataFrame:
        """Convert flat 'leg__field' columns from Rust to MultiIndex DataFrame."""
        if flat_df.empty:
            return pd.DataFrame()
        tuples = []
        for c in flat_df.columns:
            if "__" in c:
                parts = c.split("__", 1)
                tuples.append((parts[0], parts[1]))
            else:
                tuples.append(("", c))
        flat_df.columns = pd.MultiIndex.from_tuples(tuples)
        return flat_df

    # -- Internals (same logic as original, with pluggable components) --

    def _total_capital_estimate(self, stocks: pd.DataFrame,
                                options: pd.DataFrame) -> float:
        """Quick estimate of total capital for risk checks."""
        stock_cap = self._current_stock_capital(stocks)
        options_cap = self._current_options_capital(options)
        return self.current_cash + stock_cap + options_cap

    def _initialize_inventories(self) -> None:
        columns = pd.MultiIndex.from_product(
            [
                [leg.name for leg in self._options_strategy.legs],
                ["contract", "underlying", "expiration", "type", "strike", "cost", "order"],
            ]
        )
        totals = pd.MultiIndex.from_product([["totals"], ["cost", "qty", "date"]])
        self._options_inventory: pd.DataFrame = pd.DataFrame(
            columns=pd.Index(columns.tolist() + totals.tolist())
        )
        self._stocks_inventory: pd.DataFrame = pd.DataFrame(
            columns=["symbol", "price", "qty"]
        )
        # Portfolio dataclass — dual-write alongside legacy DataFrames
        self._portfolio = Portfolio(initial_cash=0.0)

    def _data_iterator(self, monthly: bool):
        if monthly:
            it = zip(self._stocks_data.iter_months(), self._options_data.iter_months())
        else:
            it = zip(self._stocks_data.iter_dates(), self._options_data.iter_dates())
        return (
            (date, stocks, options)
            for (date, stocks), (_, options) in it
        )

    def _rebalance_portfolio(self, date, stocks, options, sma_days):
        self._execute_option_exits(date, options)

        stock_capital = self._current_stock_capital(stocks)
        options_capital = self._current_options_capital(options)
        total_capital = self.current_cash + stock_capital + options_capital

        stocks_allocation = self.allocation["stocks"] * total_capital
        self._stocks_inventory = pd.DataFrame(columns=["symbol", "price", "qty"])
        self.current_cash = stocks_allocation + total_capital * self.allocation["cash"]
        self._buy_stocks(stocks, stocks_allocation, sma_days)

        if self.options_budget is not None:
            budget = self.options_budget
            options_allocation: float = budget(date, total_capital) if callable(budget) else budget
        else:
            options_allocation = self.allocation["options"] * total_capital
        if options_allocation >= options_capital:
            self._execute_option_entries(date, options, options_allocation - options_capital, stocks)
        else:
            to_sell = options_capital - options_allocation
            current_options = self._get_current_option_quotes(options)
            self._sell_some_options(date, to_sell, current_options)

    def _sell_some_options(self, date, to_sell, current_options):
        sold: float = 0
        total_costs = sum([current_options[i]["cost"] for i in range(len(current_options))])
        trade_rows: list[pd.Series] = []
        for exit_cost, (row_index, inventory_row) in zip(
            total_costs, self._options_inventory.iterrows()
        ):
            if exit_cost == 0:
                continue
            if (to_sell - sold > -exit_cost) and (to_sell - sold) > 0:
                qty_to_sell = (to_sell - sold) // exit_cost
                if -qty_to_sell <= inventory_row["totals"]["qty"]:
                    qty_to_sell = (to_sell - sold) // exit_cost
                else:
                    if qty_to_sell != 0:
                        qty_to_sell = -inventory_row["totals"]["qty"]
                if qty_to_sell != 0:
                    trade_log_append = self._options_inventory.loc[row_index].copy()
                    trade_log_append["totals", "qty"] = -qty_to_sell
                    trade_log_append["totals", "date"] = date
                    trade_log_append["totals", "cost"] = exit_cost
                    for i, leg in enumerate(self._options_strategy.legs):
                        trade_log_append[leg.name, "order"] = ~trade_log_append[leg.name, "order"]
                        trade_log_append[leg.name, "cost"] = current_options[i].loc[row_index]["cost"]
                    trade_rows.append(trade_log_append)
                    self._options_inventory.at[row_index, ("totals", "date")] = date
                    self._options_inventory.at[row_index, ("totals", "qty")] += qty_to_sell
                sold += qty_to_sell * exit_cost

        if trade_rows:
            self._trade_log_parts.append(pd.DataFrame(trade_rows))
        self.current_cash += sold - to_sell

    def _current_stock_capital(self, stocks):
        current_stocks = self._stocks_inventory.merge(
            stocks, how="left", left_on="symbol",
            right_on=self._stocks_schema["symbol"],
        )
        return (current_stocks[self._stocks_schema["adjClose"]] * current_stocks["qty"]).sum()

    def _current_options_capital(self, options):
        options_value = self._get_current_option_quotes(options)
        values_by_row: Any = [0] * len(options_value[0])
        if len(options_value[0]) != 0:
            for i in range(len(self._options_strategy.legs)):
                values_by_row += options_value[i]["cost"].values
            total: float = -sum(values_by_row * self._options_inventory["totals"]["qty"].values)
        else:
            total = 0
        return total

    def _buy_stocks(self, stocks, allocation, sma_days):
        stock_symbols = [stock.symbol for stock in self.stocks]
        sym_col = self._stocks_schema["symbol"]
        inventory_stocks = stocks[stocks[sym_col].isin(stock_symbols)]
        stock_percentages = np.array([stock.percentage for stock in self.stocks])
        stock_prices = inventory_stocks[self._stocks_schema["adjClose"]].values

        if sma_days:
            sma_values = inventory_stocks["sma"].values
            qty = np.where(
                sma_values < stock_prices,
                (allocation * stock_percentages) // stock_prices,
                0,
            )
        else:
            qty = (allocation * stock_percentages) // stock_prices

        commission = sum(
            self.cost_model.stock_cost(p, q)
            for p, q in zip(stock_prices, qty)
        )
        self.current_cash -= np.sum(stock_prices * qty) + commission
        self._stocks_inventory = pd.DataFrame(
            {"symbol": stock_symbols, "price": stock_prices, "qty": qty}
        )

    def _update_balance(self, start_date, end_date):
        # Per-method Rust dispatch for balance computation (opt-in)
        if getattr(self, '_use_rust_loop', False) and use_rust():
            try:
                return self._update_balance_rust(start_date, end_date)
            except ImportError:
                pass  # polars not installed, fall through to Python

        stocks_date_col = self._stocks_schema["date"]
        sd = self._stocks_data._data
        stocks_data = sd[(sd[stocks_date_col] >= start_date) & (sd[stocks_date_col] < end_date)]

        options_date_col = self._options_schema["date"]
        od = self._options_data._data
        options_data = od[(od[options_date_col] >= start_date) & (od[options_date_col] < end_date)]

        calls_value = pd.Series(0.0, index=options_data[options_date_col].unique())
        puts_value = pd.Series(0.0, index=options_data[options_date_col].unique())

        options_contract_col = self._options_schema["contract"]
        for leg in self._options_strategy.legs:
            leg_inventory = self._options_inventory[leg.name]
            if leg_inventory.empty or leg_inventory["contract"].isna().all():
                continue
            cost_field = (~leg.direction).value

            inv_info = pd.DataFrame({
                "_contract": leg_inventory["contract"].values,
                "_qty": self._options_inventory["totals"]["qty"].values,
                "_type": leg_inventory["type"].values,
            })

            all_current = inv_info.merge(
                options_data, how="left",
                left_on="_contract", right_on=options_contract_col,
            )

            sign = -1 if cost_field == LegacyDirection.BUY.value else 1
            all_current["_value"] = (
                sign * all_current[cost_field]
                * all_current["_qty"] * self.shares_per_contract
            )

            calls_mask = all_current["_type"] == LegacyType.CALL.value
            if calls_mask.any():
                calls_data = all_current.loc[calls_mask].groupby(options_date_col)["_value"].sum()
                calls_value = calls_value.add(calls_data, fill_value=0)
            puts_mask = ~calls_mask
            if puts_mask.any():
                puts_data = all_current.loc[puts_mask].groupby(options_date_col)["_value"].sum()
                puts_value = puts_value.add(puts_data, fill_value=0)

        stocks_current = self._stocks_inventory[["symbol", "qty"]].merge(
            stocks_data[["date", "symbol", "adjClose"]], on="symbol",
        )
        stocks_current["cost"] = stocks_current["qty"] * stocks_current["adjClose"]

        add = stocks_current.pivot_table(
            index=stocks_date_col, columns="symbol", values="cost", aggfunc="sum",
        )
        add = add.reindex(columns=[stock.symbol for stock in self._stocks])

        add["cash"] = self.current_cash
        add["options qty"] = self._options_inventory["totals"]["qty"].sum()
        add["calls capital"] = calls_value
        add["puts capital"] = puts_value
        add["stocks qty"] = self._stocks_inventory["qty"].sum()

        for symbol, qty in zip(self._stocks_inventory["symbol"], self._stocks_inventory["qty"]):
            add[symbol + " qty"] = qty

        self._balance_parts.append(pd.DataFrame(add))

    def _update_balance_rust(self, start_date, end_date):
        """Rust-accelerated balance update using the _ob_rust.update_balance function."""
        import polars as pl

        stocks_date_col = self._stocks_schema["date"]
        sd = self._stocks_data._data
        stocks_data = sd[(sd[stocks_date_col] >= start_date) & (sd[stocks_date_col] < end_date)]

        options_date_col = self._options_schema["date"]
        od = self._options_data._data
        options_data = od[(od[options_date_col] >= start_date) & (od[options_date_col] < end_date)]

        if options_data.empty:
            return

        # Build leg inventory lists for Rust
        leg_contracts = []
        leg_qtys = []
        leg_types = []
        leg_directions = []
        for leg in self._options_strategy.legs:
            leg_inv = self._options_inventory[leg.name]
            if leg_inv.empty or leg_inv["contract"].isna().all():
                leg_contracts.append([])
                leg_qtys.append([])
                leg_types.append([])
            else:
                leg_contracts.append(leg_inv["contract"].astype(str).tolist())
                leg_qtys.append(self._options_inventory["totals"]["qty"].astype(float).tolist())
                leg_types.append(leg_inv["type"].astype(str).tolist())
            leg_directions.append(leg.direction.value)

        opts_pl = pl.from_pandas(options_data)
        stocks_pl = pl.from_pandas(stocks_data)

        result_pl = rust.update_balance(
            leg_contracts, leg_qtys, leg_types, leg_directions,
            self._stocks_inventory["symbol"].tolist(),
            self._stocks_inventory["qty"].astype(float).tolist(),
            opts_pl, stocks_pl,
            self._options_schema["contract"],
            options_date_col,
            stocks_date_col,
            self._stocks_schema["symbol"],
            self._stocks_schema["adjClose"],
            self.shares_per_contract,
            self.current_cash,
        )

        # Convert Polars result back to pandas and append to balance parts
        result_pd = result_pl.to_pandas()
        if not result_pd.empty:
            # Remap column names for backward compat
            renames = {
                "calls_capital": "calls capital",
                "puts_capital": "puts capital",
                "options_qty": "options qty",
                "stocks_qty": "stocks qty",
            }
            result_pd.rename(columns=renames, inplace=True)
            if options_date_col in result_pd.columns:
                result_pd.set_index(options_date_col, inplace=True)

            # Add stock qty columns
            for symbol, qty in zip(self._stocks_inventory["symbol"],
                                   self._stocks_inventory["qty"]):
                result_pd[symbol + " qty"] = qty

            self._balance_parts.append(result_pd)

    def _compute_portfolio_greeks(self, options) -> Greeks:
        """Compute aggregate Greeks for current portfolio positions."""
        total = Greeks()
        contract_col = self._options_schema["contract"]

        for leg in self._options_strategy.legs:
            leg_inventory = self._options_inventory[leg.name]
            if leg_inventory.empty or leg_inventory["contract"].isna().all():
                continue

            inv = pd.DataFrame({
                "_contract": leg_inventory["contract"].values,
                "_qty": self._options_inventory["totals"]["qty"].values,
            })

            merged = inv.merge(
                options, how="left",
                left_on="_contract", right_on=contract_col,
            )

            sign = 1 if leg.direction == LegacyDirection.BUY else -1

            for col in ("delta", "gamma", "theta", "vega"):
                if col not in merged.columns:
                    merged[col] = 0.0

            valid = merged.dropna(subset=["_qty"])
            if valid.empty:
                continue

            d = (valid["delta"].fillna(0) * valid["_qty"] * sign).sum()
            g = (valid["gamma"].fillna(0) * valid["_qty"] * sign).sum()
            t = (valid["theta"].fillna(0) * valid["_qty"] * sign).sum()
            v = (valid["vega"].fillna(0) * valid["_qty"] * sign).sum()

            total = total + Greeks(delta=d, gamma=g, theta=t, vega=v)

        return total

    def _compute_entry_greeks(self, entries, options) -> Greeks:
        """Compute Greeks for a proposed entry."""
        total = Greeks()
        contract_col = self._options_schema["contract"]
        qty = entries["totals"]["qty"]

        for leg in self._options_strategy.legs:
            contract_id = entries[leg.name]["contract"]
            match = options[options[contract_col] == contract_id]
            if match.empty:
                continue

            row = match.iloc[0]
            sign = 1 if leg.direction == LegacyDirection.BUY else -1

            d = float(row.get("delta", 0) or 0)
            g = float(row.get("gamma", 0) or 0)
            t = float(row.get("theta", 0) or 0)
            v = float(row.get("vega", 0) or 0)

            total = total + Greeks(
                delta=d * sign * qty, gamma=g * sign * qty,
                theta=t * sign * qty, vega=v * sign * qty,
            )

        return total

    def _execute_option_entries(self, date, options, options_allocation, stocks=None):
        self.current_cash += options_allocation

        inventory_contracts = pd.concat(
            [self._options_inventory[leg.name]["contract"] for leg in self._options_strategy.legs]
        )
        subset_options = options[
            ~options[self._options_schema["contract"]].isin(inventory_contracts)
        ]

        entry_signals: list[pd.DataFrame] = []
        for leg in self._options_strategy.legs:
            flt = leg.entry_filter
            cost_field = leg.direction.value

            leg_entries = subset_options[flt(subset_options)]
            if leg_entries.empty:
                return

            if leg.entry_sort:
                col, asc = leg.entry_sort
                leg_entries = leg_entries.sort_values(col, ascending=asc)

            fields = self._signal_fields(cost_field)
            leg_entries = leg_entries.reindex(columns=fields.keys())
            leg_entries.rename(columns=fields, inplace=True)

            order = legacy_get_order(leg.direction, LegacySignal.ENTRY)
            leg_entries["order"] = order

            if leg.direction == LegacyDirection.SELL:
                leg_entries["cost"] = -leg_entries["cost"]

            leg_entries["cost"] *= self.shares_per_contract
            leg_entries.columns = pd.MultiIndex.from_product(
                [[leg.name], leg_entries.columns]
            )
            entry_signals.append(leg_entries.reset_index(drop=True))

        total_costs = sum(
            [leg_entry.droplevel(0, axis=1)["cost"] for leg_entry in entry_signals]
        )
        qty = options_allocation // abs(total_costs)
        totals = pd.DataFrame.from_dict({"cost": total_costs, "qty": qty, "date": date})
        totals.columns = pd.MultiIndex.from_product([["totals"], totals.columns])
        entry_signals.append(totals)
        entry_signals_df = pd.concat(entry_signals, axis=1)

        entry_signals_df = entry_signals_df[entry_signals_df["totals"]["qty"] > 0]

        entries = self._pick_entry_signals(entry_signals_df, subset_options)

        # Apply per-leg fill model if available
        if not entries.empty:
            entries = self._apply_fill_models(entries, subset_options)

        # Risk check: reject entry if any constraint is violated
        if not entries.empty and self.risk_manager.constraints:
            portfolio_value = (
                self._total_capital_estimate(stocks, options)
                if stocks is not None else self.current_cash
            )
            current_greeks = self._compute_portfolio_greeks(options)
            proposed_greeks = self._compute_entry_greeks(entries, options)
            allowed, _reason = self.risk_manager.is_allowed(
                current_greeks, proposed_greeks, portfolio_value, self._peak_value,
            )
            if not allowed:
                return

        entries_df = pd.DataFrame(entries) if entries.empty else pd.DataFrame([entries])
        self._options_inventory = pd.concat(
            [self._options_inventory, entries_df], ignore_index=True
        )
        self._trade_log_parts.append(entries_df)

        # Dual-write: add to Portfolio dataclass
        if not entries.empty:
            self._add_position_to_portfolio(entries, date)

        # Apply commission from cost model
        qty_val = entries["totals"]["qty"]
        cost_val = entries["totals"]["cost"]
        commission = self.cost_model.option_cost(
            abs(cost_val), int(qty_val), self.shares_per_contract
        )
        self.current_cash -= np.sum(cost_val * qty_val) + commission

    def _execute_option_exits(self, date, options):
        strategy = self._options_strategy
        current_options_quotes = self._get_current_option_quotes(options)

        filter_masks: list[pd.Series] = []
        for i, leg in enumerate(strategy.legs):
            flt = leg.exit_filter
            missing_contracts_mask = current_options_quotes[i]["cost"].isna()
            filter_masks.append(flt(current_options_quotes[i]) | missing_contracts_mask)
            fields = self._signal_fields((~leg.direction).value)
            current_options_quotes[i] = current_options_quotes[i].reindex(columns=fields.values())
            current_options_quotes[i].rename(columns=fields, inplace=True)
            current_options_quotes[i].columns = pd.MultiIndex.from_product(
                [[leg.name], current_options_quotes[i].columns]
            )

        exit_candidates = pd.concat(current_options_quotes, axis=1)
        exit_candidates = self._impute_missing_option_values(exit_candidates)

        qtys = self._options_inventory["totals"]["qty"]
        total_costs = sum(
            [exit_candidates[leg.name]["cost"] for leg in self._options_strategy.legs]
        )
        totals = pd.DataFrame.from_dict({"cost": total_costs, "qty": qtys, "date": date})
        totals.columns = pd.MultiIndex.from_product([["totals"], totals.columns])
        exit_candidates = pd.concat([exit_candidates, totals], axis=1)

        threshold_exits = strategy.filter_thresholds(
            self._options_inventory["totals"]["cost"], total_costs
        )
        filter_mask = reduce(lambda x, y: x | y, filter_masks)
        exits_mask = threshold_exits | filter_mask

        exits = exit_candidates[exits_mask]
        total_costs = total_costs[exits_mask] * exits["totals"]["qty"]

        exited_indices = self._options_inventory[exits_mask].index.tolist()
        self._options_inventory.drop(exited_indices, inplace=True)

        # Dual-write: remove from Portfolio dataclass
        for idx in exited_indices:
            self._portfolio.remove_option_position(idx)

        if not exits.empty:
            self._trade_log_parts.append(pd.DataFrame(exits))

        # Apply commission on exit
        for _, row in exits.iterrows():
            commission = self.cost_model.option_cost(
                abs(row["totals"]["cost"]),
                int(abs(row["totals"]["qty"])),
                self.shares_per_contract,
            )
            self.current_cash -= commission

        self.current_cash -= sum(total_costs)

    def _pick_entry_signals(self, entry_signals, subset_options=None):
        if entry_signals.empty:
            return entry_signals

        # Build flat candidates from the first leg for the signal selector
        first_leg = self._options_strategy.legs[0]
        candidates = entry_signals[first_leg.name].copy()

        # Use per-leg signal selector if available, otherwise engine-level
        selector = self.signal_selector
        if hasattr(first_leg, 'signal_selector') and first_leg.signal_selector is not None:
            selector = first_leg.signal_selector

        # Enrich with extra columns the selector needs (e.g. delta, openinterest)
        extra_cols = selector.column_requirements
        if extra_cols and subset_options is not None:
            contract_col = self._options_schema["contract"]
            for col in extra_cols:
                if col in subset_options.columns:
                    lookup = (
                        subset_options
                        .drop_duplicates(contract_col)
                        .set_index(contract_col)[col]
                    )
                    candidates[col] = candidates["contract"].map(lookup)

        selected = selector.select(candidates)
        return entry_signals.loc[selected.name]

    def _apply_fill_models(self, entries, subset_options):
        """Re-price entry legs using per-leg fill models when they differ from MarketAtBidAsk."""
        contract_col = self._options_schema["contract"]
        total_cost = 0.0
        for leg in self._options_strategy.legs:
            fill = getattr(leg, 'fill_model', None)
            if fill is None:
                fill = self.fill_model
            if isinstance(fill, MarketAtBidAsk):
                total_cost += entries[leg.name]["cost"]
                continue
            contract_id = entries[leg.name]["contract"]
            match = subset_options[subset_options[contract_col] == contract_id]
            if match.empty:
                total_cost += entries[leg.name]["cost"]
                continue
            new_price = fill.get_fill_price(match.iloc[0], leg.direction)
            # Handle both legacy Direction (value="bid") and new Direction (value="sell")
            is_sell = (leg.direction == LegacyDirection.SELL
                       or getattr(leg.direction, 'value', None) == "sell")
            sign = -1 if is_sell else 1
            new_cost = sign * new_price * self.shares_per_contract
            entries[leg.name, "cost"] = new_cost
            total_cost += new_cost
        entries["totals", "cost"] = total_cost
        return entries

    def _add_position_to_portfolio(self, entries, date):
        """Dual-write: create OptionPosition from entry signals and add to portfolio."""
        pid = self._portfolio.next_position_id()
        pos = OptionPosition(
            position_id=pid,
            quantity=int(entries["totals"]["qty"]),
            entry_cost=float(entries["totals"]["cost"]),
            entry_date=date,
        )
        for leg in self._options_strategy.legs:
            leg_data = entries[leg.name]
            pos.add_leg(PositionLeg(
                name=leg.name,
                contract_id=leg_data["contract"],
                underlying=leg_data["underlying"],
                expiration=leg_data["expiration"],
                option_type=OptionType.CALL if leg_data["type"] == LegacyType.CALL.value else OptionType.PUT,
                strike=float(leg_data["strike"]),
                entry_price=float(leg_data["cost"]),
                direction=Direction.BUY if leg.direction == LegacyDirection.BUY else Direction.SELL,
                order=leg_data["order"],
            ))
        # Use the DataFrame index as the position ID for exit cross-reference
        inv_idx = self._options_inventory.index[-1]
        self._portfolio.option_positions[inv_idx] = pos

    def _signal_fields(self, cost_field):
        return {
            self._options_schema["contract"]: "contract",
            self._options_schema["underlying"]: "underlying",
            self._options_schema["expiration"]: "expiration",
            self._options_schema["type"]: "type",
            self._options_schema["strike"]: "strike",
            self._options_schema[cost_field]: "cost",
            "order": "order",
        }

    def _get_current_option_quotes(self, options):
        current_options_quotes: list[pd.DataFrame] = []
        for leg in self._options_strategy.legs:
            inventory_leg = self._options_inventory[leg.name]
            leg_options = inventory_leg[["contract"]].merge(
                options, how="left",
                left_on="contract", right_on=leg.schema["contract"],
            )
            leg_options.index = self._options_inventory.index
            leg_options["order"] = legacy_get_order(leg.direction, LegacySignal.EXIT)
            leg_options["cost"] = leg_options[self._options_schema[(~leg.direction).value]]

            if ~leg.direction == LegacyDirection.SELL:
                leg_options["cost"] = -leg_options["cost"]
            leg_options["cost"] *= self.shares_per_contract
            current_options_quotes.append(leg_options)
        return current_options_quotes

    def _impute_missing_option_values(self, exit_candidates):
        df = self._options_inventory.copy()
        if not df.empty:
            for leg in self._options_strategy.legs:
                df.loc[(leg.name, "cost")] = 0
        return exit_candidates.fillna(df)

    def __repr__(self) -> str:
        return (
            f"BacktestEngine(capital={self.initial_capital}, "
            f"allocation={self.allocation}, "
            f"cost_model={self.cost_model.__class__.__name__})"
        )
