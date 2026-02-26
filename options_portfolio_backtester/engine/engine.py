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

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import pyprind

from options_portfolio_backtester.core.types import (
    Direction, OptionType, Order, Signal, Greeks, Stock, StockAllocation,
    get_order,
)
from options_portfolio_backtester.execution.cost_model import TransactionCostModel, NoCosts
from options_portfolio_backtester.execution.fill_model import FillModel, MarketAtBidAsk
from options_portfolio_backtester.execution.sizer import PositionSizer, CapitalBased
from options_portfolio_backtester.execution.signal_selector import SignalSelector, FirstMatch
from options_portfolio_backtester.portfolio.risk import RiskManager
from options_portfolio_backtester.portfolio.portfolio import Portfolio, StockHolding
from options_portfolio_backtester.portfolio.position import OptionPosition, PositionLeg
from options_portfolio_backtester.engine._dispatch import use_rust, rust
from options_portfolio_backtester.engine.algo_adapters import (
    EngineAlgo,
    EnginePipelineContext,
)

from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.data.schema import Schema
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg


class BacktestEngine:
    """Orchestrates backtest with pluggable execution components.

    Composes data providers, strategy legs, cost/fill/sizer/selector models,
    and risk constraints into a single backtest loop.  Dispatches to Rust
    when available, falls back to Python transparently.
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
        algos: list[EngineAlgo] | None = None,
        stop_if_broke: bool = False,
        max_notional_pct: float | None = None,
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
        self.algos = list(algos or [])
        self.stop_if_broke = stop_if_broke
        self.max_notional_pct = max_notional_pct

        self.options_budget: Union[Callable[[pd.Timestamp, float], float], float, None] = None
        self._stocks: list[Stock] = []
        self._options_strategy: Strategy | None = None
        self._stocks_data: TiingoData | None = None
        self._options_data: HistoricalOptionsData | None = None
        self.run_metadata: dict[str, Any] = {}
        self._event_log_rows: list[dict[str, Any]] = []

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
            sma_days: int | None = None,
            rebalance_unit: str = 'BMS') -> pd.DataFrame:
        """Run the backtest. Returns the trade log DataFrame."""
        self._event_log_rows = []
        for algo in self.algos:
            if hasattr(algo, "reset"):
                algo.reset()
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

        # Dispatch to Rust full-loop when available.
        # Requires: no algos, no custom options_budget,
        # and all execution models (engine-level AND per-leg) must have to_rust_config().
        # monthly is OK — rebalance dates are pre-computed in Python either way.
        def _has_rust_config(obj):
            return obj is None or hasattr(obj, 'to_rust_config')

        _rust_compatible = (
            use_rust()
            and not self.algos
            and self.max_notional_pct is None
            and (self.options_budget is None or isinstance(self.options_budget, (int, float)))
            and hasattr(self.cost_model, 'to_rust_config')
            and hasattr(self.fill_model, 'to_rust_config')
            and hasattr(self.signal_selector, 'to_rust_config')
            and all(hasattr(c, 'to_rust_config') for c in self.risk_manager.constraints)
            and all(
                _has_rust_config(getattr(leg, 'signal_selector', None))
                and _has_rust_config(getattr(leg, 'fill_model', None))
                for leg in self._options_strategy.legs
            )
        )
        if _rust_compatible:
            try:
                return self._run_rust(
                    rebalance_freq,
                    monthly=monthly,
                    sma_days=sma_days,
                    rebalance_unit=rebalance_unit,
                )
            except Exception:
                pass

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
                dates.groupby(pd.Grouper(freq=f"{rebalance_freq}{rebalance_unit}"))
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
                self._log_event(date, "rebalance_start", "ok", {
                    "cash": float(self.current_cash),
                })

                stock_capital = self._current_stock_capital(stocks)
                options_capital = self._current_options_capital(options)
                total_capital = self.current_cash + stock_capital + options_capital
                if self.options_budget is not None and not callable(self.options_budget):
                    options_allocation = float(self.options_budget)
                elif self.options_budget is not None and callable(self.options_budget):
                    options_allocation = self.options_budget(date, total_capital)
                else:
                    options_allocation = self.allocation["options"] * total_capital
                current_greeks = self._compute_portfolio_greeks(options)
                ctx = EnginePipelineContext(
                    date=pd.Timestamp(date),
                    stocks=stocks,
                    options=options,
                    total_capital=float(total_capital),
                    current_cash=float(self.current_cash),
                    current_greeks=current_greeks,
                    options_allocation=float(options_allocation),
                )
                should_stop, skip_day = self._apply_algos(ctx)
                if should_stop:
                    self._log_event(date, "algo_stop", "stop", {})
                    break
                if skip_day:
                    self._log_event(date, "algo_skip_day", "skip_day", {})
                    continue

                # Only use override if an algo actually modified options_allocation;
                # otherwise let _rebalance_portfolio compute it from post-exit
                # total_capital, matching the Rust path.
                algo_changed_alloc = (
                    ctx.options_allocation != float(options_allocation)
                )
                self._rebalance_portfolio(
                    date,
                    stocks,
                    ctx.options,
                    sma_days,
                    options_allocation_override=(
                        ctx.options_allocation if algo_changed_alloc else None
                    ),
                    entry_filters=ctx.entry_filters,
                    exit_threshold_override=ctx.exit_threshold_override,
                )

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
        self._attach_run_metadata(
            rebalance_freq=rebalance_freq,
            monthly=monthly,
            sma_days=sma_days,
            dispatch_mode="python",
        )

        return self.trade_log

    def events_dataframe(self) -> pd.DataFrame:
        """Structured execution event log for debugging and audit.

        The ``data`` dict from each event is flattened into top-level columns
        so that the result can be filtered directly (e.g.
        ``df[df["cash"] > 0]``).
        """
        if not self._event_log_rows:
            return pd.DataFrame(columns=["date", "event", "status"])
        flat = []
        for row in self._event_log_rows:
            entry = {"date": row["date"], "event": row["event"], "status": row["status"]}
            entry.update(row.get("data", {}))
            flat.append(entry)
        return pd.DataFrame(flat)

    def _run_rust(
        self,
        rebalance_freq: int,
        monthly: bool,
        sma_days: int | None,
        rebalance_unit: str = 'BMS',
    ) -> pd.DataFrame:
        """Run the backtest using the Rust full-loop implementation."""
        import math
        import pyarrow as pa
        import polars as pl

        strategy = self._options_strategy

        # Compute rebalance dates in Python (same logic as the Python loop)
        # to guarantee date parity with the Python path.
        dates_df = (
            pd.DataFrame(self.options_data._data[["quotedate", "volume"]])
            .drop_duplicates("quotedate")
            .set_index("quotedate")
        )
        if rebalance_freq:
            rebalancing_days = pd.to_datetime(
                dates_df.groupby(pd.Grouper(freq=f"{rebalance_freq}{rebalance_unit}"))
                .apply(lambda x: x.index.min())
                .values
            )
            # Pass rebalance dates as i64 nanoseconds (matching Polars Datetime(ns))
            rb_date_ns = [int(d.value) for d in rebalancing_days if not pd.isna(d)]
        else:
            rb_date_ns = []

        opts_date_col = self._options_schema["date"]
        stocks_date_col = self._stocks_schema["date"]
        exp_col = self._options_schema["expiration"]

        # Drop columns Rust never accesses to reduce Arrow conversion cost.
        _drop_cols = {"underlying_last", "last", "optionalias", "impliedvol"}
        # Also drop openinterest unless MaxOpenInterest selector is in use
        if not (hasattr(self.signal_selector, '__class__')
                and self.signal_selector.__class__.__name__ == 'MaxOpenInterest'):
            _drop_cols.add("openinterest")
        opts_df = self._options_data._data
        to_drop = [c for c in _drop_cols if c in opts_df.columns]
        opts_src = opts_df.drop(columns=to_drop) if to_drop else opts_df

        # Convert pandas → PyArrow → Polars (avoids intermediate copies).
        opts_pl = pl.from_arrow(pa.Table.from_pandas(opts_src, preserve_index=False))
        stocks_pl = pl.from_arrow(
            pa.Table.from_pandas(self._stocks_data._data, preserve_index=False)
        )

        leg_configs = []
        for leg in strategy.legs:
            lc = {
                "name": leg.name,
                "entry_filter": leg.entry_filter.query,
                "exit_filter": leg.exit_filter.query,
                "direction": leg.direction.price_column,
                "type": leg.type.value,
                "entry_sort_col": leg.entry_sort[0] if leg.entry_sort else None,
                "entry_sort_asc": leg.entry_sort[1] if leg.entry_sort else True,
            }
            # Per-leg overrides
            leg_sel = getattr(leg, 'signal_selector', None)
            if leg_sel is not None and hasattr(leg_sel, 'to_rust_config'):
                lc["signal_selector"] = leg_sel.to_rust_config()
            leg_fill = getattr(leg, 'fill_model', None)
            if leg_fill is not None and hasattr(leg_fill, 'to_rust_config'):
                lc["fill_model"] = leg_fill.to_rust_config()
            leg_configs.append(lc)

        config = {
            "allocation": self.allocation,
            "initial_capital": float(self.initial_capital),
            "shares_per_contract": self.shares_per_contract,
            "rebalance_dates": rb_date_ns,
            "legs": leg_configs,
            "profit_pct": (
                strategy.exit_thresholds[0]
                if strategy.exit_thresholds[0] != math.inf else None
            ),
            "loss_pct": (
                strategy.exit_thresholds[1]
                if strategy.exit_thresholds[1] != math.inf else None
            ),
            "stocks": [(s.symbol, s.percentage) for s in self._stocks],
            "cost_model": self.cost_model.to_rust_config(),
            "fill_model": self.fill_model.to_rust_config(),
            "signal_selector": self.signal_selector.to_rust_config(),
            "risk_constraints": [c.to_rust_config() for c in self.risk_manager.constraints],
            "sma_days": sma_days,
            "options_budget_fixed": (
                float(self.options_budget)
                if isinstance(self.options_budget, (int, float))
                else None
            ),
            "stop_if_broke": self.stop_if_broke,
        }

        schema_mapping = {
            "contract": self._options_schema["contract"],
            "date": opts_date_col,
            "stocks_date": stocks_date_col,
            "stocks_symbol": self._stocks_schema["symbol"],
            "stocks_price": self._stocks_schema["adjClose"],
            "underlying": self._options_schema["underlying"],
            "expiration": self._options_schema["expiration"],
            "type": self._options_schema["type"],
            "strike": self._options_schema["strike"],
        }

        balance_pl, trade_log_pl, stats = rust.run_backtest_py(
            opts_pl, stocks_pl, config, schema_mapping,
        )

        # Convert trade log from flat columns to MultiIndex
        trade_log_pd = trade_log_pl.to_pandas()
        self.trade_log = self._flat_trade_log_to_multiindex(trade_log_pd)

        # Convert balance
        self.balance = balance_pl.to_pandas()
        if "date" in self.balance.columns:
            self.balance["date"] = pd.to_datetime(self.balance["date"])
            self.balance.set_index("date", inplace=True)

        # Add initial balance row (day before first rebalance) — matches Python
        initial_date = self.stocks_data.start_date - pd.Timedelta(1, unit="day")
        initial_row = pd.DataFrame(
            {"total capital": self.initial_capital, "cash": float(self.initial_capital)},
            index=[initial_date],
        )
        self.balance = pd.concat([initial_row, self.balance], sort=False)
        for col_name in self.balance.columns:
            self.balance[col_name] = pd.to_numeric(self.balance[col_name], errors="coerce")

        # Ensure per-stock columns exist (match Python's balance format)
        for stock in self._stocks:
            sym = stock.symbol
            if sym not in self.balance.columns:
                self.balance[sym] = 0.0
            if f"{sym} qty" not in self.balance.columns:
                self.balance[f"{sym} qty"] = 0.0
        for col_name in ["options qty", "stocks qty", "calls capital", "puts capital"]:
            if col_name not in self.balance.columns:
                self.balance[col_name] = 0.0

        # Add derived columns matching Python output
        self.balance["options capital"] = (
            self.balance["calls capital"] + self.balance["puts capital"]
        ).fillna(0)
        stock_cols = [s.symbol for s in self._stocks]
        self.balance["stocks capital"] = sum(
            self.balance.get(c, 0) for c in stock_cols
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
        self.balance["accumulated return"] = (1.0 + self.balance["% change"]).cumprod()

        # Set current_cash to match Python loop's final state after rebalancing
        # (after the loop, all capital is allocated to stocks/options/cash per allocation)
        final_total = self.balance["total capital"].iloc[-1]
        self.current_cash = self.allocation["cash"] * final_total
        self._initialize_inventories()
        self._portfolio = Portfolio(initial_cash=self.current_cash)
        self._attach_run_metadata(
            rebalance_freq=rebalance_freq,
            monthly=monthly,
            sma_days=sma_days,
            dispatch_mode="rust-full",
        )

        return self.trade_log

    def _attach_run_metadata(
        self,
        rebalance_freq: int,
        monthly: bool,
        sma_days: int | None,
        dispatch_mode: str,
    ) -> None:
        metadata = self._build_run_metadata(
            rebalance_freq=rebalance_freq,
            monthly=monthly,
            sma_days=sma_days,
            dispatch_mode=dispatch_mode,
        )
        self.run_metadata = metadata
        self.balance.attrs["run_metadata"] = metadata
        self.trade_log.attrs["run_metadata"] = metadata

    def _build_run_metadata(
        self,
        rebalance_freq: int,
        monthly: bool,
        sma_days: int | None,
        dispatch_mode: str,
    ) -> dict[str, Any]:
        stocks = [
            {"symbol": stock.symbol, "percentage": float(stock.percentage)}
            for stock in self._stocks
        ]
        run_config = {
            "allocation": {k: float(v) for k, v in self.allocation.items()},
            "initial_capital": float(self.initial_capital),
            "shares_per_contract": int(self.shares_per_contract),
            "rebalance_freq": int(rebalance_freq),
            "monthly": bool(monthly),
            "sma_days": int(sma_days) if sma_days is not None else None,
            "stocks": stocks,
        }
        data_snapshot = self._data_snapshot()
        return {
            "framework": "options_portfolio_backtester.engine.BacktestEngine",
            "dispatch_mode": dispatch_mode,
            "rust_available": bool(use_rust()),
            "git_sha": self._git_sha(),
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": self._sha256_json(run_config),
            "data_snapshot_hash": self._sha256_json(data_snapshot),
            "data_snapshot": data_snapshot,
        }

    def _data_snapshot(self) -> dict[str, Any]:
        options_dates = self._options_data["date"]
        stocks_dates = self._stocks_data["date"]
        return {
            "options_rows": int(len(self._options_data._data)),
            "stocks_rows": int(len(self._stocks_data._data)),
            "options_date_start": pd.Timestamp(options_dates.min()).isoformat(),
            "options_date_end": pd.Timestamp(options_dates.max()).isoformat(),
            "stocks_date_start": pd.Timestamp(stocks_dates.min()).isoformat(),
            "stocks_date_end": pd.Timestamp(stocks_dates.max()).isoformat(),
            "options_columns": list(self._options_data._data.columns),
            "stocks_columns": list(self._stocks_data._data.columns),
        }

    @staticmethod
    def _sha256_json(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _git_sha() -> str:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip()
        except Exception:
            return "unknown"

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

    def _rebalance_portfolio(
        self,
        date,
        stocks,
        options,
        sma_days,
        options_allocation_override: float | None = None,
        entry_filters: list | None = None,
        exit_threshold_override: tuple[float, float] | None = None,
    ):
        self._execute_option_exits(date, options, exit_threshold_override=exit_threshold_override)

        stock_capital = self._current_stock_capital(stocks)
        options_capital = self._current_options_capital(options)
        total_capital = self.current_cash + stock_capital + options_capital

        stocks_allocation = self.allocation["stocks"] * total_capital
        self._stocks_inventory = pd.DataFrame(columns=["symbol", "price", "qty"])
        self.current_cash = stocks_allocation + total_capital * self.allocation["cash"]
        self._buy_stocks(stocks, stocks_allocation, sma_days)

        if options_allocation_override is not None:
            options_allocation = float(options_allocation_override)
        elif self.options_budget is not None:
            budget = self.options_budget
            options_allocation: float = budget(date, total_capital) if callable(budget) else budget
        else:
            options_allocation = self.allocation["options"] * total_capital
        self._log_event(date, "target_allocation", "ok", {
            "total_capital": float(total_capital),
            "options_allocation": float(options_allocation),
            "options_capital": float(options_capital),
        })
        if options_allocation >= options_capital:
            self._execute_option_entries(
                date,
                options,
                options_allocation - options_capital,
                stocks,
                entry_filters=entry_filters,
                total_capital=total_capital,
            )
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

        # Remove fully-sold positions (qty == 0) from inventory.
        # Without this, zero-qty ghost positions block contract re-entry.
        zero_qty = self._options_inventory[
            self._options_inventory["totals"]["qty"] == 0
        ].index
        if len(zero_qty) > 0:
            self._options_inventory.drop(zero_qty, inplace=True)
            for idx in zero_qty:
                self._portfolio.remove_option_position(idx)

        if trade_rows:
            self._trade_log_parts.append(pd.DataFrame(trade_rows))
        self._log_event(date, "partial_option_delever", "ok", {
            "target_to_sell": float(to_sell),
            "sold": float(sold),
            "trade_rows": int(len(trade_rows)),
        })
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
                # fillna(0): contracts missing from today's data are unpriced
                values_by_row += options_value[i]["cost"].fillna(0.0).values
            total: float = -sum(values_by_row * self._options_inventory["totals"]["qty"].values)
        else:
            total = 0
        return total

    def _buy_stocks(self, stocks, allocation, sma_days):
        stock_symbols = [stock.symbol for stock in self.stocks]
        sym_col = self._stocks_schema["symbol"]
        inventory_stocks = stocks[stocks[sym_col].isin(stock_symbols)]
        # Sort to match user-specified stock order so prices align with percentages
        symbol_order = {sym: i for i, sym in enumerate(stock_symbols)}
        inventory_stocks = inventory_stocks.sort_values(
            sym_col, key=lambda s: s.map(symbol_order)
        )
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
        # Per-method Rust dispatch for balance computation
        if use_rust():
            try:
                return self._update_balance_rust(start_date, end_date)
            except Exception:
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
            cost_field = (~leg.direction).price_column

            inv_info = pd.DataFrame({
                "_contract": leg_inventory["contract"].values,
                "_qty": self._options_inventory["totals"]["qty"].values,
                "_type": leg_inventory["type"].values,
            })

            all_current = inv_info.merge(
                options_data, how="left",
                left_on="_contract", right_on=options_contract_col,
            )

            sign = -1 if cost_field == Direction.BUY.price_column else 1
            all_current["_value"] = (
                sign * all_current[cost_field]
                * all_current["_qty"] * self.shares_per_contract
            )

            calls_mask = all_current["_type"] == OptionType.CALL.value
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
            leg_directions.append(leg.direction.price_column)

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

            sign = 1 if leg.direction == Direction.BUY else -1

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
            sign = 1 if leg.direction == Direction.BUY else -1

            d = float(row.get("delta", 0) or 0)
            g = float(row.get("gamma", 0) or 0)
            t = float(row.get("theta", 0) or 0)
            v = float(row.get("vega", 0) or 0)

            total = total + Greeks(
                delta=d * sign * qty, gamma=g * sign * qty,
                theta=t * sign * qty, vega=v * sign * qty,
            )

        return total

    def _compute_short_notional(self) -> float:
        """Sum of strike * qty * shares_per_contract for all short legs in inventory."""
        inv = self._options_inventory
        if inv.empty:
            return 0.0
        sell_legs = [leg for leg in self._options_strategy.legs
                     if leg.direction == Direction.SELL]
        if not sell_legs:
            return 0.0
        qty = inv["totals"]["qty"]
        return sum(
            (inv[leg.name]["strike"] * qty * self.shares_per_contract).sum()
            for leg in sell_legs
        )

    def _execute_option_entries(self, date, options, options_allocation, stocks=None, entry_filters=None, total_capital=None):
        self.current_cash += options_allocation

        inventory_contracts = pd.concat(
            [self._options_inventory[leg.name]["contract"] for leg in self._options_strategy.legs]
        )
        subset_options = options[
            ~options[self._options_schema["contract"]].isin(inventory_contracts)
        ]
        if entry_filters:
            for flt in entry_filters:
                subset_options = subset_options[flt(subset_options)]
            if subset_options.empty:
                self._log_event(date, "option_entry_filtered", "skip_day", {
                    "options_allocation": float(options_allocation),
                })
                self.current_cash -= options_allocation
                return

        entry_signals: list[pd.DataFrame] = []
        for leg in self._options_strategy.legs:
            flt = leg.entry_filter
            cost_field = leg.direction.price_column

            leg_entries = subset_options[flt(subset_options)]
            if leg_entries.empty:
                self._log_event(date, "option_entry_no_candidates", "skip_day", {
                    "leg": leg.name,
                })
                self.current_cash -= options_allocation
                return

            if leg.entry_sort:
                col, asc = leg.entry_sort
                leg_entries = leg_entries.sort_values(col, ascending=asc)

            fields = self._signal_fields(cost_field)
            leg_entries = leg_entries.reindex(columns=fields.keys())
            leg_entries.rename(columns=fields, inplace=True)

            order = get_order(leg.direction, Signal.ENTRY)
            leg_entries["order"] = order

            if leg.direction == Direction.SELL:
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

        if self.max_notional_pct is not None and total_capital is not None:
            sell_pairs = [
                (leg_entry, leg)
                for leg_entry, leg in zip(entry_signals, self._options_strategy.legs)
                if leg.direction == Direction.SELL
            ]
            if sell_pairs:
                existing_short_notional = self._compute_short_notional()
                max_notional = self.max_notional_pct * total_capital
                available = max(0.0, max_notional - existing_short_notional)
                short_notional_per_contract = sum(
                    leg_entry.droplevel(0, axis=1)["strike"] * self.shares_per_contract
                    for leg_entry, _ in sell_pairs
                )
                mask = short_notional_per_contract > 0
                if mask.any():
                    max_qty = available // short_notional_per_contract
                    qty = qty.clip(upper=max_qty)

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
                self._log_event(date, "risk_block_entry", "skip_day", {
                    "portfolio_value": float(portfolio_value),
                })
                self.current_cash -= options_allocation
                return

        if entries.empty:
            self._log_event(date, "option_entry_none_selected", "skip_day", {})
            self.current_cash -= options_allocation
            return

        entries_df = pd.DataFrame([entries])
        self._options_inventory = pd.concat(
            [self._options_inventory, entries_df], ignore_index=True
        )
        self._trade_log_parts.append(entries_df)
        self._log_event(date, "option_entry", "ok", {
            "qty": int(entries["totals"]["qty"]),
            "cost": float(entries["totals"]["cost"]),
        })

        # Dual-write: add to Portfolio dataclass
        self._add_position_to_portfolio(entries, date)

        # Apply commission from cost model
        qty_val = entries["totals"]["qty"]
        cost_val = entries["totals"]["cost"]
        commission = self.cost_model.option_cost(
            abs(cost_val), int(qty_val), self.shares_per_contract
        )
        self.current_cash -= np.sum(cost_val * qty_val) + commission

    def _execute_option_exits(self, date, options, exit_threshold_override: tuple[float, float] | None = None):
        strategy = self._options_strategy
        current_options_quotes = self._get_current_option_quotes(options)

        filter_masks: list[pd.Series] = []
        for i, leg in enumerate(strategy.legs):
            flt = leg.exit_filter
            missing_contracts_mask = current_options_quotes[i]["cost"].isna()
            filter_masks.append(flt(current_options_quotes[i]) | missing_contracts_mask)
            fields = self._signal_fields((~leg.direction).price_column)
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

        if exit_threshold_override is None:
            threshold_exits = strategy.filter_thresholds(
                self._options_inventory["totals"]["cost"], total_costs
            )
        else:
            entry_cost = self._options_inventory["totals"]["cost"]
            profit_pct, loss_pct = exit_threshold_override
            excess_return = (total_costs / entry_cost + 1) * -np.sign(entry_cost)
            threshold_exits = (excess_return >= profit_pct) | (excess_return <= -loss_pct)
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
        self._log_event(date, "option_exit", "ok", {
            "rows": int(len(exits)),
        })

        # Apply commission on exit
        for _, row in exits.iterrows():
            commission = self.cost_model.option_cost(
                abs(row["totals"]["cost"]),
                int(abs(row["totals"]["qty"])),
                self.shares_per_contract,
            )
            self.current_cash -= commission

        self.current_cash -= sum(total_costs)

    def _apply_algos(self, ctx: EnginePipelineContext) -> tuple[bool, bool]:
        should_stop = False
        skip_day = False
        for algo in self.algos:
            decision = algo(ctx)
            self._log_event(ctx.date, "algo_step", decision.status, {
                "step": algo.__class__.__name__,
                "message": decision.message,
            })
            if decision.status == "skip_day":
                skip_day = True
                break
            if decision.status == "stop":
                should_stop = True
                break
        return should_stop, skip_day

    def _log_event(self, date: pd.Timestamp, event: str, status: str, data: dict[str, Any]) -> None:
        self._event_log_rows.append({
            "date": pd.Timestamp(date),
            "event": str(event),
            "status": str(status),
            "data": data,
        })

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
            is_sell = leg.direction == Direction.SELL
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
                option_type=OptionType.CALL if leg_data["type"] == OptionType.CALL.value else OptionType.PUT,
                strike=float(leg_data["strike"]),
                entry_price=float(leg_data["cost"]),
                direction=leg.direction,
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
            leg_options["order"] = get_order(leg.direction, Signal.EXIT)
            leg_options["cost"] = leg_options[self._options_schema[(~leg.direction).price_column]]

            if ~leg.direction == Direction.SELL:
                leg_options["cost"] = -leg_options["cost"]
            leg_options["cost"] *= self.shares_per_contract
            current_options_quotes.append(leg_options)
        return current_options_quotes

    def _impute_missing_option_values(self, exit_candidates):
        df = self._options_inventory.copy()
        if not df.empty:
            for leg in self._options_strategy.legs:
                df[(leg.name, "cost")] = 0
        return exit_candidates.fillna(df)

    def __repr__(self) -> str:
        return (
            f"BacktestEngine(capital={self.initial_capital}, "
            f"allocation={self.allocation}, "
            f"cost_model={self.cost_model.__class__.__name__})"
        )
