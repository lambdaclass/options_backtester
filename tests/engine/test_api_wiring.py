"""Public options that used to be accepted and silently ignored.

``sizer=`` never reached the Rust engine (FixedQuantity(1) traded dozens of
contracts); a Strategy's own cost model / sizer / selector were ignored;
``schema=`` on a data provider raised RecursionError; ``current_cash`` was
``allocation["cash"] * final capital`` rather than the cash actually held.
"""

import math

import pandas as pd
import pytest

from options_portfolio_backtester import (
    BacktestEngine,
    FixedDollar,
    FixedQuantity,
    HistoricalOptionsData,
    PercentOfPortfolio,
    TiingoData,
)
from options_portfolio_backtester.core.types import Direction, OptionType, Stock
from options_portfolio_backtester.execution.cost_model import NoCosts, PerContractCommission
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from tests.oracles.test_golden_scenario import DATES, PUT_QUOTES, STOCK_PRICES, _option_row, _write_stocks_csv


def _files(tmp_path):
    stocks_csv, options_csv = tmp_path / "s.csv", tmp_path / "o.csv"
    _write_stocks_csv(stocks_csv)
    rows = [_option_row(d, "XYZ200717P00080000", "2020-07-17", 80.0, bid, ask, px)
            for d, px, (bid, ask) in zip(DATES, STOCK_PRICES, PUT_QUOTES)]
    pd.DataFrame(rows).to_csv(options_csv, index=False)
    return stocks_csv, options_csv


def _run(tmp_path, engine_kwargs=None, strategy_kwargs=None):
    stocks_csv, options_csv = _files(tmp_path)
    options_data = HistoricalOptionsData(str(options_csv))
    sch = options_data.schema
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (sch.underlying == "XYZ") & (sch.dte >= 40)
    leg.exit_filter = sch.dte <= 28
    strat = Strategy(sch, **(strategy_kwargs or {}))
    strat.add_legs([leg])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    eng = BacktestEngine({"stocks": 0.9, "options": 0.0, "cash": 0.1}, initial_capital=1_000_030.00,
                         **(engine_kwargs or {}))
    eng.stocks = [Stock("XYZ", 1.0)]
    eng.stocks_data = TiingoData(str(stocks_csv))
    eng.options_data = options_data
    eng.options_strategy = strat
    eng.options_budget_pct = 0.05
    eng.check_exits_daily = True
    eng.run(rebalance_freq=1, rebalance_unit="BMS")
    return eng


def _entry_qty(eng):
    return int(eng.trade_log["totals"]["qty"].iloc[0])


@pytest.mark.parametrize("sizer, qty", [
    (None, 500),                       # budget 50,001.50 / ask 1.00 * 100
    (FixedQuantity(1), 1),
    (FixedDollar(2_500.0), 25),
    (PercentOfPortfolio(0.001), 10),   # 1,000.03 / 100
], ids=["capital-based", "fixed-qty", "fixed-dollar", "pct-portfolio"])
def test_engine_sizer_is_applied(tmp_path, sizer, qty):
    eng = _run(tmp_path, engine_kwargs={"sizer": sizer} if sizer else None)
    assert _entry_qty(eng) == qty


def test_strategy_level_models_override_the_engine(tmp_path):
    eng = _run(tmp_path, engine_kwargs={"sizer": FixedQuantity(7), "cost_model": NoCosts()},
               strategy_kwargs={"sizer": FixedQuantity(3),
                                "cost_model": PerContractCommission(rate=1.0, stock_rate=0.0)})
    assert _entry_qty(eng) == 3
    # 3 contracts at 100.00 + $1/contract commission: cash drops by 303 at entry
    # on top of the stock purchase, so total capital on day 1 is below the
    # commission-free run's.
    free = _run(tmp_path, engine_kwargs={"sizer": FixedQuantity(3)})
    day1 = pd.Timestamp("2020-06-01")
    assert free.balance.loc[day1, "total capital"] - eng.balance.loc[day1, "total capital"] \
        == pytest.approx(3.0, abs=0.01)


def test_current_cash_is_the_final_cash(tmp_path):
    eng = _run(tmp_path)
    assert eng.current_cash == pytest.approx(float(eng.balance["cash"].iloc[-1]))


def test_provider_schema_argument(tmp_path):
    stocks_csv, options_csv = _files(tmp_path)
    schema = HistoricalOptionsData.default_schema()
    assert HistoricalOptionsData(str(options_csv), schema=schema).schema is schema
    assert TiingoData(str(stocks_csv), schema=TiingoData.default_schema()).start_date == pd.Timestamp(DATES[0])


def test_provider_names_missing_columns(tmp_path):
    stocks_csv, _ = _files(tmp_path)
    pd.read_csv(stocks_csv).drop(columns=["adjClose"]).to_csv(stocks_csv, index=False)
    with pytest.raises(ValueError, match="adjClose"):
        TiingoData(str(stocks_csv))


@pytest.mark.parametrize("allocation", [{"stock": 1.0}, {"stocks": 0.0}])
def test_bad_allocation_raises(allocation):
    with pytest.raises(ValueError):
        BacktestEngine(allocation)
