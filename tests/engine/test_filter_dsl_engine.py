"""Filters reach the Rust engine intact, or the run fails loudly.

The Rust filter compiler used to drop any query it could not parse and run
as if the leg had no filter: an entry filter then matched every contract and
an exit filter never fired. Python's filter DSL also emitted queries the
parser rejected (a literal on the left, chained arithmetic, unquoted strings
in ``!=``).
"""

import math

import pandas as pd
import pytest

from options_portfolio_backtester import BacktestEngine
from options_portfolio_backtester.core.types import Direction, OptionType, Stock
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.data.schema import Filter
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from tests.oracles.test_golden_scenario import DATES, PUT_QUOTES, STOCK_PRICES, _option_row, _write_stocks_csv

# Two puts on XYZ (spot 100 on 06-01): strike 80 and strike 95.
STRIKES = {"XYZ200717P00080000": 80.0, "XYZ200717P00095000": 95.0}


def _run(tmp_path, entry):
    stocks_csv, options_csv = tmp_path / "s.csv", tmp_path / "o.csv"
    _write_stocks_csv(stocks_csv)
    rows = [_option_row(d, root, "2020-07-17", k, bid, ask, px)
            for root, k in STRIKES.items()
            for d, px, (bid, ask) in zip(DATES, STOCK_PRICES, PUT_QUOTES)]
    pd.DataFrame(rows).sort_values("quotedate", kind="stable").to_csv(options_csv, index=False)
    options_data = HistoricalOptionsData(str(options_csv))
    sch = options_data.schema

    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = entry(sch)
    leg.exit_filter = sch.dte <= 28
    strat = Strategy(sch)
    strat.add_legs([leg])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    eng = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_000,
                         cost_model=NoCosts(), fill_model=MarketAtBidAsk())
    eng.stocks = [Stock("XYZ", 1.0)]
    eng.stocks_data = TiingoData(str(stocks_csv))
    eng.options_data = options_data
    eng.options_strategy = strat
    eng.options_budget_pct = 0.01
    eng.check_exits_daily = True
    eng.run(rebalance_freq=1, rebalance_unit="BMS")
    return eng


def _bought(eng):
    tl = eng.trade_log
    buys = tl[tl["leg_1"]["order"].astype(str).str.contains("BTO")]
    return set(buys["leg_1"]["contract"])


@pytest.mark.parametrize("entry", [
    lambda s: (s.underlying == "XYZ") & (s.strike <= 0.9 * s.underlying_last),
    lambda s: (s.underlying == "XYZ") & (s.strike <= s.underlying_last * 0.9 - 5),
    lambda s: (s.underlying == "XYZ") & ((s.strike - 5) / s.underlying_last < 0.9),
    lambda s: (s.underlying != "ABC") & (s.strike < 90),
], ids=["literal-left", "chained", "parenthesized", "string-ne"])
def test_dsl_forms_select_only_the_deep_strike(tmp_path, entry):
    assert _bought(_run(tmp_path, entry)) == {"XYZ200717P00080000"}


def test_unparsable_filter_raises(tmp_path):
    with pytest.raises(Exception, match="invalid entry filter"):
        _run(tmp_path, lambda s: Filter("strike >= >= 3"))
