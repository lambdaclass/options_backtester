"""Fill models price entries, exits and daily marks alike.

The fill model used to apply only when buying: exits and marks always used
the bid/ask, and contracts were sized off the bid/ask cost. Under MidPrice a
round trip in an unchanged 1.00/3.00 market bought at 2.00 and sold at the
1.00 bid.

Scenario: one XYZ put quoted 1.00/3.00 every day, bought on 2020-06-01 with a
5% budget of 1,000,030 (50,001.50) and sold on 2020-06-19 (DTE 28).
"""

import math

import pandas as pd
import pytest

from options_portfolio_backtester import BacktestEngine, MarketAtBidAsk, MidPrice, SpreadFraction
from options_portfolio_backtester.core.types import Direction, OptionType, Stock
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from tests.oracles.test_golden_scenario import DATES, STOCK_PRICES, _option_row, _write_stocks_csv

BUDGET = 0.05 * 1_000_030.00


def _run(tmp_path, fill_model):
    stocks_csv, options_csv = tmp_path / "s.csv", tmp_path / "o.csv"
    _write_stocks_csv(stocks_csv)
    rows = [_option_row(d, "XYZ200717P00080000", "2020-07-17", 80.0, 1.00, 3.00, px)
            for d, px in zip(DATES, STOCK_PRICES)]
    pd.DataFrame(rows).to_csv(options_csv, index=False)
    options_data = HistoricalOptionsData(str(options_csv))
    sch = options_data.schema
    leg = StrategyLeg("leg_1", sch, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (sch.underlying == "XYZ") & (sch.dte >= 40)
    leg.exit_filter = sch.dte <= 28
    strat = Strategy(sch)
    strat.add_legs([leg])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    eng = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=1_000_030.00,
                         cost_model=NoCosts(), fill_model=fill_model)
    eng.stocks = [Stock("XYZ", 1.0)]
    eng.stocks_data = TiingoData(str(stocks_csv))
    eng.options_data = options_data
    eng.options_strategy = strat
    eng.options_budget_pct = 0.05
    eng.check_exits_daily = True
    eng.assert_invariants = True
    eng.run(rebalance_freq=1, rebalance_unit="BMS")
    return eng


@pytest.mark.parametrize("fill_model, buy, sell", [
    (MarketAtBidAsk(), 3.00, 1.00),
    (MidPrice(), 2.00, 2.00),
    (SpreadFraction(0.5), 2.50, 1.50),
], ids=["bid-ask", "mid", "half-spread"])
def test_round_trip_prices_and_marks(tmp_path, fill_model, buy, sell):
    eng = _run(tmp_path, fill_model)
    tl = eng.trade_log
    qty = math.floor(BUDGET / (buy * 100))       # sized on the price paid
    assert tl["totals"]["qty"].tolist() == [qty, qty]
    assert tl["leg_1"]["cost"].tolist() == pytest.approx([buy * 100, -sell * 100])
    # Held on 06-12: marked at the exit price under the same model.
    held = eng.balance.loc[pd.Timestamp("2020-06-12")]
    assert held["options capital"] == pytest.approx(qty * sell * 100, abs=0.01)
    assert pd.Timestamp(tl["totals"]["date"].iloc[1]) == pd.Timestamp("2020-06-19")


def test_spread_fraction_bounds():
    with pytest.raises(ValueError):
        SpreadFraction(1.5)
