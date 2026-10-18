"""Hand-checkable guards for daily-exit accounting.

Three engine bugs survived the rest of the suite because nothing looked at
them directly:

1. With ``check_exits_daily`` the balance rows of a whole rebalance period
   were filled in at the NEXT rebalance from the then-current holdings, so
   days before a mid-period exit were valued with the post-exit position
   (look-ahead in the NAV path; drawdown and Sharpe were wrong).
2. A held contract missing from one day's quotes was treated as an exit and
   sold at intrinsic value — 0 for an OTM put — discarding the hedge on a
   data hole.
3. With several expirations sharing the best strike, which one got bought
   depended on an unspecified row order.

It also pins ``roll_exit_proceeds_pct`` (re-buying puts with part of a sale's
proceeds the same day) to a hand derivation.

The scenario reuses the golden-scenario data builders: XYZ falls from 100
to 50 over nine trading days and a strike-80 put is bought on 2020-06-01.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from options_portfolio_backtester import BacktestEngine
from options_portfolio_backtester.core.types import Direction, OptionType, Stock
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from tests.oracles.test_golden_scenario import (
    DATES,
    PUT_QUOTES,
    STOCK_PRICES,
    _option_row,
    _write_stocks_csv,
)

INITIAL_CAPITAL = 1_000_030.00
BUDGET_PCT = 0.05
ROOT = "XYZ200717P00080000"   # expires 2020-07-17; DTE 46 on 06-01, 28 on 06-19


def _run(tmp_path, option_rows, dte_exit, sort=None, roll=None):
    stocks_csv, options_csv = tmp_path / "s.csv", tmp_path / "o.csv"
    _write_stocks_csv(stocks_csv)
    # Stable sort: date order for the engine, same-day row order preserved.
    (pd.DataFrame(option_rows).sort_values("quotedate", kind="stable")
     .to_csv(options_csv, index=False))
    options_data = HistoricalOptionsData(str(options_csv))
    schema = options_data.schema

    leg = StrategyLeg("leg_1", schema, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "XYZ") & (schema.dte >= 40)
    leg.exit_filter = schema.dte <= dte_exit
    if sort:
        leg.entry_sort = sort
    strat = Strategy(schema)
    strat.add_legs([leg])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    eng = BacktestEngine({"stocks": 1.0, "options": 0.0, "cash": 0.0},
                         initial_capital=INITIAL_CAPITAL,
                         cost_model=NoCosts(), fill_model=MarketAtBidAsk())
    eng.stocks = [Stock("XYZ", 1.0)]
    eng.stocks_data = TiingoData(str(stocks_csv))
    eng.options_data = options_data
    eng.options_strategy = strat
    eng.options_budget_pct = BUDGET_PCT
    eng.check_exits_daily = True
    eng.rebalance_stocks_on_exit = True
    eng.assert_invariants = True
    eng.roll_exit_proceeds_pct = roll
    eng.run(rebalance_freq=1, rebalance_unit="BMS")
    return eng


def _quoted_rows(skip=()):
    return [
        _option_row(d, ROOT, "2020-07-17", 80.0, bid, ask, px)
        for d, px, (bid, ask) in zip(DATES, STOCK_PRICES, PUT_QUOTES)
        if d not in skip
    ]


def _orders(eng):
    tl = eng.trade_log
    return list(zip(pd.to_datetime(tl["totals"]["date"]).dt.strftime("%Y-%m-%d"),
                    tl["leg_1"]["order"].astype(str)))


class TestPointInTimeBalance:
    def test_rows_before_a_daily_exit_use_pre_exit_holdings(self, tmp_path):
        """Exit DTE 28 fires on 2020-06-19 (a non-rebalance day) at bid
        11.00; the proceeds buy more stock. Hand derivation, as in the
        golden scenario: 9,500 shares and 500 puts after rebalance 1.

          06-12: 30 + 9,500*80 + 500*5.00*100   = 1,010,030.00
          06-19: put sold for 500*11.00*100 = 550,000; total
                 30 + 9,500*70 + 550,000          = 1,215,030.00
                 reinvested: floor(1,215,030/70) = 17,357 shares

        The bug valued 06-12 with the post-exit 17,357 shares and no put
        (about 1.39M instead of 1.01M)."""
        eng = _run(tmp_path, _quoted_rows(), dte_exit=28)
        assert [d for d, o in _orders(eng) if "STC" in o] == ["2020-06-19"]
        bal = eng.balance
        before = bal.loc[pd.Timestamp("2020-06-12")]
        assert before["XYZ qty"] == pytest.approx(9_500)
        assert before["options qty"] == pytest.approx(500)
        assert before["total capital"] == pytest.approx(1_010_030.00, abs=0.01)
        exit_day = bal.loc[pd.Timestamp("2020-06-19")]
        assert exit_day["XYZ qty"] == pytest.approx(17_357)
        assert exit_day["options capital"] == pytest.approx(0.0, abs=0.01)
        assert exit_day["total capital"] == pytest.approx(1_215_030.00, abs=0.01)

    def test_last_trading_day_is_in_the_balance(self, tmp_path):
        eng = _run(tmp_path, _quoted_rows(), dte_exit=28)
        assert eng.balance.index[-1] == pd.Timestamp(DATES[-1])


class TestQuoteGap:
    def test_one_missing_quote_does_not_sell_the_put(self, tmp_path):
        """Drop the put's 2020-06-08 row (the contract is still listed until
        07-17). It must be held through the gap and sold on 06-19 as usual,
        not dumped on 06-08 at intrinsic max(80 - 90, 0) = 0. A dummy row
        keeps the date in the option data."""
        rows = _quoted_rows(skip={"2020-06-08"})
        rows.append(_option_row("2020-06-08", "XYZ200717P00010000", "2020-07-17",
                                10.0, 0.0, 0.05, 90.0))
        eng = _run(tmp_path, rows, dte_exit=28)
        exits = [(d, o) for d, o in _orders(eng) if "STC" in o]
        assert [d for d, _ in exits] == ["2020-06-19"]
        assert float(eng.trade_log["leg_1"]["cost"].iloc[-1]) == pytest.approx(-1_100.00)


class TestEntryTieBreak:
    def test_same_strike_picks_nearest_expiration(self, tmp_path):
        """Two strike-80 puts pass the entry filter on 06-01: Jul-17 (DTE 46)
        and Sep-18 (DTE 109). Sorting by strike ties; the nearer expiry
        wins, deterministically, regardless of row order in the data."""
        far = [_option_row(d, "XYZ200918P00080000", "2020-09-18", 80.0, bid, ask + 1, px)
               for d, px, (bid, ask) in zip(DATES, STOCK_PRICES, PUT_QUOTES)]
        for rows in (far + _quoted_rows(), _quoted_rows() + far):
            eng = _run(tmp_path, rows, dte_exit=28, sort=("strike", False))
            assert eng.trade_log["leg_1"]["contract"].iloc[0] == ROOT


class TestRollProceeds:
    SEP = "XYZ200918P00040000"

    def _rows(self):
        # A strike-40 Sep put, listed from 06-19 on (so 06-01 can only buy Jul).
        sep = [_option_row(d, self.SEP, "2020-09-18", 40.0, 1.80, 2.00, px)
               for d, px in zip(DATES, STOCK_PRICES) if d >= "2020-06-19"]
        return _quoted_rows() + sep

    def test_half_the_proceeds_buy_new_puts_the_same_day(self, tmp_path):
        """The Jul put is sold on 06-19 for 550,000 (as above). With roll 0.5:

          new puts: floor(275,000 / (2.00*100)) = 1,375 Sep-40s, cost 275,000
          cash 30 + 550,000 - 275,000 = 275,030; with 9,500*70 = 665,000 of
          stock that is 940,030 liquid -> floor(940,030/70) = 13,429 shares
          06-19 total: 13,429*70 + 1,375*1.80*100 = 940,030 + 247,500
                     = 1,187,530.00 (the 27,500 drop from 1,215,030 is the
                     Sep put's bid-ask spread)"""
        eng = _run(tmp_path, self._rows(), dte_exit=28, roll=0.5)
        tl = eng.trade_log
        bought = tl[(pd.to_datetime(tl["totals"]["date"]) == "2020-06-19")
                    & tl["leg_1"]["order"].astype(str).str.contains("BTO")]
        assert list(bought["leg_1"]["contract"]) == [self.SEP]
        assert float(bought["totals"]["qty"].iloc[0]) == pytest.approx(1_375)
        day = eng.balance.loc[pd.Timestamp("2020-06-19")]
        assert day["XYZ qty"] == pytest.approx(13_429)
        assert day["options capital"] == pytest.approx(247_500.00, abs=0.01)
        assert day["total capital"] == pytest.approx(1_187_530.00, abs=0.01)

    def test_off_by_default(self, tmp_path):
        eng = _run(tmp_path, self._rows(), dte_exit=28)
        day = eng.balance.loc[pd.Timestamp("2020-06-19")]
        assert day["XYZ qty"] == pytest.approx(17_357)
        assert day["total capital"] == pytest.approx(1_215_030.00, abs=0.01)
