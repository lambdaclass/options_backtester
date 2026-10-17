"""Golden-scenario correctness anchor for the backtest engine.

Every other guard in the suite is *anti-phantom-money*: it catches value
appearing from nowhere. Nothing catches value VANISHING — e.g. a suppressed
put payoff would sail through all of them. This file engineers a tiny
synthetic dataset whose correct final capital is derivable by hand from
first principles, and asserts the engine matches it to the cent.

THE EXPECTED NUMBERS BELOW ARE COMPUTED INDEPENDENTLY FROM THE ACCOUNTING
RULES — NEVER tune them to match engine output. If the engine disagrees,
the engine (or the derivation) has a bug; investigate, don't fudge.

Engine sizing rules (verified by reading the Rust sources, NOT assumed):

* rust/ob_core/src/entries.rs::compute_entry_qty —
      qty = floor(budget / |per_contract_cost|)
  where per_contract_cost = fill_price * shares_per_contract and the
  MarketAtBidAsk fill for a BUY is the ask.

* rust/ob_core/src/backtest.rs::buy_stocks —
      qty = floor(allocation * pct / price)
  i.e. WHOLE shares only; the remainder stays in cash. Price is adjClose
  (schema "stocks_price" maps to adjClose in engine.py).

* rust/ob_core/src/backtest.rs::rebalance_date — budget-funded mode
  (options_budget_pct set): the rebalance's options budget is set aside
  BEFORE stocks are bought, so stocks are sized from
      allocation_stocks * (liquid capital - budget)
  and the premium is debited from cash when the put is bought. A budget
  that buys nothing simply stays in cash.

* rust/ob_core/src/backtest.rs::execute_exits —
      cash += proceeds - exit_commission
  proceeds = bid * spc * qty when the contract is quoted that day; when the
  contract is ABSENT from the day's option rows AND has expired, the exit
  fires and the price falls back to intrinsic value computed from the
  UNADJUSTED close: max(strike - close, 0) for a put. (A contract absent
  before its expiration is a quote gap and is kept.)

------------------------------------------------------------------------
THE SCENARIO (identical economics in both test variants)
------------------------------------------------------------------------

Dates (9 trading days, two business-month starts => 2 BMS rebalances):

    2020-06-01  <- rebalance 1
    2020-06-05, 06-08, 06-12, 06-19, 06-22, 06-26
    2020-07-01  <- rebalance 2
    2020-07-02

Stock XYZ (close == adjClose, splitFactor 1, divCash 0):
    100, 95, 90, 80, 70, 60, 55, 50, 50

One long put on XYZ, strike 80, bought at rebalance 1 (ask 1.00),
disposed of at rebalance 2 for 30.00/contract-share — via the quoted bid
in variant 1, via the intrinsic fallback (80 - 50) in variant 2.

Config: allocation {stocks 1.0, options 0, cash 0}, options_budget_pct
B = 0.05, initial capital 1,000,030.00, NoCosts (zero commissions),
MarketAtBidAsk, shares_per_contract spc = 100.

------------------------------------------------------------------------
HAND DERIVATION (all numbers exact in cents)
------------------------------------------------------------------------

Rebalance 1 (2020-06-01, XYZ @ 100):
  total capital   = 1,000,030.00 (all cash)
  options budget  = B * total capital = 0.05 * 1,000,030 = 50,001.50
  stock buy       : allocation = 1.0 * (1,000,030 - 50,001.50) = 950,028.50
                    qty = floor(950,028.50 / 100) = 9,500 shares
                    cost = 9,500 * 100 = 950,000.00
                    cash after = 50,030.00
  put entry       : per-contract cost = ask * spc = 1.00 * 100 = 100.00
                    qty = floor(50,001.50 / 100.00) = 500 contracts
                    premium 500 * 100 = 50,000.00, paid now
                    cash after = 30.00

Balance row 2020-06-01 (post-rebalance state, marked at exit prices):
  cash            =        30.00
  stocks capital  = 9,500 * 100.00        =   950,000.00
  puts capital    = 500 * bid 0.90 * 100  =    45,000.00
  total capital   = 995,030.00   (the 5,000 below 1,000,030 is the
                                  bid/ask spread on the put)

Balance row 2020-06-12 (XYZ @ 80, put bid 5.00) — variant 1 only:
  total = 30 + 9,500*80 + 500*5.00*100 = 30 + 760,000 + 250,000
        = 1,010,030.00

Rebalance 2 (2020-07-01, XYZ @ 50):
  put exit at 30.00/contract-share (bid in variant 1; intrinsic
  max(80 - 50, 0) = 30.00 in variant 2):
      proceeds      = 30.00 * 100 * 500 = 1,500,000.00
      cash          = 30 + 1,500,000 = 1,500,030.00
  total capital   = 1,500,030 + 9,500 * 50 = 1,975,030.00
  options budget  = 0.05 * 1,975,030 = 98,751.50, set aside
  stock rebuy     : allocation = 1,975,030 - 98,751.50 = 1,876,278.50
                    qty = floor(1,876,278.50 / 50) = 37,525 shares
                    cost = 1,876,250.00, cash after = 98,780.00
  no contract passes the entry filter (DTE too low) => the budget stays
  in cash.

FINAL CAPITAL (any day from 2020-07-01 on; XYZ stays at 50):
  cash 98,780.00 + 37,525 * 50 = 1,975,030.00

Cross-check from first principles:
  stock path:  950,000 of stock -> 9,500 sh @100 falls to @50 = 475,000.00
  cash:        30.00 floored remainder carried along
  put:         premium paid at entry (out of the 1,000,030), proceeds
               30.00 * 100 * 500                              = 1,500,000.00
  total:       475,000 + 30 + 1,500,000                       = 1,975,030.00  OK
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from options_portfolio_backtester import BacktestEngine
from options_portfolio_backtester.core.types import Direction, OptionType, Stock
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData,
    TiingoData,
)
from options_portfolio_backtester.execution.cost_model import NoCosts
from options_portfolio_backtester.execution.fill_model import MarketAtBidAsk
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

# ── Scenario constants (single source of truth for data + derivation) ──

DATES = [
    "2020-06-01",  # rebalance 1 (first trading day of June, BMS)
    "2020-06-05",
    "2020-06-08",
    "2020-06-12",
    "2020-06-19",
    "2020-06-22",
    "2020-06-26",
    "2020-07-01",  # rebalance 2 (first trading day of July, BMS)
    "2020-07-02",
]
STOCK_PRICES = [100.0, 95.0, 90.0, 80.0, 70.0, 60.0, 55.0, 50.0, 50.0]

# Put quotes per date for the quoted-exit variant (bid, ask).
PUT_QUOTES = [
    (0.90, 1.00),    # 2020-06-01: entry at ask 1.00
    (1.50, 1.60),
    (2.00, 2.10),
    (5.00, 5.20),    # 2020-06-12: used in the intermediate balance check
    (11.00, 11.30),
    (20.00, 20.50),
    (25.00, 25.50),
    (30.00, 31.00),  # 2020-07-01: exit at bid 30.00
    (30.00, 31.00),
]

INITIAL_CAPITAL = 1_000_030.00
BUDGET_PCT = 0.05
SPC = 100  # engine default shares_per_contract

# ── Hand-derived expectations (see module docstring; computed from the
#    rules, NOT from engine output) ─────────────────────────────────────

EXPECTED_STOCK_QTY_R1 = 9_500           # floor((1,000,030 - 50,001.50) / 100)
EXPECTED_STOCKS_CAP_R1 = 950_000.00     # 9,500 * 100
EXPECTED_CASH_AFTER_R1 = 30.00          # 1,000,030 - 950,000 - 50,000 premium
EXPECTED_PUT_QTY = 500                  # floor(0.05*1,000,030 / (1.00*100))
EXPECTED_DAY1_PUTS_CAP = 45_000.00      # 500 * 0.90 * 100
EXPECTED_DAY1_TOTAL = 995_030.00        # 30 + 950,000 + 45,000
EXPECTED_JUN12_TOTAL = 1_010_030.00     # 30 + 760,000 + 250,000
EXPECTED_PUT_PNL = 1_450_000.00         # (30.00 - 1.00) * 100 * 500
EXPECTED_STOCK_QTY_R2 = 37_525          # floor((1,975,030 - 98,751.50) / 50)
EXPECTED_CASH_FINAL = 98_780.00         # 1,975,030 - 37,525*50
EXPECTED_FINAL_TOTAL = 1_975_030.00     # 98,780 + 37,525*50


# ── CSV builders ───────────────────────────────────────────────────────

def _write_stocks_csv(path):
    rows = []
    for d, px in zip(DATES, STOCK_PRICES):
        rows.append({
            "symbol": "XYZ", "date": d,
            "open": px, "close": px, "high": px, "low": px,
            "volume": 1_000_000,
            "adjClose": px, "adjHigh": px, "adjLow": px, "adjOpen": px,
            "adjVolume": 1_000_000,
            "divCash": 0.0, "splitFactor": 1.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _option_row(quotedate, optionroot, expiration, strike, bid, ask,
                underlying_last):
    return {
        "underlying": "XYZ",
        "underlying_last": underlying_last,
        "optionroot": optionroot,
        "type": "put",
        "expiration": expiration,
        "quotedate": quotedate,
        "strike": strike,
        "last": bid,
        "bid": bid,
        "ask": ask,
        "volume": 1000,
        "openinterest": 1000,
        "impliedvol": 0.5,
        "delta": -0.1,
        "gamma": 0.01,
        "theta": -0.05,
        "vega": 0.1,
        "optionalias": optionroot,
    }


def _write_options_csv_quoted(path):
    """Variant 1: one strike-80 put expiring 2020-07-17, quoted every day.

    DTE on 2020-06-01 is 46 (entry filter dte >= 40 matches only there);
    DTE on 2020-07-01 is 16 (exit filter dte <= 20 fires; entry filter
    cannot re-fire).
    """
    rows = [
        _option_row(d, "XYZ200717P00080000", "2020-07-17", 80.0, bid, ask, px)
        for d, px, (bid, ask) in zip(DATES, STOCK_PRICES, PUT_QUOTES)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_options_csv_expired(path):
    """Variant 2: strike-80 put expiring 2020-06-26 — BETWEEN the two
    rebalances — so it is ABSENT from the 2020-07-01 rows and the exit must
    take the intrinsic-fallback path: max(80 - unadjusted close 50, 0)
    = 30.00, the same number as variant 1's quoted bid.

    DTE on 2020-06-01 is 25 (entry filter dte >= 25 matches). A dummy
    far-OTM put (strike 40, expiring 2020-07-05) keeps the engine's
    "stock dates == option dates" assertion satisfied on 07-01/07-02; its
    DTE (4, 3) fails the dte >= 25 entry filter so nothing new is bought.
    """
    rows = []
    for d, px, (bid, ask) in zip(DATES[:7], STOCK_PRICES[:7], PUT_QUOTES[:7]):
        rows.append(_option_row(
            d, "XYZ200626P00080000", "2020-06-26", 80.0, bid, ask, px))
    for d, px in zip(DATES[7:], STOCK_PRICES[7:]):
        rows.append(_option_row(
            d, "XYZ200705P00040000", "2020-07-05", 40.0, 0.05, 0.10, px))
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Engine construction ────────────────────────────────────────────────

def _golden_strategy(schema, dte_min, dte_exit):
    strat = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=OptionType.PUT,
                      direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "XYZ") & (schema.dte >= dte_min)
    leg.exit_filter = schema.dte <= dte_exit
    strat.add_legs([leg])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return strat


def _run_golden(tmp_path, write_options_fn, dte_min, dte_exit):
    stocks_csv = tmp_path / "golden_stocks.csv"
    options_csv = tmp_path / "golden_options.csv"
    _write_stocks_csv(stocks_csv)
    write_options_fn(options_csv)

    stocks_data = TiingoData(str(stocks_csv))
    options_data = HistoricalOptionsData(str(options_csv))

    eng = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
        cost_model=NoCosts(),
        fill_model=MarketAtBidAsk(),
    )
    eng.stocks = [Stock("XYZ", 1.0)]
    eng.stocks_data = stocks_data
    eng.options_data = options_data
    eng.options_strategy = _golden_strategy(
        options_data.schema, dte_min=dte_min, dte_exit=dte_exit)
    eng.options_budget_pct = BUDGET_PCT
    eng.assert_invariants = True  # runtime class-A/class-B guards stay on
    eng.run(rebalance_freq=1, rebalance_unit="BMS")
    return eng


# ── Shared assertions (the golden numbers) ─────────────────────────────

def _assert_golden_numbers(eng, check_jun12=False):
    bal = eng.balance
    cents = 0.01

    # Initial synthetic row (day before data starts): exactly the initial
    # capital, all cash.
    assert bal["total capital"].iloc[0] == pytest.approx(
        INITIAL_CAPITAL, abs=cents)

    # Day 1 (2020-06-01), post-rebalance:
    day1 = bal.loc[pd.Timestamp("2020-06-01")]
    # Budget set aside before the stock buy, premium paid at entry: cash
    # is only the stock flooring remainder left after paying the premium.
    assert day1["cash"] == pytest.approx(EXPECTED_CASH_AFTER_R1, abs=cents)
    assert day1["stocks capital"] == pytest.approx(EXPECTED_STOCKS_CAP_R1, abs=cents)
    assert day1["XYZ qty"] == pytest.approx(EXPECTED_STOCK_QTY_R1)
    assert day1["options qty"] == pytest.approx(EXPECTED_PUT_QTY)
    assert day1["options capital"] == pytest.approx(
        EXPECTED_DAY1_PUTS_CAP, abs=cents)
    assert day1["total capital"] == pytest.approx(
        EXPECTED_DAY1_TOTAL, abs=cents)

    if check_jun12:
        # Mid-window mark: stock 80, put bid 5.00.
        jun12 = bal.loc[pd.Timestamp("2020-06-12")]
        assert jun12["total capital"] == pytest.approx(
            EXPECTED_JUN12_TOTAL, abs=cents)

    # Trade log: exactly one entry (BTO) and one exit (STC), 500 contracts.
    qtys = eng.trade_log["totals"]["qty"].to_list()
    assert qtys == [EXPECTED_PUT_QTY, EXPECTED_PUT_QTY], (
        f"expected one entry + one exit of {EXPECTED_PUT_QTY} contracts, "
        f"got qtys {qtys}"
    )
    orders = eng.trade_log["leg_1"]["order"].astype(str).to_list()
    assert orders[0].upper().endswith("BTO") or "BTO" in orders[0].upper()
    assert "STC" in orders[1].upper()

    # Per-contract costs: entry +100.00 (pay ask*spc), exit -3000.00
    # (receive bid*spc / intrinsic*spc). Realized P&L = 1,450,000.
    entry_cost = float(eng.trade_log["leg_1"]["cost"].iloc[0])
    exit_cost = float(eng.trade_log["leg_1"]["cost"].iloc[1])
    assert entry_cost == pytest.approx(100.00, abs=cents)
    assert exit_cost == pytest.approx(-3000.00, abs=cents)
    pnl = (-exit_cost - entry_cost) * EXPECTED_PUT_QTY
    assert pnl == pytest.approx(EXPECTED_PUT_PNL, abs=cents)

    # FINAL CAPITAL — the anchor. A suppressed put payoff would leave
    # ~525,030 here; phantom money would overshoot. Both fail loudly.
    final_total = float(bal["total capital"].iloc[-1])
    assert final_total == pytest.approx(EXPECTED_FINAL_TOTAL, abs=cents), (
        f"engine final capital {final_total:,.2f} != hand-derived "
        f"{EXPECTED_FINAL_TOTAL:,.2f}"
    )

    # Post-rebalance-2 composition: 37,525 shares @ 50 plus the unspent
    # budget in cash.
    last = bal.iloc[-1]
    assert last["XYZ qty"] == pytest.approx(EXPECTED_STOCK_QTY_R2)
    assert last["cash"] == pytest.approx(EXPECTED_CASH_FINAL, abs=cents)
    assert last["options capital"] == pytest.approx(0.0, abs=cents)


# ── Tests ──────────────────────────────────────────────────────────────

class TestGoldenScenario:
    """Hand-computable end-to-end anchor: long put crash hedge pays off."""

    def test_quoted_exit_matches_hand_derivation(self, tmp_path):
        """Variant 1: the put is quoted on the second rebalance date and
        exits at the quoted bid 30.00 (DTE 16 <= exit threshold 20)."""
        eng = _run_golden(tmp_path, _write_options_csv_quoted,
                          dte_min=40, dte_exit=20)
        _assert_golden_numbers(eng, check_jun12=True)

    def test_intrinsic_fallback_exit_matches_hand_derivation(self, tmp_path):
        """Variant 2: the put expires BETWEEN the rebalances and is absent
        from the second rebalance's rows, so the exit must price it at
        intrinsic value from the UNADJUSTED close: max(80 - 50, 0) = 30.00
        — the bug-class-B code path. Same golden numbers as variant 1."""
        eng = _run_golden(tmp_path, _write_options_csv_expired,
                          dte_min=25, dte_exit=10)
        _assert_golden_numbers(eng, check_jun12=False)
