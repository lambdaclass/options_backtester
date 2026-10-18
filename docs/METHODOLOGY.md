# Backtesting options overlays: pitfalls and how this engine handles them

Every item below changed a published or near-published number in this
repository's own research (`research/spitznagel_spy/`). They are not
hypothetical. Read this before trusting a backtest of an options overlay,
from this engine or any other.

## 1. Entry-calendar luck

An overlay that enters every N months has N × (days in a month) possible
calendars, and they do not give the same answer. On SPX 1996–2025, the same
25–30% OTM put overlay at 2%/yr ranged from **−0.84 to +0.72pp/yr** vs buy &
hold depending only on whether entries fell in odd or even months. The "best"
exit DTE (30) won because its December-2008 sale landed on 2008-11-20, the
SPX closing low; exits at DTE 28 or 32 earned about half as much.

**Do:** score every configuration over all phases of its schedule and report
the mean and the worst calendar:

```python
from options_portfolio_backtester import entry_calendars, run_over_calendars, calendar_summary

cals = entry_calendars(stocks_df["date"], every_n_months=2)   # 4 calendars
per_calendar = run_over_calendars(run, cals)                  # run(dates) -> engine
calendar_summary(per_calendar)                                # mean / worst / best / spread
```

A difference smaller than the calendar spread is not a finding. Start-year
matrices do not catch this: if every run starts in January, every run shares
one calendar.

## 2. Dirty option chains

Vendor chains contain rows an entry filter cannot tell apart from real
options:

- **Other products under the same underlying.** CBOE's SPX binary options
  (roots BSK/BSF, 2008–2010) have ordinary strikes and expirations but are
  priced as probabilities. A deep-OTM put filter bought them at $0.19 in
  October 2008, which produced a hedged GFC drawdown *worse* than buy & hold.
- **Flat prints**: one identical bid/ask across a wide strike range.
- **Impossible time value**: a bid far above intrinsic.
- **Crossed markets**: bid above ask.

**Do:** `clean_options_chain(chain, drop_roots=(...), verbose=True)` before
building `HistoricalOptionsData`, and look at what it drops. Inspect the
trades behind any result that looks too good *or too bad*.

## 3. Budget semantics

"2% a year on puts" can mean three different things, and they give different
results:

| setting | meaning |
|---|---|
| `options_budget_annual_pct = 0.02` | 2%/yr, split over the rebalances in the schedule |
| `options_budget_per_rebalance_pct = 0.02` | 2% of NAV *at every rebalance* (24%/yr if monthly) |
| `options_budget_fresh_spend = False` (default) | the budget is a *target value* of puts held: spend = budget − value of open puts, so after a crash payoff nothing new is bought |
| `options_budget_fresh_spend = True` | spend the full budget every time (what "x%/yr on premium" usually means) |

Check `engine.trade_log` to confirm the premium actually spent matches what
you meant.

## 4. Fill rate

A strike band or DTE window that matches nothing does not raise an error. No
hedge is bought, and the backtest silently reports an unhedged portfolio as
"hedged". Deep-OTM SPX puts (40%+) are barely listed before 2003.

**Do:** read `engine.option_fill_rate`; the engine warns
(`HedgeFillWarning`) when it is materially below 1, and
`option_entry_risk_blocked` separates risk-limit rejections from missing
contracts.

## 5. Path accounting

These were bugs in this engine and are now covered by tests in
`tests/oracles/test_daily_exit_accounting.py`. They are worth checking in any
engine:

- **Look-ahead in the NAV path.** With daily exits, each day must be valued
  with that day's holdings. Back-filling a period from end-of-period holdings
  makes drawdown and Sharpe wrong while CAGR looks fine.
- **Premium paid at entry.** Debiting the premium at exit overstates NAV
  for as long as the put is held.
- **Quote gaps.** A held contract missing from one day's data is not an
  expiry. Selling it at intrinsic (zero for an OTM put) throws the hedge
  away on a data hole.
- **Deterministic tie-breaks.** Several contracts can tie on the entry sort.
  Which one gets bought must not depend on row order.

## 6. Financing and leverage

The engine charges no interest on negative cash and pays none on positive
cash. A levered or cash-heavy comparison must add financing:

```python
from options_portfolio_backtester import apply_financing
total = apply_financing(engine.balance, tbill_rate, borrow_spread=0.005)
```

credits the rate on positive cash and charges rate + spread on borrowing. It
is a first-order adjustment: it changes NAV, not the positions the engine
sized. Margin calls are not modeled.

## 7. Compare against the right alternative

"Hedged vs 100% stock" answers one question. "Hedged vs the stock/cash mix
with the same drawdown" answers the one that matters for a risk-reducing
overlay. A hedge that costs 0.1pp/yr and halves crash drawdowns can be cheap
or expensive depending on that comparison.

## 8. Multiple testing

Over a thousand backtests ran on one 30-year history with two paying crashes
(2008, 2020). The best row of a large grid is selected, not discovered.
Pre-register the grid in the script ("do not tune after seeing results"),
report the whole table, and treat differences under ~0.3pp/yr as noise.
