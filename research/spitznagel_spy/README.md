# Spitznagel SPY/SPX tail-hedge research

Reproductions and robustness studies for the tail-hedge article
([federicocarrone.com](https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/)).

## Layout

```
reproduce_article.py      canonical reproduction — prints every article table
                          + a fingerprint (engine version, commit, data SHAs)
make_figures.py           regenerate the article's charts into figures/
experiments/              exploratory studies (each standalone, skips if data absent)
  spx_common.py           shared setup: data loading, put leg, engine, pool, metrics
  spx_clean.py            drops binary options / garbage quotes from the chain
  # current: fixed engine, cleaned chain, scored over entry calendars
  timing_robustness.py    exit-DTE x entry-calendar check (the exit-30 luck test)
  leverage_monetize.py    leverage x profit-taking, calendar-averaged, financed
  spitznagel_variants.py  monetize-in-crash, short-dated, monthly-ladder families
  rehedge.py              monetize + same-day re-hedge (fixed-point emulation)
  vol_aware.py            monthly ladder that skips purchases when IV is expensive
  roll_proceeds.py        roll part of each put sale's proceeds into new puts
  risk_matched.py         hedge vs the stock/T-bill mix with equal drawdown / vol
  # earlier: single entry calendar -- use for shape, not numbers
  spx_sweep.py            OTM-depth x DTE x budget sweep over 1996-2025
  parallel_sweep.py       400-config OTM x DTE x exit x budget sweep (multiprocess)
  start_year_robustness.py config x start-year matrix
  signal_experiments.py   signal-gated entry + monetization battery
  cross_underlying.py     SPY / QQQ / IWM robustness, window-matched controls
findings/                 written-up conclusions (read these first)
  ROBUSTNESS_REVIEW.md    CURRENT reference — supersedes headline numbers below
  CROSS_UNDERLYING.md
  SIGNAL_EXPERIMENTS.md
  DATA_COVERAGE.md        deep-OTM puts absent from pre-2003 SPX chain
  REGIME_RESULTS_25OTM.md per-regime hedged-vs-unhedged at a fillable depth
figures/                  generated chart output (gitignored)
```

## Data

All scripts need the processed SPX/ETF parquet under `data/processed/`
(see [`../../data/DATA_NOTICE.md`](../../data/DATA_NOTICE.md)). The SPX series
is purchased DeltaNeutral ALLSPX — **local-only, never committed**. Every
script resolves the repo root by searching upward for `pyproject.toml` and
**skips cleanly if the data is absent**, so the repo stays runnable without it.

```
python research/spitznagel_spy/reproduce_article.py
python research/spitznagel_spy/experiments/timing_robustness.py [n_workers]
python research/spitznagel_spy/experiments/roll_proceeds.py [n_workers]
```

A new experiment states its pre-registered grid and a `_run(job)` that calls
`spx_common.put_strategy` / `spx_common.backtest`; `spx_common.run_pool`
handles the data and the process pool (see `roll_proceeds.py`, ~70 lines).

## Key findings (one line each — details in `findings/ROBUSTNESS_REVIEW.md`)

1. **No reliable return edge over 1996-2025.** Averaged over entry calendars
   with the budget actually spent, the best overlays sit between about −0.1
   and +0.25pp/yr vs buy & hold. The earlier "+0.95pp" was one entry calendar
   plus one lucky sale date (2008-11-20, the SPX closing low).
2. **It does buy crash protection at ~break-even cost**, mostly against fast
   crashes. 2%/yr: COVID drawdown −15% vs −34% at every exit date; GFC about
   −50% vs −57% (−42% only at the lucky exit 30).
3. **Small leverage (1.03-1.05x) + a cheap hedge** edges out buy & hold by
   ~0.1-0.25pp/yr with smaller drawdowns, but is ≈ 0 vs the same leverage
   unhedged.
4. **Profit-taking (sell at 3-10x premium) hurts everywhere**; selling well
   before expiry (exit 60) gives up most of the GFC protection.
5. **Crash *shape* decides it.** Fast crashes pay; slow grinds and calm
   decades bleed. Deep (40-45%) OTM bands are not listed pre-2003
   ([`findings/DATA_COVERAGE.md`](findings/DATA_COVERAGE.md)).
8. **At equal risk it is roughly a wash.** Against the stock/T-bill mix
   with the same drawdown or volatility, the hedge's edge is about 0 to
   +0.2pp/yr at 1-2% budgets, at any leverage up to 1.5x; 3.3% loses
   ([`risk_matched.py`](experiments/risk_matched.py)).
6. **No timing signal beat always-on** in the earlier battery — direction
   plausible, numbers predate the engine and data fixes.
7. **Nothing clever beat holding the puts to DTE 30 and putting the payoff
   into stock.** Monetizing near the strike, same-day re-hedging,
   skipping purchases when IV is expensive, and rolling sale proceeds into
   new puts were all flat or worse. Rolling made the GFC drawdown *worse*
   (−57% at full roll): puts bought at the 2008 low, at peak IV, expired
   worthless.
