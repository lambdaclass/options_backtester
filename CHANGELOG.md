# Changelog

All notable changes to this project are documented here. Entries are
grouped by the kind of change so a reader can quickly find what
matters to them:

- **Behavioral changes** move documented backtest outputs (annual
  return, max drawdown, Sharpe, etc.). If you depend on a published
  reproduction, check this section first when upgrading.
- **API changes** add, remove, or rename public functions, classes, or
  attributes.
- **Tooling** changes the build, install, or CI surface.
- **Performance** changes runtime characteristics without changing
  results (within numeric noise).
- **Internal** refactors, formatting, and documentation that should not
  affect users.

The format follows [Keep a Changelog](https://keepachangelog.com/);
this project does not currently emit version tags, so each section is
anchored on the commit hash that introduced the change.

## Unreleased

### Behavioral changes
- **Fill models apply on both sides.** They priced entries only: exits and
  daily marks always used the bid/ask, and contracts were sized off the
  bid/ask cost. Now the fill model prices entries, exits and marks, and
  sizing uses the price paid. New ``SpreadFraction(fraction)`` fill model
  (0 = mid, 1 = bid/ask). ``SpreadSlippage`` (a cost model the engine never
  applied) warns and points to it. MarketAtBidAsk results are unchanged.
- **Options settle at expiry, every trading day.** Expiry was only detected
  where exit rules ran (rebalance days unless ``check_exits_daily``), and only
  for legs with an exit filter, so an option expiring between rebalances was
  settled at the *next* rebalance's spot (the README quickstart booked 135 of
  147 exits after expiry) or, without an exit filter, never. Now every
  trading day settles legs expiring that day (or before the next trading day:
  older SPX data has Saturday expirations) at intrinsic value. The spot is
  the unadjusted stock close, else the chain's last ``underlying_last``,
  never 0. Exit rules still run only when configured. The golden scenario's
  variant 2 was re-derived (it had pinned the late settlement: 1,975,030 ->
  1,725,030).
- **Quote gaps carry the last mark.** A held contract missing from a day's
  quotes (or a day with no option rows) was marked at intrinsic value, 0 for
  an OTM put: a fake NAV dip, and with ``loss_pct`` a sale at $0. It is now
  marked at its last quote; threshold exits only fire on fully quoted days.
- **Invalid filters fail the run.** A leg filter the Rust compiler could not
  parse used to be dropped silently: an entry filter then matched every
  contract, an exit filter never fired. It now raises with the leg, filter
  kind and query. The parser also accepts what the Python DSL emits: a
  literal on either side (``0.9 * underlying_last``), chained and
  parenthesized arithmetic with normal precedence, unary minus; and
  ``Field != 'str'`` quotes its string like ``==`` does.
- **Daily-exit balance rows are now point-in-time.** With
  ``check_exits_daily=True`` the balance rows of each rebalance period were
  filled in at the *next* rebalance from the then-current holdings, so every
  day before a mid-period exit was valued with the post-exit position (the
  put already gone, its proceeds and any reinvested shares already present).
  Final capital and CAGR were essentially unaffected; max drawdown, Sharpe and
  any windowed drawdown were wrong, most of all around crash monetizations.
  Each day's row is now recorded at that day's close. The Spitznagel article's
  pinned table moves: annual within 0.13pp, Sharpe 0.52–0.64 → 0.66–0.80,
  max DD at 2%/3.3% −30.8/−30.2 → −26.0/−31.8 (see
  ``tests/oracles/test_article_reproduction.py``).
- **Budget-funded premium is paid at entry.** With ``options_budget_*`` set,
  the premium used to net to zero at entry and be deducted from the sale
  proceeds at exit, overstating NAV by the premium of every open put and
  giving the stock leg an interest-free loan. The rebalance's budget is now
  set aside before stocks are bought and debited when the put is bought; exit
  proceeds go to cash in full. Unspent budget stays in cash.
- **A one-day quote gap no longer sells a held option.** A contract missing
  from a day's data was treated as an exit and priced at intrinsic value (0
  for an OTM put). It is now only closed that way once it has expired.
- **Deterministic entry pick.** When several rows tie on ``entry_sort`` (the
  same strike at several expirations) the nearest expiration, then contract
  id, wins, instead of an unspecified row order.
- **Rebalances per year** for the annual budget is now intervals/years
  (was dates/years, overstating it by N/(N−1)).
- **The last trading day is included in the balance** (the final period was
  half-open and dropped it).

### API changes
- ``clean_options_chain`` / ``flag_bad_quotes`` (``data.cleaning``): find and
  drop chain rows that are not tradeable quotes — other products under the
  same underlying (e.g. SPX binaries), flat prints across a strike run,
  impossible time value, crossed markets.
- ``BacktestEngine.option_entry_risk_blocked``: entries a risk constraint
  rejected are now counted separately, and ``HedgeFillWarning`` says so
  instead of blaming the entry filter for finding no contract.
- ``apply_financing`` (``analytics.financing``): credit interest on positive
  cash and charge rate + spread on borrowing, after the run (the engine
  itself charges no interest).
- ``entry_calendars`` / ``run_over_calendars`` / ``calendar_summary``
  (``analytics.robustness``): run one configuration over every phase of an
  every-N-months schedule and report mean / worst / spread, so a result that
  depends on the entry calendar shows up as such.
- ``BacktestEngine.options_budget_fresh_spend`` (default ``False``) exposes the
  existing Rust option: ``False`` treats the budget as a target value for
  options held (spend = budget − value of open puts), ``True`` spends the full
  per-rebalance budget every time. ``use_external_budget`` now also clears a
  stale ``options_budget_per_rebalance_pct``, which would otherwise win.
- ``BacktestEngine.roll_exit_proceeds_pct`` (default ``None``, off): spend
  that fraction of each option sale's proceeds on new contracts the same day,
  from the strategy's entry filter, on top of the regular budget (on a
  rebalance date it is added to that rebalance's budget). Single-strategy
  only; multi-strategy runs raise ``NotImplementedError``. Roll entries are
  not counted in the fill-rate diagnostics.
- **Hedge fill-rate diagnostics.** Every run now records how many rebalances
  attempted to open an option position vs how many matched no tradeable
  contract, exposed as ``engine.option_entry_attempts``,
  ``engine.option_entry_unfilled``, and ``engine.option_fill_rate``. When the
  unfilled fraction exceeds ``BacktestEngine.HEDGE_FILL_WARN_THRESHOLD`` (10%),
  a ``HedgeFillWarning`` is emitted. This catches a silent failure mode where a
  strategy's entry filter (commonly a deep-OTM strike band or long-DTE window)
  finds nothing to buy on some rebalances, degrading the "overlay" into partial
  buy-and-hold and producing misleading conclusions. Concretely: the article's
  40–45%-OTM puts only fill 11 of 24 bi-monthly rebalances on 2000–2003 SPX (a
  20-month gap straddling a −38% decline) because strikes that deep were not
  listed at 90–180 DTE in that era; the chain reliably supports only ~25–30%
  OTM before 2003. The Rust core (``BacktestResult.option_entry_attempts`` /
  ``option_entry_unfilled``) carries the counts across the PyO3 boundary in the
  stats dict.
- **Tearsheet with embedded charts.** ``build_tearsheet`` now accepts
  ``benchmark_balance``, ``trade_log`` (any representation: ``TradeLog``,
  flat frame, or the legacy MultiIndex from ``engine.run()``), and
  ``budget_annual_pct``; ``TearsheetReport`` gains ``charts()``,
  ``to_file()``, and chart-embedding ``to_html()``. New
  ``analytics/options_charts.py`` module with the options-specific
  panels (P&L attribution by leg, premium spend vs budget, crash-window
  zooms, per-trade symlog P&L, options exposure) and new generic charts
  in ``analytics/charts.py`` (equity curve vs benchmark, underwater,
  rolling Sharpe/volatility, Altair allocation chart). New
  ``BacktestResults.returns`` property and ``returns_from_balance``
  helper expose a quantstats-ready daily-returns series; new optional
  dependency group ``reports = [quantstats]``, and ``vl-convert-python``
  added to the ``charts`` extra for fully offline SVG reports.
  ``research/spitznagel_spy/make_figures.py`` regenerates the article's
  figures from the pinned engine configuration. See
  ``docs/REPORTING_ROADMAP.md``.
- **Tearsheet panel redesign after review.** ``pnl_attribution_chart``
  (balance-column deltas) is replaced by
  ``options_pnl_decomposition_chart`` built from actual trades — the old
  panel conflated rebalancing flows with P&L (day-one cash investment
  rendered as a −100% "loss"). The allocation stack (``weights_area_chart``,
  shown as the "Capital allocation" panel) is gated on the balance carrying
  a ``stocks capital`` column, so it is emitted for multi-sleeve strategies
  (e.g. the 97/3 stock+options overlay, where the options sleeve is the
  point) and skipped for single-sleeve runs where it would be
  information-free. Equity curve defaults to log scale with the top-5
  drawdown episodes shaded; the return histogram gets symlog counts plus a
  fitted-normal overlay; new ``trade_return_histogram`` (per-trade return on
  premium), ``holding_period_chart`` ("Holding periods" — days from entry to
  exit), and ``yearly_pnl_chart`` ("Realized P&L by year"); rolling Sharpe
  window widened to 252d; heatmap cells annotated compactly with the color
  domain clamped to ±10% so crash-payoff outliers don't wash out the
  scale. ``tests/oracles/test_article_figures.py`` pins the article's
  figure inputs in CI alongside the published numbers.
- **pyfolio-parity stats.** New ``extended_stats()`` (stability R², omega
  ratio, daily VaR 95%, alpha/beta vs benchmark) appended to the tearsheet
  summary; the stats table gains a side-by-side ``Benchmark`` column when a
  benchmark balance is supplied; new ``stress_events_table()`` — pyfolio's
  "interesting times" view — reports return and max drawdown per crash
  window for strategy and benchmark in the HTML report.
- **Fix ``TradeLog.from_legacy_trade_log`` on Rust engine output.** The
  Rust engine emits the ``order`` column as plain strings (``"BTO"``)
  while the converter compared against the ``Order`` enum, so it
  recovered zero round-trip trades from any modern run. It now accepts
  both representations.

### Behavioral changes
- **Fix externally-funded exit accounting (the "free puts" bug).** In
  externally-funded budget mode (``options_budget_pct`` or
  ``options_budget_annual_pct`` set), ``execute_exits`` now subtracts
  the entry cost from cash on exit. Before this change the engine
  credited put proceeds in full but never debited the entry cost
  anywhere, so every put trade contributed (proceeds) to cash instead
  of (proceeds − cost) = realized P&L. Lifetime cash flow per trade
  is now the realized P&L as it should be.

  Effect on published reproductions: the Spitznagel deep-OTM table
  in ``tests/test_article_reproduction.py`` shifts substantially.
  At 0.5% per-rebalance budget on the 17-year SPY data, annual
  return drops from 13.54% to 10.52%; at 1.0% from 16.29% to 10.11%;
  at 2.0% from 21.34% to 8.71%; at 3.3% from 27.23% to 5.81%. The
  qualitative shape — Sharpe peaks at the lowest budgets and
  degrades monotonically as budget grows — survives the fix.

  Also resolves what was previously documented as "Budget-mode
  capital leak with non-deep-OTM puts" in this CHANGELOG. The
  near-ATM 3.3%-per-rebalance "$3.5B from $1M" case was an
  amplified-by-time symptom of the same exit accounting bug, plus
  the per-rebalance interpretation of ``options_budget_pct`` being
  treated as if it were annual (3.3% per month is ~40%/yr of
  premium spend, not 3.3%/yr). Documented in
  Single authoritative reference for the corrected reproduction is
  now ``docs/SPITZNAGEL_RECONSTRUCTION.md``; the earlier
  ``docs/PRE_FIX_REGIME_SWEEP.md``, ``docs/POST_FIX_REGIME_SWEEP.md``,
  and ``docs/POST_FIX_ANNUAL_SWEEP.md`` were written against engine
  states with one or more of the five bug classes still active and
  have been removed. ``tests/test_known_bugs.py`` removed.

- **Option intrinsic value now uses the unadjusted close (valuation
  bug B).** When a contract is not quoted on a given day (expired or
  missing from the chain), the engine falls back to the option's
  intrinsic value. Previously it computed intrinsic from the
  dividend-*adjusted* close (``stocks_price`` → ``adjClose``) against
  the raw (unadjusted) strike. Because adjusted closes are lower than
  raw closes in the past, this manufactured phantom intrinsic value
  for any expired put whose strike sat inside the adjustment gap. The
  engine now carries the unadjusted close (``close``) alongside the
  adjusted close and uses it for intrinsic value; stock capital / P&L
  still uses the adjusted close. A missing spot now yields $0 intrinsic
  (untradeable) rather than ``strike`` from a 0.0 spot. New
  ``stocks_unadj_price`` schema key (defaults to ``adjClose`` for
  back-compat). Unlike the cash-flow fix above, this is a *valuation*
  bug: both the exit proceeds and the held mark-to-market used the same
  wrong number, so the books stayed balanced — a cash-conservation
  check cannot see it, which is why it needs its own guard.

  Effect: configurations that let contracts reach the intrinsic
  fallback (profit-taking, shorter DTE, closer-to-ATM puts, or any run
  with ``check_exits_daily=False``) lose phantom value. The article's
  deep-OTM 90-180 DTE config exits at real bids near DTE 14, so its
  numbers are largely unaffected.

- **``test_article_reproduction.py`` moved to true-annual budget
  framing.** The reproduction now configures the put overlay with
  ``options_budget_annual_pct`` (and ``check_exits_daily=True``), matching
  the article's "X%/yr" language and ``use_external_budget``, instead of
  the per-rebalance ``options_budget_pct``. Under the corrected engine
  the deep-OTM overlay TRACKS SPY with a small monotonic premium drag
  and Sharpe ≈ buy-and-hold; the earlier "overlay beats SPY / Sharpe
  sweet spot" result was an artifact of the two bugs above. Re-pinned
  table (annual budget, 2008-2024): 0.5%/yr 10.50%, 1.0%/yr 10.34%,
  2.0%/yr 9.99%, 3.3%/yr 9.52% (SPY buy-and-hold 10.65%); full-period
  max drawdown is essentially unchanged from SPY (≈ −51.9%) at these
  budgets. The ``test_spitznagel_sweet_spot_shape`` test is replaced by
  ``test_spitznagel_monotone_drag_and_tracking``.

### Bug fixes
- **``check_exits_daily`` set as an attribute is now honored.** It was
  only a ``run()`` parameter (default ``False``); assigning
  ``engine.check_exits_daily = True`` silently did nothing because
  ``run()`` never read it. ``BacktestEngine`` now has a
  ``check_exits_daily`` attribute and ``run(check_exits_daily=None)``
  falls back to it, so the attribute and the explicit argument agree
  (the argument still overrides per-run). The global default remains
  ``False``.
- **`rebalance_stocks_on_exit` is now reachable from Python.** The Rust
  core already supported redeploying freed cash into stocks immediately
  after a daily option exit (monetize-and-reinvest), but the flag was
  never passed through from `BacktestEngine`, so it was dead. Exposed as
  `BacktestEngine.rebalance_stocks_on_exit` (default ``False``; only
  meaningful with `check_exits_daily=True`).
- **Filter columns ``underlying_last``, ``impliedvol``, and
  ``open_interest`` are no longer silently dropped before Rust
  dispatch.** ``_run_rust`` and ``_run_rust_multi`` previously dropped
  these three options-dataframe columns "to reduce Arrow conversion
  cost." That broke every leg filter referencing them: strike-based
  OTM filters (``schema.strike <= schema.underlying_last * X``)
  silently matched zero rows so no entries fired at all; IV- and
  OI-based filters raised ``RuntimeError: unable to find column …``.
  Silently broke every preset whose depth filter expressed OTM as a
  multiple of spot — ``near_atm_put_protection``, ``Strangle``,
  ``IronCondor``, ``CoveredCall``, ``CashSecuredPut``, ``Collar``,
  ``Butterfly``. The ``deep_otm_put`` preset was not affected because
  it filters by delta. The drop set is now ``{"last", "optionalias"}``
  (only columns that no filter or selector inspects). On the 17-year
  SPY parquet, conversion cost is unchanged within noise.

### Invariants / defense-in-depth
- **Golden hand-computed scenario** (``tests/oracles/test_golden_scenario.py``,
  default suite, ~0.4s): a 9-day synthetic dataset (stock 100 → 50, one
  strike-80 put, 5% per-rebalance external budget) whose correct final
  capital — $1,950,030.00 — is derived by hand-arithmetic in comments
  from first principles (floored share/contract sizing, net-zero
  externally-funded entry, exit credits realized P&L) and asserted to the
  cent, with runtime invariants armed. Two variants cover both exit
  paths: quoted-bid and the expired-contract intrinsic fallback. This is
  the anchor every other guard lacked: it catches value *vanishing*
  (suppressed payoffs) as well as phantom money, because the expected
  number comes from paper arithmetic, not from the engine's past output.
- **README first-example budget misconfiguration fixed.** The example
  called ``use_external_budget(annual_pct=0.005)`` and then also set
  ``options_budget_pct = 0.005`` — and the per-rebalance knob overrides
  the annual one in the engine, so the example silently configured ~6%/yr
  instead of the 0.5%/yr its comment claimed. The redundant line is
  removed; the deprecation of ``options_budget_pct`` (see API changes)
  exists precisely to make this class of misconfiguration loud.
- **Exit-price envelope oracle (independent class-B check).**
  ``tests/heavy/test_invariants.py::TestExitPriceEnvelope`` reconstructs,
  in pandas directly from the raw CSVs, the price bound every trade-log
  exit must satisfy: quoted contracts must exit inside the day's
  [bid, ask]; unquoted (expired/missing) contracts must exit at no more
  than intrinsic computed from the *unadjusted* close. Unlike the
  in-engine ``assert_invariants`` class-B guard — which recomputes
  intrinsic via the same helpers the engine uses and is therefore blind
  to a bug inside them — this oracle shares no code with the engine, so
  the pre-fix adjClose phantom pricing fails it no matter where in the
  engine the wrong price originates. The test also asserts both branches
  (quoted and fallback) are actually exercised, so it cannot silently
  become vacuous.
- **Article-reproduction fixtures clamp to the declared data window.**
  The pinned tables are only valid for 2008-01-02..2024-12-31; the
  fixtures now filter the loaded data to that window, so fetching a
  longer range with ``fetch_data.py`` can no longer shift the pinned
  numbers and masquerade as an engine regression.
- **`BacktestEngine` rejects unknown attributes.** The engine is sealed
  after ``__init__``: assigning an attribute it doesn't declare raises
  ``AttributeError`` (with a did-you-mean suggestion) instead of becoming
  silently-ignored dead config. This closes the whole
  ``engine.check_exits_daily = True``-was-a-no-op bug class at the API
  boundary. All legitimately-settable attributes are pre-declared in
  ``__init__``.
- **Regression-guard oracle tests in the default suite**
  (``tests/oracles/test_regression_guards.py``, small bundled data, runs in CI on
  every push): unknown-config rejection; the runtime class-A/class-B
  invariants exercised on both self- and externally-funded paths; a
  zero-budget run must equal stock-only exactly; an unfilled external
  budget must be fully clawed back (no-fill leak branch); and
  per-rebalance vs annual budget knobs must produce distinct runs
  (catches one knob being rewired to the other's semantics).
- **Optional runtime self-checks (`assert_invariants`).** New
  `BacktestEngine.assert_invariants` attribute (and `assert_invariants`
  config key; default ``False``, zero cost when off) turns on two
  in-engine guards that fail the run loudly on violation:
  - *Class A — cash flow:* on every option exit, the change in portfolio
    cash must equal realized P&L net of commission, computed
    independently of the mutation. Catches the "free puts" regression
    (crediting full proceeds without returning the externally-funded
    entry cost). A cash-conservation check is the right shape for this
    class — a per-trade cash-flow bug unbalances the ledger.
  - *Class B — valuation:* when an unquoted (expired/missing) contract
    falls back to intrinsic value, that value must equal the
    unadjusted-spot intrinsic, recomputed independently. Catches a
    regression back to the adjusted close. A cash check cannot see this
    class — both proceeds and mark use the same wrong price, so the
    ledger stays balanced; it needs its own valuation guard.
  Exercised by ``tests/heavy/test_invariants.py::TestRuntimeInvariantChecks``
  across the budget configurations where both bugs originally lived.

### Budget API clarification
- ``options_budget_pct`` is **per-rebalance**, not annual. On monthly
  rebalancing, ``options_budget_pct = 0.033`` means 3.3% of NAV
  spent on options each month (~40%/yr of premium), not 3.3%/yr.
  Use ``options_budget_annual_pct`` if you want true annual
  semantics, which is what most of the tail-hedge literature
  (including Spitznagel's *Safe Haven*) actually describes. The
  README's first-backtest example and ``BacktestEngine.use_external_budget``
  already use annual semantics. The corrected, consolidated reference
  is ``docs/SPITZNAGEL_RECONSTRUCTION.md``.

### API changes
- **`options_budget_pct` deprecated → `options_budget_per_rebalance_pct`.**
  The bare name reads as if it were annual, and that misreading reached a
  published table once (3.3%/month is ~40%/yr of premium). The new name
  makes the per-rebalance semantics impossible to misread. The old name
  remains a working alias: reads are silent, writes emit a
  ``DeprecationWarning`` pointing at the new name (or at
  ``options_budget_annual_pct`` for %/yr semantics). The Rust config key
  is unchanged.
- **`run(rebalance_dates=...)` — signal-gated entry.** `BacktestEngine.run`
  accepts an explicit list of rebalance dates (snapped to the trading
  calendar), overriding the `rebalance_freq`/`rebalance_unit` schedule, so a
  caller can buy puts only on dates a regime signal is "on". Default behavior
  (``None``) is unchanged. Used by the entry-timing research in
  ``research/spitznagel_spy/signal_experiments.py``.
- `BacktestEngine.use_external_budget(annual_pct)` and
  `BacktestEngine.use_allocation(stocks, options, cash)` —
  first-class helpers that configure the two put-overlay framings in
  the literature's vocabulary. `use_external_budget` is the
  Spitznagel framing (100% stocks + external put budget);
  `use_allocation` is the AQR / allocation-reducing framing.
  Equivalent to the existing attribute-level configuration but
  self-documenting at the call site. (commit `60e6e91`)
- `options_portfolio_backtester.results.BacktestResults` —
  dataclass returned by `BacktestEngine.get_results()`. Bundles
  balance, trade log, config, engine version (read from
  `importlib.metadata`), and an optional data hash. Exposes computed
  properties for annual return, max drawdown, annualized volatility,
  and Sharpe ratio, plus a `summary()` dict. The companion
  `hash_data_file` helper computes the SHA-256 used in `data_hash`.
  (commit `e6605ec`)
- `options_portfolio_backtester.strategy.presets.deep_otm_put` and
  `near_atm_put_protection` — function-style presets for the two
  canonical tail-hedge configurations. `deep_otm_put` matches the
  Spitznagel configuration (delta -0.10 to -0.02, DTE 90-180, exit
  DTE 14). `near_atm_put_protection` matches AQR's PPUT-style
  configuration (~5% OTM, monthly DTE). Compose with the framing
  helpers above. (commit `e6605ec`)
- `HistoricalOptionsData` and `TiingoData` now accept `.parquet`
  files in addition to `.csv` and `.h5`. The SPY option chain loads
  in ~0.4s as parquet vs ~15s as CSV. (commit `e6605ec`)
- `scripts/fetch_data.py` writes both `processed/options.csv` and
  `processed/options.parquet`. Existing CSV-using code continues to
  work unchanged. Pass `--allow-fallback` to permit the secondary
  mirrors (options-data CDN, dataset-hist repo, yfinance) when the
  canonical GitHub Releases source is unreachable; default behaviour
  is now to fail loudly rather than silently substitute different
  bytes. The canonical SPY parquets are pinned by SHA-256 and the
  downloader warns on hash mismatch. (commit `60e6e91`)

### Internal
- **Dead-weight purge; convexity tests re-enabled.**
  - ``tests/convexity/`` (18 tests, 0.08s) now runs in the default suite
    — it was gated by an addopts ignore although the module it covers is
    live API (strategy presets import it) and the tests pass.
  - Deleted ``tests/test_deep_analytics_convexity.py``: permanently
    broken at collection (imports ``_find_target_put``, removed when
    scoring moved to Rust) and hidden by an addopts ignore.
  - Deleted the bt-comparison tooling: ``tests/compat/``,
    ``benchmarks/compare_with_bt.py``, ``benchmarks/benchmark_matrix.py``
    and the ``compare-bt``/``benchmark-matrix``/``parity-gate`` Makefile
    targets. They require the ``bt`` library, which is not (and never
    was) a dependency of this project, so none of it was runnable from a
    fresh checkout. ``/benchmarks`` is gone; ``make rust-bench``
    (criterion) remains the performance benchmark entry point.
  - Deleted dead Makefile targets ``notebooks`` (no ``notebooks/``
    directory exists) and ``walk-forward-report`` (the script it invoked
    does not exist), and the ``bench`` target (no pytest-benchmark tests
    exist); ``install-dev`` no longer installs the ``notebooks`` extra.
- **Directory tidy (renames; import/path updates throughout).**
  - ``tests/bench/`` → ``tests/heavy/`` and the ``bench`` pytest marker
    → ``heavy``: the tier holds data-heavy *correctness* tests, not
    benchmarks. The Makefile target is now ``make test-heavy``.
  - ``tests/test_data/`` → ``tests/fixtures/`` (committed small CSVs;
    distinct from gitignored ``tests/_generated/``).
  - ``data/fetch_data.py``, ``fetch_signals.py``, ``convert_optionsdx.py``
    → ``scripts/``: ``data/`` is now purely storage (``raw/`` cache +
    ``processed/`` outputs); executable tooling lives in ``scripts/``.
    ``fetch_data`` still writes to ``<repo>/data`` regardless of its own
    location.
  - ``options_portfolio_backtester/results.py`` →
    ``options_portfolio_backtester/analytics/results.py`` (its unit test
    moved to the mirrored ``tests/analytics/``). The top-level re-exports
    (``from options_portfolio_backtester import BacktestResults,
    hash_data_file``) are unchanged; only the deep module path moved.
  - Deleted the Makefile ``bench-rust-vs-python`` target (script removed
    earlier) and fixed ``data/README.md``'s stale example importing the
    pre-rename ``backtester.datahandler`` package.
- **Removed the last remnant of the Python valuation engine.**
  ``BacktestEngine._current_options_capital`` and
  ``_get_current_option_quotes`` had no production callers since the
  Rust engine took over valuation; they were kept alive only by a test
  that exercised the dead methods directly
  (``tests/test_intrinsic_sign.py``, also removed — the behavior is
  covered by Rust unit tests, the golden scenario, and the envelope
  oracle). Dead code that *looks* like the live valuation path is
  actively misleading. The ``_intrinsic_value`` helper stays (pure
  function, separately unit-tested).
- **Generated test data no longer dumps into the unit-test package.**
  ``tests/data/`` is the unit-test package for the data module, but
  ``generate_test_data.py`` / ``extract_prod_slices.py`` also wrote
  their large regenerable CSVs there. They now write to
  ``tests/_generated/`` (gitignored).
- **Removed ``setup.cfg``** — its ``[mypy]`` section duplicated
  ``[tool.mypy]`` in ``pyproject.toml`` (which takes precedence in
  mypy's config discovery anyway); two sources of truth for the same
  config.
- **Fixed broken import in ``tests/compat``** —
  ``test_bt_overlap_gate.py`` imported ``scripts.compare_with_bt``;
  the module lives in ``benchmarks/`` (the ``scripts/`` directory no
  longer exists).
- **Test suite reorganized by intent.** The four cross-cutting
  correctness anchors moved from the `tests/` top level into
  `tests/oracles/` (golden scenario, regression guards, engine fuzzing,
  article reproduction), and `tests/README.md` documents the tier model
  (unit mirrors the package; oracles check against engine-independent
  references; `tests/heavy/` is the data-heavy opt-in tier — historically
  misnamed, it holds correctness tests, not benchmarks). All live path
  references (Makefile, CI, docs, reproduce_article.py) updated;
  historical CHANGELOG entries keep their original paths.
- **Removed dead benchmarks.** `benchmarks/benchmark_rust_vs_python.py`,
  `benchmark_large_pipeline.py`, and `benchmark_sweep.py` imported the
  Python-engine dispatch layer deleted in `0951478` and crashed on
  import; their purpose (Rust-vs-Python comparison) no longer exists.
  `benchmark_matrix.py` and `compare_with_bt.py` were removed in a follow-up (below) along with the rest of the bt-comparison tooling.

### Tooling
- **CI now gates merges on the published numbers.** New
  ``article-reproduction`` job in ``.github/workflows/ci.yml`` restores
  the pinned SPY dataset from an actions cache (~600MB, keyed on the
  data-v1 SHA-256 hashes; cache miss re-downloads from the release) and
  runs ``tests/oracles/test_article_reproduction.py`` plus the data-dependent
  invariant/oracle suite on every push and PR. Previously the strongest
  defenses (reproduction pins, runtime invariants at SPY scale, the
  exit-price envelope oracle) only ran locally. Also removed the fast
  ``test`` job's "Download data" step — it called ``fetch_data.py`` with
  a nonexistent ``--force`` flag and without the required
  ``--start/--end``, so it could never have succeeded; the default suite
  is designed to pass with data-dependent tests skipping.
- **Parquet is the canonical processed format.** The SPY-scale test
  fixtures (``tests/oracles/test_article_reproduction.py``,
  ``tests/heavy/test_invariants.py``) now load
  ``data/processed/options.parquet`` instead of the 3.3GB CSV — ~30x
  faster load (~0.4s vs ~15s), cutting the reproduction suite from ~37s
  to ~25s and shrinking the CI cache. The CSV is still written by
  ``fetch_data.py`` for tools that need it; ``data/README.md`` documents
  parquet as canonical.
- **`reproduce_article.py` prints a reproduction fingerprint** — engine
  version, git commit, and SHA-256 of the exact data files — so every
  published table is traceable to the exact code and bytes that produced
  it.
- **Property-based engine fuzzing** (``tests/oracles/test_engine_fuzz.py``,
  default suite, ~1s): hypothesis generates random small option chains
  (including dividend-adjusted closes and contracts that vanish after
  expiry) and random engine configs across allocation/budget/exit-mode
  space, runs each with ``assert_invariants=True``, and asserts the
  engine never violates its own cash-flow/valuation invariants and
  produces structurally sane balances. Every historical bug here was "a
  config nobody had pinned"; this samples the config space instead of
  pinning points.
- **`fetch_data.py` keeps parquet outputs in sync with date alignment.**
  ``align_dates()`` filtered the stock/option CSVs to their shared
  trading days but never rewrote the parquet twins, so
  ``options.parquet`` kept rows on dates absent from ``stocks.csv``.
  The engine asserts stock/option date alignment at ``run()``, so every
  parquet-based run after a fresh fetch failed that assert — including
  ``research/spitznagel_spy/reproduce_article.py``. The filtered frames
  are now also written to the sibling ``.parquet`` files. If your local
  parquet predates this fix, re-run the fetch (or ``align_dates``) once
  to regenerate it. (PR #107)
- `PyO3 abi3-py311` feature enabled in `rust/ob_python/Cargo.toml`.
  The same extension now runs on Python 3.11+ including 3.14. The
  `make install-dev` target also sets
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` as a belt-and-braces fix
  for users on the latest Pythons. (commit `60e6e91`)
- `tests/test_article_reproduction.py` — pins the published
  Spitznagel article's SPY baseline and Spitznagel-framing budget
  table (0.5%, 1.0%, 2.0%, 3.3%) at the engine's current output,
  with tolerance 0.5pp on annual return, 1.0pp on max drawdown, and
  0.05 on Sharpe. Includes a sweet-spot shape test that asserts
  Sharpe peaks at 0.5%-1.0% budget. The CI workflow runs this as a
  dedicated step after the regular pytest pass. `make
  verify-articles` and `make verify-articles-smoke` run the full
  table and the fast qualitative check respectively from the shell.
  (commits `60e6e91`, `e6605ec`)
- `tests/test_results.py` — 8 unit tests for `BacktestResults` and
  `hash_data_file`. Synthetic balance series, no engine or data
  dependencies, runs in ~0.4s. (commit `00070d1`)
- `README.md` first-backtest example switched to the new ergonomic
  API: `deep_otm_put` preset + `use_external_budget` framing +
  `get_results()`. Roughly half the lines of the previous example,
  and every step names its own intent. (commit `6324a32`)

### Performance
- Parquet loading: a 17-year SPY option chain (22M rows) loads in
  ~0.4 seconds via `HistoricalOptionsData("options.parquet")`
  versus ~15.4 seconds via `options.csv`. Disk footprint drops from
  3.2 GB to 593 MB. CI / smoke runs against article reproductions
  speed up accordingly. (commit `e6605ec`)

### Internal
- Module-level logger added to `options_portfolio_backtester.data.providers`.
  Foundation for follow-up structured-logging conversions; the
  package itself does not emit `print()` calls and behaviour is
  unchanged. (commit `e6605ec`)

---

## Behavioral changes pre-Unreleased (historical, from git log)

These are the engine commits that moved documented backtest outputs
in the period before this CHANGELOG was started. Listed here so
downstream reproductions can audit which engine they were generated
against.

- `523ba10` **Fix cash leakage in externally-funded budget path.**
  The externally-funded budget path (Spitznagel framing) was
  inflating returns due to cash leaking in across rebalances. The
  fix correctly accounts cash; post-fix annual returns for typical
  budgets dropped by 2-20 percentage points relative to the pre-fix
  values. Articles that depended on pre-fix numbers — notably the
  Spitznagel tail-hedge piece at federicocarrone.com — have since
  been updated against the corrected engine, and
  `tests/test_article_reproduction.py` pins the post-fix numbers.
- `d42eec0` **Fix budget mode: 100% stock allocation + credit sale
  proceeds.** Earlier the budget mode misallocated proceeds from
  option sales; the fix credits them correctly into the cash bucket.
  Affects any backtest using `options_budget_pct`.
- `5840620` **Fix budget-mode stock allocation: use liquid capital,
  not total capital.** The stock-position sizing in budget mode
  previously sized off total capital including the value of open
  option positions, which double-counted. The fix uses liquid
  capital. Small effect at small budgets, larger as budget grows.
- `b78602d` **Fix budget-mode double-counting of options capital in
  Python engine.** Companion to `5840620`, in the Python engine
  rather than the Rust path. (Now obsolete since the Python engine
  was removed in `0951478` — Rust is the only backend.)
- `ffdfd1d` **Revert full liquidation, fix accounting with cash =
  total - options_capital.** The earlier "full liquidation at
  rebalance" behaviour was reverted in favour of a partial trim;
  cash accounting was also corrected. Affects any multi-rebalance
  backtest.
- `62c4fd5` **Restore `_sell_some_options` to trim excess puts at
  rebalance.** Restores incremental trim behaviour that was lost
  during the Rust port.
- `264e440` **Full options liquidation at rebalance.** The earlier
  step before `ffdfd1d` reverted this. Listed for completeness.

For complete history, see `git log`. The convention going forward
is: any commit that moves a documented backtest output gets a
`Behavioral changes` entry above and a `tests/test_article_reproduction.py`
update in the same PR.
