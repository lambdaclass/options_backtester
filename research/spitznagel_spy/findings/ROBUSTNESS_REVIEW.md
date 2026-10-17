# Robustness review: what survives (Oct 2026)

This is the current reference for the SPX tail-hedge study. It supersedes the
headline numbers in the other files in this directory and in
[`docs/SPITZNAGEL_RECONSTRUCTION.md`](../../../docs/SPITZNAGEL_RECONSTRUCTION.md)
wherever they disagree.

**Bottom line.** Over SPX 1996–2025, a deep-OTM put overlay does not
reliably beat buy & hold on return. Scored fairly (averaged over entry
calendars, full budget actually spent), the best configurations land between
about −0.1 and +0.25pp/yr versus buy & hold, while cutting crash drawdowns
substantially (GFC −57% → −42%, COVID −34% → −15% at a 2% budget). The
earlier "+0.95pp" optimum was an artifact of one entry calendar and one sale
date. Profit-taking ("sell at k× premium") never helped.

## What changed underneath the old numbers

Every earlier result was produced with at least one of these problems:

| Problem | Effect | Fix |
|---|---|---|
| Daily-exit balance rows back-filled from end-of-period holdings | Drawdown, Sharpe and windowed DDs wrong (CAGR ~fine) | Engine records each day at its close |
| Budget-funded premium paid at exit, not entry | NAV overstated by open premia | Engine debits premium at entry |
| Budget = *target value* of puts held, not a spend | Real spend < nominal; no new hedge after a payoff | Research scripts set `options_budget_fresh_spend = True` |
| One-day quote gap force-sold the put at intrinsic (≈0) | Hedge lost on data holes | Engine keeps it until expiry |
| CBOE SPX **binary options** (BSK/BSF, 2008–10) in the chain | Bought as $0.19 "puts" in Oct 2008, marked like puts | [`spx_clean.py`](../experiments/spx_clean.py) drops them + flat/garbage prints |
| Single entry calendar (1st trading day of odd months) | ±0.8pp swing from calendar alone | Score over 4 calendars, all starting invested on day 1 |

Engine fixes: see `CHANGELOG.md` (Unreleased). Data: see
[`DATA_COVERAGE.md`](DATA_COVERAGE.md).

## 1. The exit-30 optimum is luck

[`timing_robustness.py`](../experiments/timing_robustness.py), 25–30% OTM,
90–180 DTE entry, 1996–2025. Excess CAGR vs buy & hold (pp), **mean over four
entry calendars** (odd/even months × 1st/10th trading day):

| exit DTE | 1% | 2% | 3.3% |
|---|---|---|---|
| 21 | −0.37 | −0.84 | −1.49 |
| 25 | −0.23 | −0.59 | −1.13 |
| 28 | −0.29 | −0.67 | −1.23 |
| **30** | **+0.04** | **−0.09** | **−0.39** |
| 32 | −0.25 | −0.60 | −1.13 |
| 45 | −0.47 | −0.98 | −1.67 |

- **Calendar:** at exit 30 / 2%, odd-month entries give +0.72 / +0.30, even
  months −0.84 / −0.55. Every older result used the odd-month calendar.
- **One day:** exit 30 beats 28 and 32 in every calendar. The Dec-2008 puts
  hit DTE 30 on 2008-11-20, the SPX closing low; selling two days earlier or
  later roughly halves that payoff (2007–09 probe, 2% budget: $36.3k vs
  $19.7k / $18.8k of proceeds on $1M). Exits 28/32 are the honest neighbours: −0.6 to −0.7pp at 2%.
- The start-year matrix ([`start_year_robustness.py`](../experiments/start_year_robustness.py))
  cannot detect this: every start year begins in January, so all of them
  share the odd-month calendar.

## 2. Leverage and monetization (Spitznagel's actual claim)

[`leverage_monetize.py`](../experiments/leverage_monetize.py): 1.0/1.03/1.05×
equity (borrowing at T-bill + 0.5%, cash earning T-bill), budgets 1%/2%,
exits 30/45/60, sell at 3×/5×/10× premium or hold; all calendar-averaged.

| config | CAGR | vs B&H | vs unhedged same lev | max DD | GFC DD | COVID DD | Sharpe |
|---|---|---|---|---|---|---|---|
| unhedged 1.00× (B&H) | 8.33 | — | — | −56.8 | −56.8 | −33.9 | 0.436 |
| unhedged 1.05× | 8.54 | +0.21 | — | −58.8 | −58.8 | −35.5 | 0.424 |
| 1.05×, 1%, exit 30, hold | 8.57 | **+0.24** | +0.03 | −54.7 | −51.7 | −24.3 | 0.464 |
| 1.05×, 2%, exit 30, hold | 8.48 | +0.14 | −0.06 | −53.0 | −44.3 | −15.8 | 0.480 |
| 1.00×, 2%, exit 30, hold | 8.24 | −0.09 | −0.09 | −50.6 | **−42.3** | **−14.6** | **0.492** |
| 1.00×, 2%, exit 60, hold | 8.12 | −0.21 | −0.21 | −52.9 | −52.9 | −24.0 | 0.457 |

- **Small leverage + a cheap hedge** does edge out buy & hold with smaller
  drawdowns, but by ~0.1–0.25pp/yr, and only vs B&H — versus the same
  leverage unhedged it is ≈ 0.
- **Profit-taking hurts in every cell** (−0.3 to −0.75pp): selling at 3–10×
  monetizes before the crash finishes and gives back the COVID protection
  (COVID DD ≈ −34%, same as unhedged).
- **Exit 60 (selling well before expiry)** costs ~0.1–0.2pp vs exit 30 and
  loses most of the GFC protection.
- The exit-30 rows carry the one-day luck of §1; discount them accordingly.

## 3. What the hedge does buy

Crash protection at roughly break-even cost. At 2%/yr, calendar-averaged:
GFC drawdown −42% vs −57%, COVID −15% vs −34%, Sharpe 0.49 vs 0.44, for
−0.09pp/yr. That is a risk-reduction result, not a return-enhancement one.

## 4. Spitznagel-style mechanisms

[`spitznagel_variants.py`](../experiments/spitznagel_variants.py): pre-registered
families built on what Spitznagel / Universa describe, 1.0× equity, full budget
spent, scored over entry calendars (4 bi-monthly, 2 monthly). Excess vs B&H,
pp/yr, mean (worst…best calendar):

| strategy | 1% | 2% | GFC DD @2% | COVID DD @2% |
|---|---|---|---|---|
| **core** 25–30% OTM, 90–180 DTE, sell at DTE 30, bi-monthly | **+0.04** (−0.39…+0.50) | −0.09 (−0.84…+0.72) | **−42%** | **−15%** |
| core, monthly ladder (half size) | −0.08 (−0.26…+0.09) | −0.26 (−0.57…+0.04) | −47% | −13% |
| + monetize when strike ≥ 0.95·spot | −0.09 | −0.28 | −52% | −18% |
| + monetize when strike ≥ 0.90·spot | −0.17 | −0.41 | −54% | −21% |
| + monetize when strike ≥ 0.85·spot | −0.41 | −0.86 | −55% | −28% |
| short-dated 50–100 DTE, sell at DTE 14 (20–30% OTM, ± monetize) | −0.51…−0.57 | −1.06…−1.18 | −54…−56% | −19…−31% |

- **Monetizing in the crash hurts, and the earlier it fires the worse.**
  Selling once SPX nears the strike cashes out partway down; in 2008 the
  market kept falling for months while the next hedge waited for the next
  scheduled entry. *Caveat:* the engine only opens positions on scheduled
  entry dates, so Universa-style immediate re-hedging after a monetization is
  not tested — that is the one variant here that could change the verdict.
- **Short-dated puts cost too much.** Rolling ~2-month puts pays more decay
  per year for similar crash payoffs.
- **The monthly ladder is the realistic number:** a narrower calendar range
  and ≈ −0.1 to −0.3pp/yr for real crash protection.

## Status of earlier findings

| Claim (file) | Status |
|---|---|
| "+0.95pp over full history at 25–30/90–180/exit 30" (README, REGIME_RESULTS) | **Superseded** — single calendar + one sale date; calendar-averaged ≈ 0 |
| Article table 2008–2024 SPY, +6pp/yr (RECONSTRUCTION) | **In-sample** — starts just before the GFC; Sharpe/DD columns also moved after the balance fix (see `tests/oracles/test_article_reproduction.py`) |
| "Monthly rolling halved the excess" (full_sweep.py) | **Unreliable** — monthly = both bi-monthly calendars, so halving is what calendar luck predicts |
| "No timing signal beats always-on" (SIGNAL_EXPERIMENTS) | **Direction plausible, numbers superseded** — baseline and signals all on the odd calendar; gated runs also spend more per entry |
| "Monetization hurt −0.7pp" (SIGNAL_EXPERIMENTS) | **Confirmed** on the fixed engine, calendar-averaged (§2) |
| Crash shape decides it (README) | **Stands qualitatively** — fast crashes pay, slow grinds and calm decades bleed |
| Deep-OTM bands unlisted pre-2003 (DATA_COVERAGE) | **Stands** |
| OTM/DTE/budget optima (parallel_sweep_results.csv) | **Single-calendar** — use for shape, not for picking a config |

## Caveats

- Only ~2 crashes paid (2008, 2020) in 30 years; every number here leans on
  them. Treat differences under ~0.3pp/yr as noise.
- SPX price index, no dividends, on both sides.
- Margin calls are not modeled for the levered runs.
- Well over 1,000 backtests have been run on this one history; any single
  "best" row is selected, not discovered.
