# Data coverage: deep-OTM puts are not listed in the early-2000s SPX chain

The article's headline configuration buys **40–45% OTM** puts at **90–180 DTE**.
On the purchased DeltaNeutral ALLSPX data that depth is **not reliably listed
before ~2003**, so any dot-com (2000–2002) result *at that depth* is largely a
no-hedge artifact, not a hedged outcome. This was found by the engine's hedge
fill-rate diagnostic (`engine.option_fill_rate`; `HedgeFillWarning`).

## Fill rate by OTM band, 2000–2003 (24 bi-monthly rebalances, DTE 90–180)

| OTM band | fills/24 | tradeable (bid>0) |
|---|---|---|
| 10–15% | 24 | 24 |
| 15–20% | 24 | 24 |
| 20–25% | 23 | 23 |
| **25–30%** | **24** | **24** ✅ |
| 30–35% | 19 | 18 |
| 35–40% | 16 | 16 |
| **40–45%** | **13** | **11** ❌ |

At 40–45% the engine attempts 24 entries and **13 go unfilled (46% fill rate)**,
including a contiguous **20-month gap (2001-01 → 2002-09)** that straddles the
core −38% leg of the decline. On several rebalances the chain's deepest listed
90–180-DTE put is only ~38–40% OTM, so a 40–45% strike simply does not exist.

## Consequence for the findings

- **The "dot-com slow grind loses (−1.7pp)" claim is only trustworthy at
  ≤ 25–30% OTM**, where the chain fills every rebalance with sane prices. The
  signal battery (`SIGNAL_EXPERIMENTS.md`) already uses 25–30%, so those
  conclusions stand. The crash-*shape* thesis survives qualitatively, but the
  specific 40–45% dot-com number should be treated as **data-limited**, not a
  clean hedged-vs-unhedged comparison.
- **Fill rate is not the problem in the GFC and COVID windows** — by 2008+ the
  chain lists the full deep-OTM ladder, and those runs fill ≥ 95%. But the GFC
  window *does* contain non-option rows; see the next section.

## Rows in the chain that are not tradeable SPX quotes

Tracing an impossible result (a hedged run with a *worse* GFC drawdown than buy
& hold) to its trades showed the engine buying 361 contracts of
`BSK090321P00725000` at $0.19 on 2008-10-14 — a 27%-OTM "put" at peak crash
volatility, when real ones cost ~$30. BSK and BSF are **CBOE SPX binary
options** (2008-10 → 2010-04): same strikes and expirations as the real chain,
but priced as a probability. A strike/DTE filter cannot tell them apart. Those
"puts" were then marked at $886k on 2009-03-02 and expired worthless, which
manufactured the drawdown.

The chain also holds flat prints — one identical price across a wide strike
range, e.g. 2003-01-02 Jun-03 puts at 64.30/75.10 for strikes 650–725 with SPX
at 909 (one entry paid ~10× fair value), Dec-04 LEAPS at ~$94 for strikes
500–850, and 2025–26 LEAPS stubbed at 5.00/85.00 — plus 206 rows (2001–04)
whose bid exceeds intrinsic by more than 25% of spot.

[`experiments/spx_clean.py`](../experiments/spx_clean.py) drops all three
(4,876 of 9.9M put rows, 0.05%) and every SPX experiment script now loads the
chain through it. Results produced before this (and before the engine
accounting fixes in the same period) are superseded.

## How to detect this yourself

```python
eng.run(...)
print(eng.option_fill_rate, eng.option_entry_unfilled, "/", eng.option_entry_attempts)
```

A `HedgeFillWarning` fires automatically when > 10% of entry attempts match no
contract. Treat a low fill rate as "the data can't express this strategy in
this window" — widen the band, shorten the DTE, or restrict the window.
