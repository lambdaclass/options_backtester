# Regime results at 25–30% OTM (the depth the data actually supports)

> **Superseded numbers.** Produced before the engine accounting fixes and the
> chain cleaning, on a single entry calendar. See
> [`ROBUSTNESS_REVIEW.md`](ROBUSTNESS_REVIEW.md) for current results and the
> status of each claim below.

The article's 40–45% OTM band is not listed in the pre-2003 SPX chain
(see [`DATA_COVERAGE.md`](DATA_COVERAGE.md)). 25–30% OTM fills every rebalance
back to 1996, so it is the deepest band that gives a trustworthy hedged-vs-
unhedged comparison across all crash regimes. These are those numbers.

**Config:** SPX, strike-based 25–30% OTM puts, DTE 90–180 entry / exit at
DTE 30, bi-monthly rebalance, daily exit checks, monetize-and-reinvest,
externally-funded annual budget, runtime invariants armed. `Fill` is the
engine's hedge fill-rate (`engine.option_fill_rate`); rows below ~90% are
flagged.

| Regime | Budget | B&H CAGR | Overlay | Excess | B&H DD | Overlay DD | Sharpe | Fill |
|---|---|---|---|---|---|---|---|---|
| **dot-com** 2000–03 | 1.0% | −6.52% | −7.04% | **−0.52pp** | −49.1% | −49.7% | −0.33 | 95% |
| **dot-com** 2000–03 | 3.3% | −6.52% | −8.25% | **−1.73pp** | −49.1% | −51.2% | −0.40 | 95% |
| **GFC** 2007–09 | 1.0% | −11.66% | −9.45% | **+2.21pp** | −56.8% | −54.2% | −0.30 | 79% ⚠ |
| **GFC** 2007–09 | 3.3% | −11.66% | −5.53% | **+6.13pp** | −56.8% | −47.7% | −0.16 | 100% |
| **COVID** 2019–21 | 1.0% | +23.87% | +27.91% | **+4.04pp** | −33.9% | −28.5% | 1.17 | 82% ⚠ |
| **COVID** 2019–21 | 3.3% | +23.87% | +35.06% | **+11.19pp** | −33.9% | −28.5% | 1.14 | 100% |
| **Full** 1996–25 | 1.0% | +8.33% | +8.86% | **+0.53pp** | −56.8% | −49.8% | 0.47 | 99% |
| **Full** 1996–25 | 3.3% | +8.33% | +9.28% | **+0.95pp** | −56.8% | −51.3% | 0.43 | 99% |

## Interpretation

- **Dot-com — hedging genuinely loses.** −0.5 to −1.7pp excess *and* no
  drawdown protection (DD slightly worse). This is a real slow-grind result at
  95% fill, not a coverage artifact. The −1.73pp here is the figure the
  top-level README cites as "dot-com −1.7pp"; it has always been the 25–30%
  number, so that headline is sound — only the separate *40–45%* dot-com run
  was the broken (46%-fill) one.
- **GFC and COVID — hedging pays.** +2 to +6pp (GFC) and +4 to +11pp (COVID),
  with genuine drawdown protection (GFC −56.8 → −47.7; COVID −33.9 → −28.5).
  Fast crashes fall far enough, fast enough, to reach the strikes before the
  90–180-DTE puts roll off.
- **Full 1996–2025 — roughly neutral.** +0.5 to +0.95pp with modest DD help.
  The article's strong full-period result is largely a 2008-start effect; the
  dot-com grind and the calm 2010s offset the crash wins.

Net: the **crash-*shape* thesis holds cleanly at a depth the data supports** —
deep-OTM hedging wins in fast crashes, bleeds in slow grinds, and nets out
near-neutral over a full multi-cycle history.

## Fill-rate caveat (1.0%-budget rows)

The GFC and COVID **1.0%-budget** rows fill only 79% / 82% — **not** missing
strikes but *affordability*: implied vol spiked during those crises, so some
25–30% OTM puts cost more than the thin per-rebalance budget and were skipped
(the `qty == 0` pre-filter). The hedged excess on those two rows is therefore
slightly understated. The **3.3%-budget rows fill 100%** and are the cleaner
read for the crash windows.

## Reproduce

```
python research/spitznagel_spy/experiments/spx_sweep.py dotcom   # needs local ALLSPX data
python research/spitznagel_spy/experiments/spx_sweep.py gfc
python research/spitznagel_spy/experiments/spx_sweep.py covid
```
