# Research Notes: From the Taleb Barbell to Cross-Asset Tail Hedging

**Date:** February 2026
**Data:** SPY options 2008-2025 (~24.7M rows)
**Notebook:** `notebooks/beyond_spitznagel.ipynb`

---

## 1. Can You Make Money Selling Vol + Buying Tail Protection? (No Equity)

The idea sounds elegant: sell near-ATM strangles to collect premium, buy deep OTM strangles for crash insurance, keep everything else in cash. A "pure volatility" portfolio. It's what people imagine Universa does, or what a literal reading of Taleb's barbell would suggest — no equity exposure, just harvest the Variance Risk Premium on one side and buy convexity on the other.

So we tested it. Two separate engines: one runs the short strangle (sell OTM call + put near ATM), the other runs the long strangle (buy deep OTM put + call). Combined by weighted daily returns (97% short / 3% long, matching Universa's rough proportions).

### Individual components

| Component | Annual Return | Max DD | Volatility | Sharpe |
|-----------|-------------|--------|------------|--------|
| Short strangle alone | -1.33%/yr | -22.2% | 3.3% | -1.61 |
| Long OTM alone | -10.39%/yr | -79.1% | 15.6% | -0.92 |

The short strangle is only slightly negative — the VRP is real but small. Around 2-3%/yr gross, and after bid-ask spreads eat their share, you're left with roughly break-even to slightly negative. The long OTM bleed is brutal: deep OTM options decay to zero most months.

### The 10-config parameter sweep

We swept across strike widths, DTE ranges, OTM depths, allocation sizes, and short/long weight ratios. Every configuration lost money:

| Config | Annual Return |
|--------|-------------|
| Wide short + shallow long, 90/10 | -1.59%/yr |
| Tight short + deep long, 97/3 | -1.72%/yr |
| Short only, optimized | -1.25%/yr |
| Long only, 35% OTM | -10.39%/yr |
| Combined "Taleb" (2% width) | -1.59%/yr |
| Combined "Mine" (5% width) | -1.72%/yr |
| High allocation (5%) | -2.41%/yr |
| Wider strangle (15%) | -1.33%/yr |
| Longer DTE (45-60) | -3.86%/yr |
| Most conservative | -1.25%/yr |

Range: **-1.25% to -55.93%/yr**. Not a single positive configuration.

### Why it can't work

The core problem is arithmetic. You're ~97% cash earning 0%, with a small allocation to a negative-sum options market. The VRP is real but tiny: academic estimates put it at 2-3%/yr gross on SPY options. On 3% of your capital, that's 0.06-0.09%/yr. Bid-ask spreads on monthly strangles easily consume 1-2% of notional per round trip, which on 3% allocation is another 0.03-0.06%/yr in drag.

Meanwhile the long OTM side bleeds steadily. At 1% allocation with 97/3 weighting, the effective long weight is 0.03% of portfolio — not enough to matter when a crash does hit, but enough to add 0.3%/yr of drag.

**Conclusion:** Without an underlying return engine (equity, carry, spread), you're playing a negative-sum game. The VRP is too small to harvest profitably at retail, and the tail insurance costs more than it pays over any reasonable horizon.

---

## 2. What About Adding Bonds?

The natural follow-up: if cash earns zero, replace it with bonds. Two candidates: long-duration treasuries (TLT) and short-term bills (SHY/BIL).

### Long bonds (TLT): a historical artifact

The 40-year bull market in bonds (1980-2020, rates from 15% to 0%) flatters every backtest that includes TLT. In our data period (2008-2025), TLT looks decent because it started with the 10-year at ~4% and includes the 2020 flight-to-quality rally.

But the regime is over. TLT was destroyed in 2022-2023 as rates normalized from 0% to 5%. And the key selling point — negative correlation with equities during crises — is regime-dependent, not structural. During the 1970s inflationary period, stocks and bonds fell together. During the 2022 bear market, they fell together again.

The bond-equity negative correlation of 1998-2020 was a feature of a specific monetary regime (falling rates, low inflation, Fed put). Backtesting TLT as a permanent portfolio allocation extrapolates a trend that already reversed.

### Short-term bonds (SHY/BIL): just cash with yield

Short-term treasuries have no duration risk, track the Fed Funds rate, and are uncorrelated with equities. This is what Universa actually holds — essentially cash equivalents that earn whatever risk-free rate is available.

But this doesn't change the fundamental math. At 4% yields, SHY on 97% of portfolio earns ~3.9%/yr. That almost covers the VRP losses, making the overall strategy roughly break-even. At 0% rates (as in 2009-2021), you're right back to losing 1-2%/yr.

**Conclusion:** Short-term bonds are the right base (and what Universa uses), but they don't transform a losing options strategy into a winning one. They just add the risk-free rate.

---

## 3. Dynamic Put Sizing from Bond Yield

One refinement we explored: instead of a fixed 0.5% put budget, fund it dynamically from bond yield. Allocate 20% to SHY and use the yield to buy puts.

### The math

- 20% SHY x 4% yield = 0.80%/yr available for puts
- At 0% rates: almost nothing for protection
- At 5% rates: 1.0%/yr — generous budget

### The timing argument

There's an appealing logic: high rates tend to precede downturns (the Fed hikes to cool the economy, then something breaks). So you'd naturally have more protection going into crises and less coming out of them — exactly when it matters most.

### Why it loses

The problem: you sacrifice 20% equity exposure to fund this. At SPY's ~9-10%/yr real return, that 20% costs you ~1.8%/yr in foregone equity gains. You're saving 0.5%/yr in put cost (the fixed budget you're replacing) while giving up 1.8%/yr in equity returns.

Net cost: approximately -1.3%/yr versus 100% SPY with fixed puts.

### The key insight

0.5% is already so cheap that no funding mechanism is needed. The Spitznagel strategy works precisely because the put cost is negligible relative to equity returns. At 0.5% of portfolio per year, you don't need to "fund" it from anything — the equity return funds it with massive headroom. The simplicity is the feature, not a limitation.

---

## 4. The Spitznagel Trade Across Asset Classes

The confirmed result from this backtester: **100% SPY + 0.5% fixed put budget = 16%/yr CAGR, Sharpe 1.879.** This works because SPY options are liquid, the put cost is tiny, and the crash protection improves geometric compounding by reducing variance drain.

But the *structure* of this trade — earn steady carry in normal times, buy cheap tail protection for extreme moves — isn't unique to equities. The general pattern works anywhere you find: (1) steady carry in normal times, (2) extreme moves rarely but violently, (3) protection underpriced for those extremes.

### Rates — most reliable asymmetry

The Fed reaction function creates the most predictable convexity in finance. They always cut in crises. Always. It's not a probability — it's a policy mandate.

**The asymmetry:** Rates grind higher 25bps at a time over months and years, then collapse violently. 5.25% to 0.25% in months during 2008. 1.50% to 0% in two weeks during COVID. This is a structural feature of central banking: they raise slowly and cut in panic.

**The instruments:** SOFR/eurodollar futures, Treasury futures, swaptions. Long SOFR futures earn near the risk-free rate; OTM calls on rate futures bet on panic cuts.

**Why it's underpriced:** Rate vol models assume mean-reversion around a stable level. They don't properly price "emergency 300bps cut in 6 weeks" scenarios because these events are outside the Gaussian framework the models are built on.

**Ranking:** Best structural asymmetry of any market. The Fed's reaction function is the closest thing to a guaranteed asymmetry in finance.

### FX carry — well-documented crash risk premium

The carry trade is one of the most studied anomalies in finance. High-yield currencies (AUD, MXN, BRL) vs funding currencies (JPY, CHF) earn 3-5% annual rate differential. This differential persists because it compensates for crash risk — and the crashes, when they come, are devastating.

**The crashes:** AUD/JPY fell ~40% in 2008. CHF unpegged from the Euro in January 2015 — a -30% move in minutes that bankrupted multiple brokers. EM currencies can lose 20-50% in weeks during contagion events (1997, 1998, 2018).

**The trade:** Long high-yielder (earn carry), buy OTM puts. The carry funds the protection, and the left tail is genuinely underpriced by Gaussian models. Academic literature strongly supports the existence of a "carry trade crash risk" premium — insurance is cheap because short-vol carry strategies are crowded, creating a persistent risk premium.

**Ranking:** Best risk premium. The carry pays for the protection, and the fat tails are well-documented.

### Credit — spread compression/blowout cycle

Corporate bonds earn a spread over treasuries that compensates for default risk. Credit spreads grind tight for years during economic expansions, then blow out overnight when fear returns.

**The numbers:** Investment-grade CDS went from 50bps to 250bps in 2008. High-yield CDS went from 300bps to 2000bps. These are 5-7x moves in the cost of protection — massive asymmetry.

**The trade:** Hold investment-grade bonds (earn the spread), buy CDS protection on HY or IG index. Spreads grind tight for years then blow out overnight.

**Limitations:** CDS rolls, counterparty risk (ironic for a protection instrument), and the carry is smaller than FX. This is primarily an institutional trade.

### Commodities — supply/demand shocks

Oil is wildly asymmetric. It grinds between $60-80 for years, then spikes to $140 or crashes to $20. Both tails are fat. Agriculture faces weather-driven tail events — a drought in the Midwest can double corn prices in weeks.

**The trades:**
- **Oil:** Earn contango roll yield, buy OTM calls and puts. Both tails pay, and vol is cheap during the grinding periods.
- **Gold:** Steady in normal times, spikes in crises. OTM calls as a chaos hedge.
- **Agriculture:** Weather events create massive tail moves in corn, wheat, soybeans. Insurance is especially cheap because most market participants are hedgers, not speculators.

### Volatility itself — the purest expression

VIX sits at 12-15 for months of calm, then spikes to 40-80 in crises. This is the most direct measure of the phenomenon all the other trades are trying to capture.

**The trade:** Deep OTM VIX calls (or call spreads to reduce cost). You're directly buying "the world gets scary" insurance.

**The catch:** Most expensive of all these trades because everyone knows about this asymmetry. VIX options are priced to reflect the spike potential. But they're still underpriced for true extremes (VIX 80+) because models calibrate to the full distribution, and the 99th percentile events are structurally underweight.

### Emerging market sovereign debt

High yield in normal times, violent contagious crises. The 1997 Asian crisis, 1998 Russian default, 2001 Argentine default, 2015 China devaluation fears, 2018 Turkey/Argentina — EM crises cluster and spread.

**The trade:** Earn EM sovereign spread, buy CDS or OTM puts on EM bond ETFs. The contagion dynamics mean protection on any one EM instrument provides indirect exposure to the whole complex.

---

## 5. The Common Thread

Markets systematically price risk as if the future looks like the recent past. During calm periods:
- Implied vol drops (options get cheap)
- Credit spreads tighten (protection gets cheap)
- Carry trades get crowded (more people selling insurance)
- Risk models say everything is low-risk

Then a shock — geopolitical event, financial contagion, policy error — reprices everything violently. The gap between "normal-times pricing" and "crisis pricing" is the structural edge.

The Spitznagel/Universa insight isn't about equities specifically. It's about **scanning all markets for the cheapest tail convexity and buying it there**. Sometimes that's equity vol (2007). Sometimes it's credit protection (2006). Sometimes it's rate swaptions (2019). The portfolio manager's job is to always own the cheapest crash insurance, wherever it lives.

**Here's a practical ranking by investor type:**

| If you are... | Best market |
|----------------|-------------|
| Retail with simple tools | SPY/QQQ — don't overthink it |
| Can access futures/options | Rates — the Fed asymmetry is unbeatable |
| Running a multi-strategy fund | All of them — rotate to cheapest convexity |

---

## 6. Why SPY + Puts Is Still the Best Retail Implementation

Despite rates and FX having arguably better structural asymmetries, SPY wins for most investors:

1. **SPY options are the most liquid derivatives on earth.** Penny-wide bid-ask, 0-DTE to 2-year expiry, strikes every $1. No other market comes close to this execution quality.

2. **100% SPY + 0.5% fixed puts = 16%/yr, Sharpe 1.879** — confirmed by this backtester over 2008-2025. The equity risk premium provides the return engine. The puts improve geometric compounding by reducing variance drain. The cost is genuinely negligible.

3. **Everything else requires OTC markets, futures accounts, specialized data, and higher transaction costs.** SOFR swaptions? You need an ISDA master agreement. CDS? Same, plus counterparty risk management. FX options? Futures account plus understanding of settlement mechanics.

4. **0.5% is so cheap it doesn't need funding.** No bond yield, no carry trade, no complex allocation formula. Just SPY and monthly puts. The simplicity is the feature, not a limitation.

The entire barbell exploration — selling vol, buying tail protection, adding bonds, dynamic put sizing — was a search for something more sophisticated than "buy SPY, spend 0.5% on puts." Nothing beat it.
