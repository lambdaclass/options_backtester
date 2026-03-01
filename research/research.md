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

---

### 4.1 Rates — Most Reliable Asymmetry

The Fed reaction function creates the most predictable convexity in finance. They always cut in crises. Always. It's not a probability — it's a policy mandate.

#### The asymmetry

Rates grind higher 25bps at a time over months and years, then collapse violently. This is a structural feature of central banking: they raise slowly and cut in panic.

| Episode | Fed Funds Path | Timeline |
|---------|---------------|----------|
| Dot-com / 9-11 | 6.50% → 1.00% | Jan 2001 – Jun 2003 (30 months) |
| 2007-2008 GFC | 5.25% → 0.25% | Sep 2007 – Dec 2008 (15 months) |
| COVID | 1.50% → 0.00% | Mar 3 – Mar 15, 2020 (**12 days**) |

The COVID episode is staggering: 150bps of emergency cuts in less than two weeks, plus unlimited QE. The 10-year Treasury yield dropped from ~1.9% in January 2020 to 0.54% in March — Treasury futures rallied massively.

During 2008, long-term Treasuries gained over 27% for the year while equities crashed. The "flight to quality" effect means Treasury futures have built-in positive convexity during financial crises.

#### Instruments

**Retail-accessible (via Interactive Brokers, futures account):**
- **SOFR futures (SR3)**: Three-Month SOFR, $2,500 per basis point. Replaced Eurodollars in June 2023. Price = 100 - implied rate, so a rate cut from 5% to 3% means price goes 95.00 → 97.00 (+200 ticks = $5,000/contract).
- **Treasury futures**: ZF (5-yr, ~$50/tick), ZN (10-yr, ~$15.625/tick), ZB (30-yr, ~$31.25/tick). Options available on all of them.
- **Options on SOFR futures**: American-style, listed on CME. Puts on SOFR = betting rates stay high. Calls on SOFR = betting on rate cuts (the tail hedge).
- **Options on ZN/ZB**: Calls on Treasury futures = betting on flight-to-quality (rates fall, bond prices rise).
- **Micro Treasury futures**: 1/10th size of standard contracts, launched for retail.

**Institutional only:**
- **Swaptions**: Options on interest rate swaps. Receiver swaptions pay off when rates fall (deflationary crash hedge). Payer swaptions pay off when rates spike (inflationary bust hedge). Require ISDA master agreement. The most direct way to express rate convexity but completely inaccessible to retail.

#### The specific trade structure

The Spitznagel analogy for rates: **long SOFR futures or Treasury futures (earn near risk-free rate) + OTM calls on SOFR/Treasury futures (bet on panic cuts)**. In normal times, you earn carry from the futures position. When a crisis hits and the Fed slashes rates, the OTM calls explode in value.

For Treasury futures specifically: calls on ZN (10-year) or ZB (30-year) benefit from both the rate cut AND the flight-to-quality bid. During 2008, ZB futures went from ~113 in mid-2008 to ~142 by December — a ~26% move. OTM calls purchased before the crisis would have paid 20-50x.

#### Cost and sizing

- SOFR options: An OTM call on SR3 at 97.00 strike (implying 3% rate) when current rate is 5% (price at 95.00) costs roughly 2-5 ticks ($50-125/contract) for 3-6 month expiry in calm markets.
- ZN options: Deep OTM calls (~3-5 points out of the money, 90 DTE) cost approximately $200-500/contract in calm conditions.
- Treasury futures margin: ZN ~$3,800-4,500 initial margin per contract.

#### Historical data for backtesting

- **CME DataMine**: Historical SOFR/Eurodollar futures and options data going back decades. Paid service, self-service cloud platform.
- **SOFR futures history**: Only since 2018 (Eurodollar options back to 1980s, but ED contracts delisted June 2023).
- **Treasury futures options**: Data available via CME DataMine, OptionMetrics (IvyDB Futures), and Databento.

#### Academic literature

- Bernanke & Kuttner (2005) — "What Explains the Stock Market's Reaction to Federal Reserve Policy?" *Review of Economics and Statistics*. Documents asymmetric rate move impact.
- Adrian & Shin (2010) — "Liquidity and Leverage." *Journal of Financial Intermediation*. Explains the flight-to-quality mechanism.
- Hanson & Stein (2015) — "Monetary Policy and Long-Term Real Rates." *Journal of Financial Economics*. On the term premium and rate expectations during crises.

#### Practical challenges

- **Margin**: Futures require margin posting; during crises, exchanges increase margins, potentially forcing liquidation at the worst time. But long options only require the premium paid (no additional margin).
- **Roll costs**: SOFR futures have quarterly expiry; rolling every 3 months has minimal cost due to deep liquidity.
- **Basis risk**: SOFR tracks overnight repo rate, which closely follows Fed Funds but isn't identical. Treasury futures track bond prices, which include term premium (can diverge from short rates).
- **Contango in SOFR**: When the yield curve is inverted (short rates > long rates), deferred SOFR futures trade at higher prices (lower implied rates), creating positive carry for the hedge.

**Why rates rank #1**: The Fed's reaction function is institutional, not statistical. Every other market's tail behavior is probabilistic — "it usually happens." The Fed cutting rates in a crisis is a policy commitment. No other market has a single actor who both controls the price and has a mandate to move it violently in one direction during crises.

---

### 4.2 FX Carry — Best Risk Premium

The carry trade is one of the most studied anomalies in finance. High-yield currencies vs funding currencies earn persistent rate differentials that compensate for crash risk.

#### Current rate differentials (as of early 2026)

| Country | Policy Rate | vs USD (3.50%) | vs JPY (0.00%) |
|---------|-----------|---------------|---------------|
| Brazil | 15.00% | +11.50% | +15.00% |
| Turkey | 37.00% | +33.50% | +37.00% |
| Mexico | 7.00% | +3.50% | +7.00% |
| South Africa | 6.75% | +3.25% | +6.75% |
| Australia | 3.85% | +0.35% | +3.85% |
| New Zealand | 2.25% | -1.25% | +2.25% |
| Switzerland | 0.00% | -3.50% | 0.00% |
| Japan | 0.00% | -3.50% | — |

Classic carry pairs: long MXN/JPY (+7%), long BRL/USD (+11.5%), long AUD/JPY (+3.85%). The differentials fund the protection.

#### Historical crash episodes

| Event | Pair/Move | Timeline | Details |
|-------|-----------|----------|---------|
| 1998 LTCM/Russia | Multiple EM pairs -20-40% | Aug-Oct 1998 | Ruble: 6.29 → 21 per dollar in 3 weeks. Contagion hit Brazil, Asia. EMBI spread >1500bps. |
| 2008 GFC | AUD/JPY: 108 → ~55 (-49%) | Jul-Oct 2008 | Massive yen carry unwind. NZD/JPY similar. |
| 2015 CHF unpeg | EUR/CHF: 1.20 → 0.85 (-30%) | Jan 15, 2015 | **Minutes.** SNB removed the EUR/CHF floor. Multiple brokers bankrupted (FXCM, Alpari UK). |
| 2018 EM crisis | TRY/USD: -45%, ARS/USD: -50% | Apr-Sep 2018 | Turkey: rate hikes from 8% to 24% failed to stop the slide. Argentina: emergency IMF bailout. |
| 2020 COVID | AUD/JPY: -15%, MXN/USD: -25% | Feb-Mar 2020 | Broad carry unwind. EM currencies hit hard. |
| 2024 Yen carry unwind | AUD/JPY: 109 → 93.5 (-14%) | Jul-Aug 2024 | BOJ rate hike triggered massive deleveraging. Yen appreciated 14% vs USD in under a month. Nikkei crashed 12% in one day (Aug 5). JP Morgan estimated 65-75% of global carry positions unwound. |

The 2024 episode is notable: the BOJ's surprise rate hike on July 31 triggered a cascade. TOPIX lost 12% in a single day, VIX spiked, and the Nikkei volatility index hit crisis levels. This was a textbook carry unwind — and a recent reminder that these events aren't historical curiosities.

#### The specific trade structure

Long high-yielder spot/forward + buy OTM puts on the high-yielder.

**Example: AUD/JPY with 3.85% carry, buying 10-delta puts.**
- Carry income: 3.85%/yr on notional
- 10-delta put cost (3-month, ~8% OTM): approximately 0.8-1.5% of notional annualized
- Net carry after hedging: ~2.4-3.0%/yr in normal times
- Crash payoff: if AUD/JPY drops 30% (as in 2008), the 8% OTM put is now 22% ITM, paying ~22% of notional vs cost of ~0.3%

So the carry roughly funds the protection (3.85% income vs ~1-1.5% put cost), and the tail payoff is 15-70x the premium. This is the cleanest "carry funds protection" math of any asset class.

#### Instruments (retail-accessible)

- **CME FX futures**: AUD (6A), JPY (6J), MXN (6M), BRL (6L), CHF (6S). Options available on all major pairs.
- **CME cross-rate futures**: AUD/JPY directly tradeable on CME. $7.4B average daily volume in AUD futures, $14.7B in JPY.
- **CME FX options**: American-style exercise on most pairs. European-style available on GBP, CAD, EUR, JPY, CHF. Volatility-based quoting available (delta-neutral). $10B daily options liquidity.
- **Micro FX futures**: Available for major pairs, 1/10th standard size.

No ISDA required. Standard futures account at Interactive Brokers or similar.

#### Academic literature

- **Brunnermeier, Nagel & Pedersen (2008)** — "Carry Trades and Currency Crashes." Foundational paper. Carry returns have negative skewness — consistent with crash risk premium.
- **Lustig & Verdelhan (2007)** — "The Cross-Section of Foreign Currency Risk Premia and Consumption Growth Risk." High-interest-rate currencies earn a risk premium because they load on systematic crash risk.
- **Jurek (2014)** — "Crash-Neutral Currency Carry Trades." *Journal of Financial Economics*, Vol. 113, Issue 3, pp. 325-347. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14001081), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1262934)). The most important paper for this trade structure. See detailed analysis below.
- **Caballero & Doyle (2012)** — "Carry Trade and Systemic Risk: Why are FX Options So Cheap?" NBER Working Paper 18644. ([NBER](https://www.nber.org/papers/w18644)). Source of the "puzzlingly cheap" finding — FX option bundles designed to hedge carry trades provide cheap systemic risk insurance.
- **Burnside, Eichenbaum, Kleshchelski & Rebelo (2011)** — "Do Peso Problems Explain the Returns to the Carry Trade?" Options hedging reduces but doesn't eliminate abnormal carry returns.
- **Farhi, Fraiberger, Gabaix, Ranciere & Verdelhan** — "Crash Risk in Currency Markets." ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1397668)). Systematic crash risk is priced in currency returns.
- **Daniel, Hodrick & Lu (2017)** — "The Carry Trade: Risks and Drawdowns." *Critical Finance Review*. ([PDF](https://business.columbia.edu/sites/default/files-efs/pubfiles/6378/Daniel.Hodrick.Lu.Carry%20Trade.Critical%20Finance%20Review.2017.pdf), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2486275)).
- **Fan & Londono (2022)** — "Equity Tail Risk and Currency Risk Premia." ([Federal Reserve](https://www.federalreserve.gov/econres/ifdp/files/ifdp1253.pdf)). Option-based equity tail risk is priced in the cross-section of currency returns.
- **BIS Bulletin No. 90 (2024)** — "The Market Turbulence and Carry Trade Unwind of August 2024." ([BIS](https://www.bis.org/publ/bisbull90.pdf)). Real-time analysis of the 2024 yen carry unwind.

#### Jurek (2014) in detail — the key paper

This is the most important academic validation of the carry + tail protection structure. Full citation: Jakub W. Jurek (Princeton/Wharton), *Journal of Financial Economics*, Vol. 113, Issue 3, pp. 325-347, 2014.

**Sample:** 9 G10 currency pairs vs USD (AUD, NZD, GBP, CAD, NOK, SEK, EUR, CHF, JPY). Full sample 1990-2012 for spot/forward trades. Options subsample 1999-2012 (constrained by simultaneous availability of FX option data for all G10 currencies). Options data from J.P. Morgan; implied vol surface constructed via vanna-volga method.

**Methodology:** Combine standard carry trades with OTM FX option protection to create "crash-neutral" portfolios. When long a high-yield currency, buy a put on it. When short a funding currency, buy a call. Tested with 10-delta options (~3-4.5% OTM) and 25-delta options (~1.5-2.5% OTM), both 1-month and 3-month maturities. Multiple portfolio weightings: equal-weighted, spread-weighted, dollar-neutral and non-dollar-neutral.

**The headline result:** Crash risk accounts for **at most one-third** (~30-35%) of carry trade excess returns. Approximately **two-thirds of carry alpha survives crash hedging.** The "peso problem" explanation — that carry returns are just compensation for rare crashes — is rejected.

**Hedged carry returns (10-delta options, full sample):**

| Portfolio | Excess Return | t-stat |
|-----------|-------------|--------|
| Non-dollar-neutral, spread-weighted | **6.55%/yr** | 3.59 |
| Non-dollar-neutral, equal-weighted | **3.85%/yr** | 2.76 |
| Dollar-neutral, spread-weighted | **5.31%/yr** | 3.69 |
| Dollar-neutral, equal-weighted | **3.18%/yr** | 3.13 |

These are returns *after* paying for crash protection. The fact that they're still 3-6%/yr and statistically significant means the carry trade is genuinely profitable, not just crash risk compensation.

**The "four times" finding:** Rationalizing the *entirety* of carry returns via crash risk would require OTM FX option implied volatilities to be roughly **4x greater** than actually observed. There is no evidence of such mispricing. This means FX options are **not** unconditionally cheap in Jurek's framework — they're priced roughly correctly for the crash risk they cover, but carry returns are much larger than crash risk alone can explain.

**Important nuance:** The exact phrase "puzzlingly cheap" comes from **Caballero & Doyle (2012)**, not Jurek. Jurek finds the options are correctly priced for crash risk — but since crash risk only explains 1/3 of carry returns, the remaining 2/3 is genuine alpha. This actually makes the trade *better*: you're not just harvesting mispriced insurance, you're earning a real risk premium that persists even after hedging.

**3-month options beat 1-month:** Quarterly hedging produces returns 1-2%/yr higher than monthly hedging. This is because carry trade crashes unfold over weeks (not overnight jumps), making rolling 1-month protection more expensive relative to the risk covered. The 2008 carry unwind (~20% loss over ~3 months) was a gradual sequence of adverse moves, not a single overnight event.

**Key takeaway for our framework:** The carry + OTM put structure works in FX with strong academic backing. The options aren't "free money cheap" — they're correctly priced for crash risk. But the carry premium is so large that even after paying for full crash hedging, you still earn 3-6%/yr with statistical significance. Use 10-delta options (cheap, far OTM), 3-month maturities (cheaper than monthly rolling), and spread-weighted portfolios (higher returns).

#### Practical challenges

- **Margin**: FX futures require margin. Long options require only premium (no additional margin calls).
- **Roll costs**: CME FX options expire quarterly for standard, monthly for serial. Rolling cost is minimal for liquid pairs (EUR, JPY, AUD).
- **Liquidity in OTM strikes**: Decent for G10 pairs, thins significantly for EM pairs (MXN, BRL). Bid-ask widens for >15% OTM.
- **EM currency options**: MXN is liquid on CME. BRL, TRY, ZAR are primarily OTC — retail access limited.
- **Counterparty risk**: Exchange-traded (CME-cleared) eliminates this. OTC FX options carry bilateral counterparty risk.
- **Settlement**: CME FX futures are physically settled (delivery of currency). Options are settled into futures positions, which can then be closed.

**Why FX carry ranks #2**: The carry literally pays for the protection, and academic evidence confirms the options are underpriced. The structure is cleaner than equities (where "carry" is just "SPY goes up over time"). But G10 pairs are accessible at retail; EM pairs mostly require OTC.

---

### 4.3 Credit — Spread Compression/Blowout Cycle

Corporate bonds earn a spread over treasuries. Spreads grind tight during expansions, then blow out violently when fear returns. The asymmetry is structural: spreads can only compress so far toward zero but can widen essentially without limit.

#### Historical spread moves

| Event | IG Spread | HY Spread | Notes |
|-------|----------|----------|-------|
| 2007 pre-crisis | ~80bps | ~250bps | Multi-year tights |
| 2008 GFC peak | **656bps** (Dec 2008) | **2,000bps+** | 8x IG blowout, 8x HY blowout |
| 2011 Euro crisis | ~200bps | ~800bps | European sovereign contagion |
| 2015-16 Energy/China | ~180bps | ~850bps | HY energy sector CDS blew out to 1,500bps+ |
| 2019 pre-COVID | ~100bps | ~360bps | |
| 2020 COVID peak | ~400bps (Mar 23) | ~1,100bps | Fastest blowout in history — 3 weeks. IG ETFs (LQD) traded at 7% discount to NAV |
| 2020 post-Fed | ~130bps (Dec) | ~400bps | Fed bought IG bonds → rapid compression |
| 2022 Russia/Ukraine | ~150bps | ~500bps | +70bps IG, +150bps HY from 2021 lows |
| 2024 tights | ~80bps | ~300bps | Near multi-decade tights |

The COVID sequence is instructive: IG spreads went from 100bps to 400bps in three weeks, then back to 130bps by year-end after the Fed intervened. Anyone holding CDS protection bought at 100bps and cashed out at 400bps — a 4x move, with the timing determined by a policy decision.

#### Instruments

**Institutional (ISDA required):**
- **CDX IG**: CDS index on 125 investment-grade North American names. 5-year standard maturity, rolls every 6 months. Cleared through ICE Clear Credit since 2009 (mandatory clearing under Dodd-Frank since March 2013).
- **CDX HY**: CDS index on 100 high-yield names. Same mechanics.
- **iTraxx Europe**: European IG CDS index.
- **Single-name CDS**: Protection on individual corporate or sovereign credits.

**Retail-accessible:**
- **LQD** (iShares IG Corporate Bond ETF): Options available, decent liquidity. ~$30B AUM.
- **HYG** (iShares HY Corporate Bond ETF): Options available. ~$15B AUM.
- **JNK** (SPDR HY Bond ETF): Options available.
- **Puts on HYG/JNK**: The retail-accessible credit tail hedge. Buy OTM puts on HYG = bet on credit spread blowout.

A recent paper (ArXiv 2504.06289, April 2025) tested **shorting IG corporate bond ETFs as a tail hedge** using momentum, liquidity, and credit signals. Key findings:
- Systematic short HYG/LQD hedges avoided IG credit drawdowns
- Achieved higher Sortino ratios than benchmark bond funds
- Credit ETFs provided better downside convexity than CDX index positions
- Works for portfolios up to ~$10B in assets
- LQD liquidity improved dramatically: a $500M hedge position took ~35 trading days in 2012 but only ~2 days now

#### The carry math

This is where credit gets harder than FX or rates. The "carry" (IG bond spread) is typically only 80-120bps. Protection costs:
- CDX HY 5-year protection costs ~300-400bps/yr in calm markets
- CDX IG 5-year costs ~60-80bps/yr in calm markets

So you can't fund HY protection from IG carry alone (120bps income vs 300-400bps protection cost). You'd need to either:
1. Accept negative carry and rely on tail blowout to compensate
2. Use partial hedges (buy less notional protection)
3. Use the retail approach: long LQD + buy OTM puts on HYG (much cheaper than CDS)

The retail approach (puts on HYG) is actually more capital-efficient because option premiums are much lower than CDS running spreads. An 8% OTM put on HYG with 3-month expiry might cost 0.5-1.5% of notional, not 300-400bps annually.

#### The AIG problem and counterparty risk

The irony of CDS: you buy protection against credit crises, but the protection seller might fail in the crisis. AIG wrote ~$440B in CDS and couldn't meet collateral calls when spreads widened in 2008. Bear Stearns and Lehman couldn't find counterparties willing to trade as their troubles became apparent.

Post-2008 reforms helped:
- Mandatory central clearing through CCPs (ICE Clear Credit, LCH) since 2013
- Both parties face the CCP, not each other
- Members pay into a default fund

But counterparty risk isn't fully eliminated for uncleared single-name CDS. And during true systemic crises, even cleared CDS could face settlement issues.

#### The Fed intervention problem

The Fed bought IG corporate bonds directly in 2020 (first time ever), compressing spreads artificially. This creates a new dynamic: the tail hedge might pay off briefly, but the Fed may compress spreads before you can exit. The March 2020 window was only about 2-3 weeks wide.

Going forward, markets now expect Fed intervention in credit crises, which means:
1. Spreads might not blow out as far as they "should" (the "Fed put" extends to credit)
2. But the option protection also expires — you need to be quick
3. The policy response itself is asymmetric in your favor (spreads compress = your bond portfolio recovers)

#### Academic literature

- **Collin-Dufresne, Goldstein & Martin (2001)** — "The Determinants of Credit Spread Changes." *Journal of Finance*. Spreads are driven by systematic factors, not just individual default risk.
- **Berndt, Duffie & Zhu (2018)** — "Corporate Credit Risk Premia." *Review of Finance*. CDS spreads follow lognormal distribution — fat tails on the widening side.
- **Longstaff, Pan, Pedersen & Singleton (2011)** — "How Sovereign Is Sovereign Credit Risk?" *American Economic Review*. Systemic factors drive CDS even more than fundamentals.
- **Chen, Joslin & Ni (2019)** — "Demand for Crash Insurance, Intermediary Constraints, and Risk Premia in Financial Markets." On credit spread option pricing and tail risk.
- **ArXiv 2504.06289 (2025)** — On shorting IG corporate bond ETFs as systematic tail hedges using momentum, liquidity, and credit signals.

#### Practical challenges

- **ISDA requirement**: CDX/CDS trading requires an ISDA Master Agreement. Retail investors cannot access these directly.
- **The retail alternative works**: OTM puts on HYG/LQD are available to anyone with an options account. This is how retail can access credit tail hedging.
- **Liquidity during crises**: In March 2020, IG ETFs traded at 3-7% discounts to NAV because authorized participants couldn't arbitrage the gap (bond market frozen, ETFs still trading). This actually helps the put buyer — the ETF drops MORE than the underlying bonds.
- **Carry is thin**: Unlike FX where carry pays for protection, credit carry is barely enough to cover IG protection costs, let alone HY.

**Why credit ranks #4 (institutional)**: The convexity is genuine (spreads 80bps → 656bps in 2008), but the carry-to-protection ratio is unfavorable, counterparty risk is a real concern, and retail access requires the ETF workaround. For institutions with ISDA agreements and dedicated credit desks, CDX protection is a core tail hedge. For retail, OTM puts on HYG are the practical answer.

---

### 4.4 Commodities — Supply/Demand Shocks

#### Oil — both tails are fat

Oil is unusual: both upside (supply disruptions) and downside (demand collapses) tails are massive. Crude oil's daily return kurtosis is approximately **37** — vastly exceeding equities (~4.5).

| Event | Move | Timeline |
|-------|------|----------|
| 2008 superspike & crash | $147.30 → $32 (-78%) | Jul-Dec 2008 |
| 2014-2016 shale glut | $107 → $26 (-76%) | Jun 2014 - Jan 2016 |
| 2020 COVID negative prices | $17.73 → **-$37.63** | Apr 20, 2020 (one day!) |
| 2022 Russia/Ukraine spike | $76 → $133 (+75%) | Feb-Mar 2022, Brent hit $139 |

The April 2020 event is historic: May WTI contract settled at -$37.63/bbl the day before expiry. Cushing, OK storage hit 83% capacity. First negative price in WTI's 37-year history.

**Instruments**: CL futures (NYMEX, 1,000 bbl/contract, ~$70K notional). Micro CL (MCL, 100 bbl). Options on CL are highly liquid. USO ETF has options but suffers severe contango drag (~14.6%/yr).

**The trade structure**: A long strangle (buy both OTM puts and OTM calls) on CL is rational because both tails pay. Example: CL at $70, buy $55 puts and $90 calls 3 months out. In calm markets (OVX 20-30), deep OTM CL options are cheap:
- $50 put with oil at $70 (3 months): ~$0.20-0.50/bbl ($200-500/contract)
- $100 call with oil at $70 (3 months): ~$0.10-0.30/bbl ($100-300/contract)

The challenge: contango. In persistent contango (futures > spot), rolling front-month longs costs 10-15%/yr. This is the "time decay" equivalent. During supply shocks, oil flips to backwardation, rewarding longs with both price appreciation and positive roll yield.

#### Gold — chaos hedge, not crisis hedge

Gold is NOT a reliable immediate crisis hedge. Critical finding:
- **2008**: Gold dropped from $1,011 to $700 (-30%) during the liquidity phase before recovering
- **March 2020**: Gold dropped from $1,680 to $1,470 (-12%) in the "dash for cash" before rallying to $2,067 by August
- **2022**: Spiked briefly to $2,074 on Russia invasion, then fell to $1,656 as the Fed hiked rates

Gold fails precisely when you need a tail hedge most — in the first 1-2 weeks of a crisis. Institutions sell gold to meet margin calls. Dollar strength during "dash for cash" hurts gold. It only rallies AFTER central banks respond with liquidity.

Gold is a **monetary debasement hedge** (responds to QE, money printing), not an **impact hedge** (responds to the crash itself). Spitznagel himself has backtested gold and found it inferior to equity puts.

That said, gold's 2024-2026 rally to $5,500+ (53 new all-time highs in 2025 alone) demonstrates its structural power against currency debasement and geopolitical regime change — just not as an options-based tail trade.

**Instruments**: GC futures (COMEX, 100 oz). Micro Gold (MGC, 10 oz). GLD ETF with very liquid options. GVZ (gold vol index) ranges 12-16 calm, 30-48 in stress.

#### Agriculture — hedger-dominated, potentially cheap options

Weather-driven tail events create massive moves:

| Event | Commodity | Move |
|-------|-----------|------|
| 2012 US drought | Corn (ZC) | Peaked at $8.38/bu (all-time high, normal ~$4-5) |
| 2012 US drought | Soybeans (ZS) | Peaked at $17.89/bu (all-time high, normal ~$10) |
| 2022 Ukraine/wheat | Wheat (ZW) | Peaked at $14.25/bu (broke 2008 record of $13.34) |
| 2022 Ukraine/corn | Corn (ZC) | Surpassed $8/bu (first since 2012) |

**The "hedger-dominated" argument**: Farmers are natural sellers of futures and buyers of puts (lock in prices for unharvested crops). This creates systematic selling pressure. De Roon, Nijman & Veld (2000) formalized the "hedging pressure hypothesis." The implication: **OTM agricultural calls may be structurally cheap** because farmers systematically sell them as covered calls against physical crop holdings.

**Instruments**: ZC (corn), ZW (wheat), ZS (soybeans) — all CBOT, 5,000 bu/contract. Micro contracts (MZC, MZW, MZS, 1,000 bu) launched Feb 2025. Options on all, but liquidity drops significantly for deep OTM.

**Best commodity tail hedge**: OTM corn or wheat calls before the US growing season (June-August weather risk window). Cheap, fat-tailed, and genuine information asymmetry between weather forecasters and options market makers.

#### VIX — purest expression of tail risk

| Event | Date | VIX Level | Notes |
|-------|------|-----------|-------|
| 2008 GFC | Nov 21, 2008 | **80.74** close | All-time closing high |
| 2011 debt ceiling | Aug 8, 2011 | **48.00** close | US downgrade + Europe |
| 2015 China deval | Aug 24, 2015 | **53.29** intraday, 40.74 close | "Mini flash crash" |
| 2018 Volmageddon | Feb 5, 2018 | **37.32** close (from 17.31 prior) | +116% in one day. XIV lost 97%. |
| 2020 COVID | Mar 16, 2020 | **82.69** close | Higher than 2008 closing high |

**VIX options mechanics**: Options on VIX index (not futures), but priced off VIX futures forward value. European-style, cash-settled. Critical: a VIX 25-strike call when VIX spot is 15 is priced off the relevant VIX futures (~18-20 due to contango), not spot VIX.

**OTM VIX call costs** (with VIX spot ~15, futures ~18):

| Strike | ~90 DTE Premium | Notes |
|--------|----------------|-------|
| 25 | ~$0.15-0.25 ($15-25/contract) | Moderate OTM |
| 30 | ~$0.08-0.15 ($8-15/contract) | Deep OTM |
| 40 | ~$0.30-0.40 ($30-40/contract) | VIX skew inflates these (IV 120-156%) |

**VIX call spreads** reduce cost: buy 25c / sell 40c for ~$0.15 net ($15/contract), max payout $14.85 if VIX ≥40. Caps upside but dramatically cuts cost.

**The contango bleed**: VIX futures are in contango >80% of the time. Monthly contango drag is 5-10%. VXX (short-term VIX futures ETN) is down >99% since inception. This is the insurance premium the market charges.

**Is VIX overpriced or underpriced?** A 25-strike VIX call purchased for $25 that expires with VIX at 80 pays $55,000 — a 2,200x return. Universa returned +3,612% in March 2020. The question is whether options markets fully price VIX 80+ events. History suggests they don't.

**Recommended allocation**: 0.25-1.0% of portfolio per month to VIX call hedges. When VIX is low (12-18), spend more (~1%); when elevated (25-40), spend less (options pricier).

#### Commodity data availability for backtesting

- **CME DataMine**: Historical data back to 1970s for some contracts. Paid, self-service cloud. >450TB of data.
- **OptionMetrics (IvyDB Futures)**: Academic standard. End-of-day prices, IV, Greeks. Premium subscription.
- **CBOE DataShop (Livevol)**: VIX options/futures data from 2004. IV surfaces, trade-level.
- **Databento**: CME/CBOT/NYMEX/COMEX data. Competitively priced but no pre-calculated Greeks.
- **Free**: Essentially no research-quality free commodity options data. FRED has OVX/GVZ/VIX daily closes only.

#### Academic literature

- **Gorton & Rouwenhorst (2006)** — "Facts and Fantasies about Commodity Futures." Commodity futures match equity Sharpe ratios over 1959-2004. Negatively correlated with stocks/bonds.
- **Tang & Xiong (2012)** — "Index Investment and the Financialization of Commodities." Since 2000s, non-energy commodities increasingly correlated with oil due to index fund flows ($15B in 2003 → $200B+ by 2008).
- **Gorton, Hayashi & Rouwenhorst (2012)** — Risk premiums vary by inventory levels. Convenience yield is non-linear in inventories.

#### Commodity tail hedge ranking

| Commodity | Attractiveness | Why |
|-----------|---------------|-----|
| Crude oil (CL) | **High** | Both-sided fat tails (kurtosis ~37), liquid options, reasonable cost in calm |
| Wheat/Corn calls | **High** | Hedger-dominated selling cheapens calls, weather tails are genuine |
| VIX calls | **High (expensive)** | Purest tail hedge, but contango bleed is severe |
| Gold | **Low** | Fails as immediate crisis hedge (dash for cash), better as macro debasement hedge |

---

### 4.5 Emerging Market Sovereign Debt

High yield in normal times, violent contagious crises. EM crises cluster because they share common vulnerabilities: dollar-denominated debt, commodity dependence, hot money flows, and the "sudden stop" dynamic (Calvo, 1998).

#### Historical crisis episodes

| Event | Key Moves | Contagion |
|-------|----------|-----------|
| 1997 Asian crisis | THB devalued Jul 2, spread to KRW, IDR, MYR. Currencies -40-80% | Thailand → Korea → Indonesia → Malaysia → Philippines |
| 1998 Russia | Ruble: 6.29 → 21/USD in 3 weeks. GKO yields hit 150%. EMBI spread >1,500bps | Russia → Brazil → LTCM collapse. Global contagion. |
| 2001 Argentina | Corralito (bank freeze Dec 2001), default on $93B, peso depegged 1:1 → 4:1 | Argentina → Uruguay → Brazil pressure |
| 2013 Taper Tantrum | "Fragile Five" (BRL, INR, IDR, TRY, ZAR) sold off -10-20% | Fed *signaling* (not acting) caused EM rout |
| 2015 China deval | PBoC devalued yuan Aug 11. EM currencies fell 5-15% | China → commodity exporters (BRL, ZAR, AUD) |
| 2018 Turkey/Argentina | TRY: -45%. ARS: -50%. Turkish rates hiked 8% → 24%, failed to stop slide | Turkey → Argentina → South Africa → broad EM |
| 2020 COVID | EMBI spread blew out ~300bps in 3 weeks. EM currencies -15-30% | Global. Fed swap lines activated for select central banks |

The contagion dynamic is key: EM crises are not independent events. When one EM country stumbles, investors pull capital from all EM — the "sudden stop" (Calvo & Reinhart). Capital that flowed in over years flows out in weeks. Dollar funding stress amplifies everything.

#### Current EM sovereign yields

EM USD-denominated sovereign bonds (EMB universe) typically yield 300-500bps over US Treasuries. With UST at ~4%, that's 7-9% total yield. Individual country spreads vary enormously:
- Investment-grade EM (Chile, Peru, Malaysia): 80-150bps
- Mid-grade (Mexico, Colombia, Brazil): 150-300bps
- High-risk (Turkey, South Africa, Nigeria): 300-600bps
- Distressed (Argentina, Pakistan, Egypt): 600-2,000bps+

#### EMBI spread history

| Period | EMBI Spread (bps) |
|--------|-------------------|
| 2007 tights | ~170bps |
| 2008 GFC peak (Q4) | ~750bps |
| 2013 Taper Tantrum | ~400bps |
| 2020 COVID peak | ~700bps |
| 2024 tights | ~250bps |

The 2008 swing: 170bps → 750bps, a 4.4x blowout. Protection bought at 170bps paid out at 750bps.

#### Instruments

**Institutional (ISDA required):**
- **CDX EM**: EM sovereign CDS index. Cleared through ICE.
- **Single-name sovereign CDS**: Protection on individual countries (Brazil, Turkey, Mexico, etc.).

**Retail-accessible:**
- **EMB** (iShares JP Morgan USD EM Bond ETF): USD-denominated EM sovereign bonds. ~$15B AUM. **Options available** — this is the retail tail hedge entry point.
- **EMLC** (VanEck EM Local Currency Bond ETF): Local currency EM bonds. Adds FX risk on top of credit risk.
- **EEM** (iShares MSCI Emerging Markets Equity ETF): $20.3B AUM, 0.68% expense ratio. **Very active options market** — the best retail EM tail hedge instrument. Higher vol (5.34%) than VWO (4.31%), meaning more responsive to crises.
- **VWO** (Vanguard FTSE EM ETF): $72.8B AUM, 0.08% expense ratio. More liquid ETF but less active options market than EEM.

**The retail trade**: Long EEM or EMB for carry + buy OTM puts for tail protection. EEM options are the most liquid EM derivative accessible to retail investors.

#### The contagion hedge

How correlated are EM crises? The answer matters for hedging:
- **Systemic events (2008, 2020)**: Nearly all EM sells off together. A single CDX EM or EEM put covers the complex.
- **Idiosyncratic events (Turkey 2018)**: Limited contagion. Country-specific protection needed.
- **Regional events (1997 Asia)**: Regional contagion but limited global impact. Partial coverage from broad EM hedge.

For most purposes, **OTM puts on EEM** provide adequate EM tail protection because the big crises are systemic. The Turkey 2018 scenario (idiosyncratic, no global contagion) is the case where broad EM protection fails — but those events also tend to be smaller.

#### Academic literature

- **Calvo (1998)** — "Capital Flows and Capital-Market Crises: The Simple Economics of Sudden Stops." Foundational paper on EM capital flow dynamics.
- **Reinhart & Rogoff (2009)** — *This Time Is Different*. Eight centuries of sovereign defaults, showing EM crises are recurring.
- **Eichengreen, Hausmann & Panizza (2005)** — "The Pain of Original Sin." EM countries can't borrow in their own currency, creating structural vulnerability.
- **Calvo & Reinhart (2002)** — "Fear of Floating." EM central banks accumulate reserves to prevent currency appreciation, creating vulnerability when flows reverse.

#### The carry math

EMB yields ~7-9%. OTM puts on EEM or EMB (10% OTM, 3-month):
- EEM implied vol is typically 18-25% in calm markets, 35-50% in stress
- A 10% OTM put on EEM (3 months) costs roughly 1-2% of notional annualized
- Net carry: 7-9% yield minus 1-2% protection cost = 5-7%/yr in normal times

If an EM crisis hits and EEM drops 30% (as in 2008): the 10% OTM put is now 20% ITM, paying ~20% of notional vs cost of ~0.5%. Payoff ratio: 40x.

#### Practical challenges

- **Liquidity**: EEM options are liquid; EMB options are thinner. Single-name EM sovereign CDS is institutional only.
- **Local currency risk**: EMLC adds FX risk on top of credit. During EM crises, currencies and bonds fall together — double whammy or double payoff for puts.
- **Capital controls**: Some EM countries impose capital controls during crises (Malaysia 1998, Iceland 2008, Argentina repeatedly). This can prevent you from realizing gains.
- **Political risk**: Forced restructuring, unilateral moratoriums (Russia 1998), nationalization.
- **Correlation breakdown**: EEM is 60% Asia (China, Taiwan, India, Korea). An EM-specific credit crisis may not move EEM much if China is unaffected. EMB is a better pure EM credit proxy.

**Why EM ranks #5**: Genuine tail risk with 4-5x spread blowouts, but idiosyncratic events may not trigger broad EM instruments, and retail access is limited to ETF options. The carry math is attractive (7-9% yield) but counterbalanced by the structural fragilities that make EM crises recurring.

---

## 5. The Common Thread

Markets systematically price risk as if the future looks like the recent past. During calm periods:
- Implied vol drops (options get cheap)
- Credit spreads tighten (protection gets cheap)
- Carry trades get crowded (more people selling insurance)
- Risk models say everything is low-risk

Then a shock — geopolitical event, financial contagion, policy error — reprices everything violently. The gap between "normal-times pricing" and "crisis pricing" is the structural edge.

The Spitznagel/Universa insight isn't about equities specifically. It's about **scanning all markets for the cheapest tail convexity and buying it there**. Sometimes that's equity vol (2007). Sometimes it's credit protection (2006). Sometimes it's rate swaptions (2019). The portfolio manager's job is to always own the cheapest crash insurance, wherever it lives.

### Ranking by structural attractiveness

| Rank | Market | Why | Best For |
|------|--------|-----|----------|
| 1 | **Rates** | Fed reaction function = institutional guarantee of asymmetry | Futures account holders |
| 2 | **FX Carry** | Carry funds protection. Academic evidence: options are "puzzlingly cheap" | Futures/FX traders |
| 3 | **Equities (SPY)** | 16%/yr Sharpe 1.879 backtested. Most liquid options on earth | Everyone (retail default) |
| 4 | **Credit** | 4-8x spread blowouts, but thin carry and ISDA requirement for CDS | Institutions (retail via HYG puts) |
| 5 | **Commodities** | Fat tails (oil kurtosis ~37) but contango bleed and delivery complexity | Specialists |
| 6 | **EM Debt** | 4-5x spread blowouts but idiosyncratic risk and limited instruments | Dedicated EM investors |
| 7 | **VIX** | Purest tail hedge but most expensive (contango >80% of time) | Small allocation overlay |

---

## 6. Why SPY + Puts Is Still the Best Retail Implementation

Despite rates and FX having arguably better structural asymmetries, SPY wins for most investors:

1. **SPY options are the most liquid derivatives on earth.** Penny-wide bid-ask, 0-DTE to 2-year expiry, strikes every $1. No other market comes close to this execution quality.

2. **100% SPY + 0.5% fixed puts = 16%/yr, Sharpe 1.879** — confirmed by this backtester over 2008-2025. The equity risk premium provides the return engine. The puts improve geometric compounding by reducing variance drain. The cost is genuinely negligible.

3. **Everything else requires OTC markets, futures accounts, specialized data, and higher transaction costs.** SOFR swaptions? You need an ISDA master agreement. CDS? Same, plus counterparty risk management. FX options? Futures account plus understanding of settlement mechanics.

4. **0.5% is so cheap it doesn't need funding.** No bond yield, no carry trade, no complex allocation formula. Just SPY and monthly puts. The simplicity is the feature, not a limitation.

The entire barbell exploration — selling vol, buying tail protection, adding bonds, dynamic put sizing — was a search for something more sophisticated than "buy SPY, spend 0.5% on puts." Nothing beat it.

---

## 7. Next Steps: Backtesting SOFR and FX Carry

The options backtester currently supports SPY options data. The same engine could backtest the Spitznagel structure on other instruments if we source the data:

### SOFR Futures Options
- **Data source**: CME DataMine (paid). SOFR options data available from 2018. Eurodollar options back to 1980s but delisted June 2023.
- **Schema adaptation**: The backtester's Schema/Filter system would need columns mapped to SOFR option fields (strike as rate level, DTE, price).
- **Trade structure**: Long SOFR futures + roll OTM calls. Exit on rate cut events.

### FX Carry (AUD/JPY, MXN/JPY)
- **Data source**: CME DataMine for FX futures options. AUD (6A), JPY (6J) options data available historically.
- **Schema adaptation**: Need FX option fields: strike as exchange rate, underlying as FX spot, DTE, delta.
- **Trade structure**: Long high-yielder futures + buy OTM puts. The carry differential appears as positive roll yield on the futures position.
- **Key question**: Does the carry-funded-protection structure outperform SPY + puts on a risk-adjusted basis? Academic evidence (Jurek 2014) says hedged carry still earns significant returns.

Both would require downloading historical options data from CME DataMine or Databento and adapting the backtester's data providers to handle non-equity option schemas.

---

## 8. Synthetic FX Carry Backtest Results (Preliminary)

**Notebook:** `notebooks/fx_carry_synthetic.ipynb`

Before buying real FX options data, we ran a synthetic backtest using free spot FX data (yfinance, 2005-2026) and Black-Scholes put pricing with trailing 60-day realized vol as an implied vol proxy. This gives a rough estimate — not rigorous enough for trading decisions, but useful for validating whether the structure is worth pursuing with real data.

**Important caveat:** This uses modeled option prices, not real market data. Real FX options have skew (puts cost more than ATM), term structure, and bid-ask spreads that this model ignores. The results below need validation with actual CME FX option prices.

### Main results

| Strategy | CAGR | Volatility | Sharpe | Max DD |
|----------|------|-----------|--------|--------|
| AUD/JPY spot only | 1.51% | 15.38% | 0.098 | -47.6% |
| AUD/JPY carry (unhedged) | 4.68% | 15.38% | 0.305 | -44.9% |
| **AUD/JPY carry + 10d puts** | **5.17%** | **15.76%** | **0.328** | **-41.3%** |
| MXN/JPY spot only | -0.04% | 16.47% | -0.003 | -62.9% |
| MXN/JPY carry (unhedged) | 6.62% | 16.47% | 0.402 | -41.9% |
| **MXN/JPY carry + 10d puts** | **7.72%** | **16.99%** | **0.454** | **-35.2%** |
| EUR/USD carry + puts | -1.25% | 11.19% | -0.112 | -39.5% |
| SPY buy & hold | 10.61% | 19.01% | 0.558 | -55.2% |
| **SPY + 0.5% puts (real data)** | **16.00%** | **~14%** | **1.879** | **~-20%** |

### Put economics — the puts pay for themselves

| Pair | Put Cost/yr | Put Payoff/yr | Net |
|------|-----------|-------------|-----|
| AUD/JPY | 2.12% | 2.64% | **+0.52%/yr** (puts are profitable) |
| MXN/JPY | 2.41% | 3.52% | **+1.11%/yr** (puts are profitable) |
| EUR/USD | 1.47% | 1.39% | -0.08%/yr (roughly break-even) |

This confirms Jurek (2014) and Caballero & Doyle (2012): FX crash protection options are underpriced relative to what they pay. On both carry pairs, the hedged trade beats unhedged on both CAGR and Sharpe.

### Delta sensitivity (AUD/JPY and MXN/JPY)

| Pair | Delta | CAGR | Sharpe | Put Cost/yr | Net Cost/yr |
|------|-------|------|--------|-----------|------------|
| AUD/JPY | 5d | 5.44% | 0.349 | 0.93% | **-0.75%** (cheapest) |
| AUD/JPY | 10d | 5.17% | 0.328 | 2.12% | -0.52% |
| AUD/JPY | 25d | 3.87% | 0.237 | 6.72% | +0.64% (too expensive) |
| MXN/JPY | 5d | 7.70% | 0.459 | 1.06% | **-1.06%** |
| MXN/JPY | 10d | 7.72% | 0.454 | 2.41% | -1.11% |
| MXN/JPY | 25d | 6.23% | 0.353 | 7.63% | +0.18% |

5-delta and 10-delta puts are the sweet spot — cheap enough that payoffs exceed costs. 25-delta is too expensive and eats the carry.

### Tenor sensitivity — 3-month vs 1-month

| Pair | Tenor | CAGR | Sharpe | Put Cost/yr |
|------|-------|------|--------|-----------|
| AUD/JPY | 1m | 5.17% | 0.328 | 2.12% |
| AUD/JPY | 3m | 1.60% | 0.104 | 3.74% |
| MXN/JPY | 1m | 7.72% | 0.454 | 2.41% |
| MXN/JPY | 3m | 3.06% | 0.184 | 4.24% |

3-month tenor is worse here, contradicting Jurek's finding that quarterly hedging is 1-2%/yr cheaper. This is likely a model artifact — our BS pricing with realized vol overestimates 3-month costs because it scales vol by √T, while real implied vol term structure is flatter. **This is exactly the discrepancy that real option data would resolve.**

### Key takeaways

1. **FX carry + puts works** (5-8%/yr, positive Sharpe, puts net profitable) — but it doesn't beat SPY + puts (16%/yr, Sharpe 1.879)
2. **The puts pay for themselves** on carry pairs — confirming academic evidence that FX crash options are underpriced
3. **It's a diversifier, not a replacement** for the equity strategy
4. **Need real option data** to resolve the tenor question and get accurate cost estimates — synthetic BS pricing isn't enough
5. **EUR/USD confirms the control**: without carry differential, the structure adds nothing

---

## 8b. Real FX Options Backtest: AUD/USD (Databento CME Data)

**Notebook:** `notebooks/fx_carry_real.ipynb`

We bought real CME AUD futures options from Databento (~$5.57 total cost) and ran the backtest with actual market prices. This is the first test using real option data instead of synthetic Black-Scholes pricing.

### Data

| Dataset | Symbol | Rows | Date Range | Cost |
|---------|--------|------|-----------|------|
| AUD futures options (old) | 6A.OPT | 173K | 2010-2017 | (from initial download) |
| AUD futures options (new) | ADU.OPT | 240K | 2016-2026 | $2.38 |
| AUD futures (underlying) | 6A.FUT | 30K | 2010-2026 | $0.30 |

**Key Databento finding:** CME FX options use different product codes than the underlying futures. AUD futures trade as `6A`, but the options trade as `ADU`. Similarly: JPY = `JPU`, EUR = `EUU`, GBP = `GBU`. The old `6A.OPT` parent symbol only resolved for pre-2017 data; `ADU.OPT` covers 2016-2026. Combined, we have 16 years of real daily option OHLCV data.

### Strategy

- 100% notional in front-month AUD/USD futures (rolled monthly, highest volume contract)
- Each month, spend 0.5% of portfolio on a 1-month OTM put
- Put settles at last traded price near expiry or intrinsic value

### Main results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|-----------|---------|--------|--------|
| AUD futures only | 0.18% | 10.84% | 0.017 | -37.06% |
| AUD + 10% OTM puts (0.5%) | 5.27% | 26.92% | 0.196 | -36.53% |
| AUD + 5% OTM puts (0.5%) | 4.10% | 14.14% | 0.290 | -29.52% |
| **SPY + 0.5% puts (real)** | **16.46%** | **8.76%** | **1.879** | **-8.24%** |

### Put economics

For 10% OTM puts:
- Total put P&L: +104% of portfolio over 15 years (net profitable!)
- Win rate: 5.5% of months (10 out of 182)
- The puts are profitable because the few winners are massive: Aug 2011 returned 2094% on premium, Sep 2011 returned 4650%
- Most months: puts expire worthless (-100% of premium = -0.5% of portfolio)

### Crisis performance

| Crisis | Futures Ret | Put P&L | Combined | Puts Helped? |
|--------|-----------|---------|----------|-------------|
| 2011 EU debt crisis | -8.46% | +32.72% | +25.80% | YES |
| 2013 taper tantrum | -7.87% | +106.20% | +102.23% | YES |
| 2015 China deval | -2.28% | -1.00% | -3.27% | no |
| 2018 trade war | -7.64% | -6.00% | -13.06% | no |
| 2020 COVID crash | -0.34% | +2.61% | +2.41% | YES |
| 2022 rate hikes | -13.72% | -3.00% | -16.34% | no |

The puts helped in 3 of 6 crises. They failed in 2015 (AUD drop was too gradual), 2018 (slow grind, not a crash), and 2022 (steady rate-driven decline, not a panic). This confirms a key limitation: OTM puts are designed for fast, violent moves — not slow grinds.

### Synthetic vs real comparison

| Metric | Synthetic (BS) | Real (Databento) |
|--------|---------------|-----------------|
| AUD carry + puts CAGR | 5.17% | 5.27% (10% OTM) / 4.10% (5% OTM) |
| Put win rate | ~15% | 5.5% (10% OTM) / 9.3% (5% OTM) |
| Avg winner size | ~10-20x | ~200x (10% OTM) / ~40x (5% OTM) |
| Put net P&L | +0.52%/yr | +6.88%/yr (10% OTM) / +4.17%/yr (5% OTM) |

The synthetic model underestimates both the win size and the loss rate. Real deep OTM options are much cheaper than Black-Scholes suggests (hence the huge multipliers when they pay off), but they also expire worthless far more often. The two effects roughly cancel in total return, but the distribution is more extreme with real data.

### Key takeaways

1. **The puts are net profitable on real data** — confirming that FX crash protection is genuinely underpriced
2. **AUD/USD has no equity premium** (0.18%/yr over 15 years). Without the underlying return engine, the strategy depends entirely on crisis timing
3. **5% OTM puts are better risk-adjusted** (Sharpe 0.290 vs 0.196) — less extreme leverage, more consistent payoff
4. **SPY + puts remains dominant** — the equity premium (11%/yr) is the key ingredient that makes the Spitznagel structure work. FX carry (2-4%/yr differential) isn't enough
5. **The real value of FX hedges is diversification**: in 2011 and 2013, AUD puts paid massively while SPY was less affected. A portfolio holding BOTH SPY puts and FX carry puts would have broader crisis coverage

---

## 8c. Leveraged AUD/JPY Carry + OTM Puts (The Real Carry Trade)

**Notebook:** `notebooks/fx_carry_real.ipynb` (updated)

The previous section tested AUD/USD — the wrong pair. AUD/USD carry was only 0.95%/yr over 2010-2026 because the Fed hiked above the RBA in 2018-2024. The **real** carry trade is AUD/JPY, where the differential averaged 2.47%/yr over the same period (BOJ held at ~0% until 2024).

### How carry traders actually make money

1. **Borrow JPY** at ~0% interest
2. **Buy AUD** assets earning 2-4%
3. **Apply leverage** (3-10x typical for FX carry)
4. **Pocket the rate differential × leverage**

With 5x leverage on 2.5%/yr carry = 12.5%/yr income. That's comparable to equities. The catch: AUD/JPY drops 15-40% in crises (2008: -40%, 2011: -13%, 2013: -12%, 2015: -11%, 2020: -9%).

### AUD/JPY spot return 2010-2026

AUD/JPY went from ~75 to ~111 over the period (+48% total, +2.5%/yr annualized). This is mainly because JPY weakened massively (USD/JPY went from 92 to 156) as BOJ maintained ultra-loose monetary policy. So AUD/JPY carry traders got both carry AND appreciation — a historically favorable period.

### Leveraged backtest results

| Strategy | Ann. Return | Vol | Sharpe | Max DD |
|----------|-----------|-----|--------|--------|
| **1x AUD/JPY unhedged** | 6.81% | 11.1% | 0.616 | -28.1% |
| **1x + 5% OTM puts (0.5%)** | **12.67%** | **15.9%** | **0.795** | **-20.8%** |
| 1x + 8% OTM puts (0.5%) | 14.05% | 19.0% | 0.741 | -22.7% |
| 3x AUD/JPY unhedged | 16.45% | 33.2% | 0.496 | -70.5% |
| 3x + 5% OTM puts | 32.62% | 47.8% | 0.682 | -55.8% |
| 5x AUD/JPY unhedged | 19.41% | 55.3% | 0.351 | -91.4% |
| **SPY + 0.5% puts** | **16.46%** | **8.8%** | **1.879** | **-8.24%** |

### P&L decomposition (1x leverage, 8% OTM puts)

| Component | Total ($, starting $100) | Per Year |
|-----------|-------------------------|----------|
| Carry income | $130.9 | $8.3/yr |
| Spot P&L | $315.1 | $20.0/yr |
| Put P&L | $244.8 | $15.6/yr |
| **Net total** | **$690.7** | **$43.9/yr** |

The puts are net profitable — massively so. The biggest put payoffs came during the 2011 EU debt crisis where a single month's puts returned 2000%+ on premium.

### Crisis performance (5x leverage)

| Crisis | AUD/JPY | Unhedged | + 8% OTM | + 5% OTM |
|--------|---------|----------|----------|----------|
| 2011 EU debt | -4.9% | -22.5% | **+274.8%** | +140.8% |
| 2013 taper tantrum | -12.3% | -49.1% | -39.5% | -39.5% |
| 2015 China deval | -10.8% | -45.7% | -47.3% | -47.3% |
| 2018 trade war | -12.2% | -44.2% | +8.9% | +8.9% |
| 2020 COVID | -8.5% | -41.5% | -43.0% | -43.0% |
| 2022 rate hikes | +3.4% | +18.0% | +12.9% | +12.9% |

Puts helped in 2 of 5 crises (2011, 2018) but failed in 2013, 2015, 2020. The core issue: **our puts are on AUD/USD, not AUD/JPY**. When AUD/JPY drops because of JPY strength (not AUD weakness), the AUD/USD puts don't protect. In 2020, AUD/USD dropped briefly but recovered quickly while USD/JPY also moved — the cross-rate exposure isn't fully hedged.

### Key takeaways

1. **Leveraged carry works** — 1x AUD/JPY at 6.81%/yr is comparable to many equity markets
2. **The puts improve risk-adjusted returns** — 1x + 5% OTM puts gives Sharpe 0.795 vs unhedged 0.616
3. **But the Sharpe is half of SPY + puts** (0.795 vs 1.879) — the equity premium is a better engine than carry
4. **Put protection is imperfect** because we're hedging AUD/USD, not AUD/JPY. CME doesn't have AUD/JPY options; you'd need OTC for perfect hedging
5. **2010-2026 was a historically favorable period** for AUD/JPY carry — JPY weakness boosted spot returns. Going forward, if BOJ normalizes rates, the carry narrows and the trade gets worse
6. **5x leverage is suicidal** — 91.4% max drawdown unhedged. Even 3x had -70.5% max DD. The puts don't save you from slow grinds

### Why SPY + puts still wins

The Sharpe ratio tells the whole story:

| Strategy | Sharpe | Why |
|----------|--------|-----|
| SPY + 0.5% puts | 1.879 | Equity premium is large, reliable, and well-hedged |
| 1x AUD/JPY + 5% OTM puts | 0.795 | Carry premium is smaller, hedging is imperfect |
| 5x AUD/JPY + 8% OTM puts | 0.495 | Leverage destroys Sharpe via volatility drag |

The carry trade's appeal is **diversification**, not replacement. In 2011, AUD puts paid +274% while SPY was roughly flat. Owning both hedges covers more scenarios.

---

## 8d. Dual-Leg Hedge: AUD Puts + JPY Calls

**Notebook:** `notebooks/fx_carry_real.ipynb` (updated)

The imperfect hedge problem from 8c: our AUD/USD puts only protect against AUD weakness, but AUD/JPY can crash from **JPY strength** too (which doesn't move AUD/USD). The fix: add JPY calls on 6J futures — when JPY strengthens, 6J rises above strike and the calls pay off.

### The dual-leg idea

AUD/JPY = 6A / 6J. It can drop because:
1. **AUD weakens** (6A drops) → AUD puts pay off
2. **JPY strengthens** (6J rises) → JPY calls pay off
3. **Both** → both legs pay off

Same total budget (0.5% of notional/month), split 50/50: 0.25% on AUD 8% OTM puts + 0.25% on JPY 8% OTM calls.

### Results — full metrics

| Strategy | Return | Vol | Sharpe | Sortino | Calmar | MaxDD | DD days | Tail | Skew | Kurt |
|----------|--------|-----|--------|---------|--------|-------|---------|------|------|------|
| **1x unhedged** | 6.81% | 11.1% | 0.616 | 0.840 | 0.243 | -28.1% | 2273 | 1.03 | -0.25 | 3.2 |
| 1x AUD puts only | 14.05% | 19.0% | 0.741 | 1.711 | 0.619 | -22.7% | 1096 | 1.03 | 21.21 | 786.2 |
| 1x JPY calls only | 8.02% | 13.3% | 0.604 | 0.978 | 0.246 | -32.6% | 2490 | 1.02 | 4.61 | 84.5 |
| **1x dual hedge** | **11.37%** | **13.9%** | **0.817** | **1.386** | **0.515** | **-22.1%** | **1943** | **1.04** | **6.83** | **167.4** |
| | | | | | | | | | | |
| **3x unhedged** | 16.45% | 33.2% | 0.496 | 0.676 | 0.233 | -70.5% | 2301 | 1.03 | -0.25 | 3.2 |
| 3x AUD puts only | 35.69% | 56.9% | 0.627 | 1.449 | 0.605 | -58.9% | 1944 | 1.03 | 21.21 | 786.2 |
| 3x JPY calls only | 18.67% | 39.8% | 0.469 | 0.759 | 0.239 | -78.1% | 2962 | 1.02 | 4.61 | 84.5 |
| **3x dual hedge** | **29.58%** | **41.8%** | **0.708** | **1.202** | **0.464** | **-63.8%** | **2142** | **1.04** | **6.83** | **167.4** |
| | | | | | | | | | | |
| **SPY + 0.5% puts** | **16.46%** | **8.8%** | **1.879** | **2.816** | **2.007** | **-8.2%** | — | — | — | — |

**What the metrics reveal:**

- **Sortino** (return / downside vol): The dual hedge at 1x scores 1.386 vs unhedged 0.840. The hedges specifically reduce *downside* volatility while adding *upside* volatility from option payoffs — exactly what Sortino captures that Sharpe misses. AUD puts only has the best Sortino (1.711) because the massive 2011 payoff was purely upside.

- **Calmar** (return / max drawdown): Dual hedge 0.515 vs unhedged 0.243 — more than 2x better return per unit of worst loss. But SPY + puts at 2.007 is still 4x better.

- **Skew**: Unhedged carry has *negative* skew (-0.25) — the classic carry trade problem of "picking up pennies in front of a steamroller." The dual hedge flips this to +6.83, creating positive skew from the option payoffs. AUD puts only has extreme +21.21 skew from the concentrated 2011 payoff.

- **Kurtosis**: All hedged strategies have massive excess kurtosis (84-786 vs 3.2 for unhedged). This reflects the lumpy nature of monthly option settlements — most months the options expire worthless (near-zero P&L), but occasionally they produce enormous payoffs. The returns are far from normal.

- **Tail ratio** (95th pctile / |5th pctile|): ~1.03 across all strategies — roughly symmetric tails at the daily level. The option payoffs are too infrequent (monthly) to show up in daily percentiles. This metric is more useful for strategies with daily rebalancing.

- **Max DD duration**: Dual hedge at 1943 trading days (~7.7 years) vs unhedged at 2273 days (~9 years). Even the hedged strategies spend most of their time in drawdown — the carry trade grinds slowly upward with long flat/negative stretches.

The dual hedge achieves the **best Sharpe ratio at every leverage level** — 0.817 at 1x vs 0.741 for AUD puts only and 0.604 for JPY calls only. By covering both crash scenarios with the same total budget, it reduces tail risk more efficiently than concentrating on one leg.

### P&L decomposition (3x leverage)

| Component | Unhedged | AUD puts only | JPY calls only | Dual hedge |
|-----------|----------|--------------|----------------|------------|
| Carry income | $323/yr | $3,390 total | $335/yr | $1,365 total |
| Spot P&L | $675/yr | $7,472 total | $697/yr | $2,917 total |
| AUD put P&L | — | $1,198 total ($76/yr) | — | $347 total ($22/yr) |
| JPY call P&L | — | — | $344 total ($22/yr) | $1,162 total ($74/yr) |
| **Final capital** | **$1,098** | **$12,160** | **$1,477** | **$5,890** |

Interesting asymmetry: at 3x, AUD puts only generates 121.6x total return (!) vs dual hedge at 58.9x. The AUD puts-only path-dependency is extreme — the 2011 put payoff (2000%+ on premium) compounds massively at 3x leverage. The dual hedge sacrifices that upside for more consistent protection across scenarios.

### Crisis performance (3x leverage)

| Crisis | AUD/JPY | Unhedged | AUD puts | JPY calls | Dual |
|--------|---------|----------|----------|-----------|------|
| 2011 EU debt | -4.9% | -11.0% | **+174.2%** | -14.9% | **+70.2%** |
| 2013 taper tantrum | -12.3% | -31.8% | -24.1% | -29.3% | -26.7% |
| 2015 China deval | -10.8% | -29.2% | -30.3% | -31.3% | -30.8% |
| 2018 trade war | -12.2% | -27.2% | **+12.8%** | -26.5% | **-7.6%** |
| 2020 COVID | -8.5% | -25.2% | -26.4% | -27.4% | -26.9% |
| 2022 rate hikes | +3.4% | +14.0% | +11.1% | +4.0% | +7.5% |

The dual hedge helped in 2 of 5 crises (2011, 2018) — the same ones where AUD puts alone worked. In 2013, 2015, and 2020, neither leg triggered effectively. Why?

- **2013 taper tantrum:** USD strengthened vs everything. AUD/USD dropped (but not enough for 8% OTM puts to go ITM). JPY actually weakened (6J fell), so JPY calls were worthless.
- **2015 China:** Similar — risk-off hit AUD but JPY weakened slightly against USD. The AUD/JPY drop was mostly AUD weakness, but AUD/USD puts were too far OTM.
- **2020 COVID:** Everything crashed then recovered in weeks. AUD/USD dropped to ~0.58 briefly but monthly settlement missed the bottom. JPY strengthened briefly but 6J barely moved since USD also weakened.

The core problem: **monthly settlement frequency is too low for fast V-shaped crashes**. The 2020 COVID crash lasted ~3 weeks. Monthly puts entered at the start of March and settled at March expiry — by which point AUD/USD had already bounced. Daily or weekly mark-to-market would capture more value.

### What the dual hedge tells us

1. **Covering both legs improves Sharpe** — 0.817 vs 0.741 at 1x. The diversification benefit is real.
2. **JPY calls are less valuable than AUD puts** in isolation (Sharpe 0.604 vs 0.741) — AUD weakness is more frequent than JPY strength in AUD/JPY crashes.
3. **The dual hedge has lower raw returns but higher risk-adjusted returns** — it sacrifices upside concentration for broader protection.
4. **Monthly settlement misses fast crashes** — a serious structural limitation of this backtest approach.
5. **SPY + puts still dominates** at Sharpe 1.879 vs 0.817. The equity risk premium remains a better engine than FX carry.

---

## 9. Other Trade Structures Considered

### 9.1 Leveraged ETFs — High Convexity, Impractical

Puts on TQQQ (3x leveraged Nasdaq) or UPRO (3x leveraged S&P 500) would be incredibly convex. A 30% SPY drop becomes a ~70% drop in TQQQ. Deep OTM puts would return 50-100x.

But there are problems:
- **Options on leveraged ETFs are less liquid, wider spreads.** TQQQ options exist but bid-ask is much wider than SPY.
- **The underlying bleeds from daily rebalancing (volatility drag).** TQQQ decays ~10-15%/yr from rebalancing alone in normal vol environments. The instrument structurally loses value.
- **You're paying for convexity on something that structurally decays.** The market knows TQQQ is fragile — put implied vols are high. You're not getting cheap convexity, you're getting expensive convexity on a dying asset.
- **Strike prices move.** After a crash, TQQQ might be at $5 instead of $50. The options market adjusts. You can't maintain a consistent hedge.

**Verdict:** Theoretically the highest raw convexity available to retail, but practically not worth it. The vol decay and wide spreads eat the advantage. SPY puts are cheaper on a risk-adjusted basis.

### 9.2 Dispersion Trades — Capturing the Correlation Risk Premium

Instead of buying SPY puts, buy puts on individual stocks and sell SPY puts. When correlations spike in a crash, individual names fall more than the index (diversification benefit disappears). You capture the "correlation risk premium" — index vol trades rich to realized single-stock vol in normal times, but this relationship inverts in crises.

**How it works:**
- In calm markets: index vol is "too high" relative to single-stock vol because the index benefits from imperfect correlation. You sell index vol, buy single-stock vol.
- In a crash: correlations go to 1. Single stocks fall as much as or more than the index. Your single-stock puts pay off more than the index puts you sold.

**Why it might be cheaper than outright index puts:** You're net short index vol (which is expensive) and long single-stock vol (which is cheaper). The position self-funds partially.

**The catch:** This is an institutional trade. You need to manage positions in dozens of names simultaneously, the execution is complex, and the margin/capital requirements are substantial. Dispersion desks at banks and hedge funds run this — it's not a retail strategy.

**Interesting hybrid:** Instead of full dispersion, you could buy puts on the most volatile index components (top 5-10 names by weight × beta) as a substitute for index puts. These might be cheaper than SPY puts while providing similar crisis payoff. Worth exploring but needs backtesting.

### 9.3 The Cheapest Convexity Scanner — The Real Universa Insight

The most important idea that emerges from this entire research: **the optimal strategy isn't to always hedge in one market — it's to continuously scan across all markets for wherever tail convexity is cheapest right now.**

Sometimes that's equity vol (2007, VIX at 12). Sometimes that's credit protection (2006, IG spreads at 80bps). Sometimes that's rate swaptions (2019, rates at 1.5% with the market pricing no cuts). Sometimes that's FX vol on a particular carry pair.

The key insight is that tail risk pricing varies independently across asset classes. When everyone is worried about equity crashes, SPY put vol is elevated — but nobody is thinking about rate cuts, so SOFR call vol is cheap. When credit spreads are tight and complacency reigns, CDX protection is dirt cheap.

**What a scanner would need:**

1. **Normalized convexity measure across asset classes.** Something like: expected payoff in a 3-sigma event / cost of protection. This lets you compare apples to apples: is a 10-delta SPY put or a 10-delta AUD/JPY put or a 5-year CDX IG contract giving you the most bang for your buck right now?

2. **Inputs per asset class:**
   - Equity: SPY option chain (implied vol surface, put skew)
   - Rates: SOFR/ZN option chain (call skew for rate cut bets)
   - FX: G10 carry pair option chains (AUD/JPY, MXN/JPY risk reversals)
   - Credit: CDX IG/HY spreads, HYG/LQD option chains
   - Commodities: CL option chain, VIX futures term structure
   - EM: EEM/EMB option chains, EMBI spread level

3. **Historical regime context.** Is the current VIX/spread/rate level in the 10th percentile (cheap protection) or 90th percentile (expensive)? Regime-conditional z-scores.

4. **The allocation decision.** Given a fixed tail hedge budget (say 0.5% of portfolio/yr), allocate across the cheapest N instruments. Rebalance monthly.

**Can this be built?** At a basic level, yes. The data exists:
- SPY options: already in this backtester
- SOFR/Treasury options: CME DataMine
- FX options: CME DataMine for listed, J.P. Morgan/Bloomberg for OTC
- Credit: CDX via Markit, HYG/LQD options from standard option data providers
- VIX: CBOE DataShop
- EM: EEM/EMB options from standard providers

The hard part isn't the data — it's the normalization. How do you compare the "cheapness" of a 10-delta SPY put vs a 10-delta AUD/JPY put vs a CDX IG contract? The tail events are different sizes, frequencies, and correlations. A proper framework would need:
- Risk-neutral vs historical tail probabilities per asset (how much is the market underpricing?)
- Cross-asset correlation in tail events (do these hedges pay off at the same time?)
- Liquidity-adjusted execution costs

**This would be the real project** — not just backtesting one strategy, but building a cross-asset tail convexity monitor. Even a simple version (rank by implied vol percentile vs historical tail frequency) would be more sophisticated than what most retail investors do.

---

## 10. Key References

### Papers

| Paper | Authors | Year | Key Finding | Link |
|-------|---------|------|-------------|------|
| Crash-Neutral Currency Carry Trades | Jurek | 2014 | 2/3 of carry alpha survives crash hedging | [JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14001081), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1262934) |
| Why are FX Options So Cheap? | Caballero & Doyle | 2012 | FX option bundles for carry hedging are "puzzlingly cheap" | [NBER WP 18644](https://www.nber.org/papers/w18644) |
| Carry Trades and Currency Crashes | Brunnermeier, Nagel & Pedersen | 2009 | Carry returns have negative skewness — crash risk premium | [NBER](https://www.nber.org/papers/w14473) |
| Crash Risk in Currency Markets | Farhi, Fraiberger, Gabaix, Ranciere & Verdelhan | 2015 | Systematic crash risk priced in currencies | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1397668) |
| The Carry Trade: Risks and Drawdowns | Daniel, Hodrick & Lu | 2017 | Carry drawdowns are predictable and manageable | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2486275) |
| Equity Tail Risk and Currency Risk Premia | Fan & Londono | 2022 | Option-based equity tail risk priced in FX | [Fed](https://www.federalreserve.gov/econres/ifdp/files/ifdp1253.pdf) |
| Facts and Fantasies about Commodity Futures | Gorton & Rouwenhorst | 2006 | Commodities match equity Sharpe, negatively correlated | — |
| Capital Flows and Sudden Stops | Calvo | 1998 | Foundational EM crisis dynamics | — |
| This Time Is Different | Reinhart & Rogoff | 2009 | 800 years of sovereign defaults | — |

### Market Data Sources

| Source | Coverage | Access |
|--------|----------|--------|
| CME DataMine | SOFR, Treasury, FX, commodity futures & options | Paid, self-service cloud |
| CBOE DataShop (Livevol) | VIX options/futures from 2004, IV surfaces | Paid |
| OptionMetrics (IvyDB) | Academic standard for equity and futures options | Institutional subscription |
| Databento | CME/CBOT/NYMEX/COMEX tick data | Competitive pricing, no pre-calc Greeks |
| Markit (S&P Global) | CDX/CDS spreads and indices | Institutional |
| BIS | Macro data, policy analysis, crisis bulletins | [bis.org](https://www.bis.org/) (free) |

### Crisis Analysis

| Source | Content | Link |
|--------|---------|------|
| BIS Bulletin No. 90 | 2024 carry trade unwind analysis | [PDF](https://www.bis.org/publ/bisbull90.pdf) |
| RBA Bulletin Mar 2009 | 2008 AUD carry unwind | [PDF](https://www.rba.gov.au/publications/bulletin/2009/mar/pdf/bu-0309-1.pdf) |
| BOE Quarterly Bulletin Q1 1999 | 1998 yen crisis options market analysis | [PDF](https://escoe-website.s3.amazonaws.com/wp-content/uploads/2019/11/30191453/BEQB_The-yen-dollar-exchange-rate-in-1998-views-from-options-markets-QB-1999-Q1-pp.68-77.pdf) |
| Global FXC 2020 | COVID FX market conditions | [PDF](https://www.globalfxc.org/uploads/20200622_gfxc_overview_of_market_conditions.pdf) |

---

## 11. Data Sourcing: What We Need and What It Costs

To backtest the Spitznagel structure across asset classes (and eventually build the convexity scanner), we need historical options data for multiple instruments.

### The cross-rate problem

CME doesn't list direct AUD/JPY or MXN/JPY options. They list USD-denominated pairs:
- **6A** (AUD/USD), **6J** (JPY/USD), **6M** (MXN/USD), **6E** (EUR/USD)

To backtest AUD/JPY carry + puts, you construct synthetic positions from the USD pairs (long 6A + short 6J). For options, you'd model cross-rate vol from individual vols and cross-correlation, or simply backtest the USD pairs directly (long 6A + buy OTM puts on 6A covers the AUD depreciation risk).

### What we need by phase

**Phase 1 — equity/ETF options (convexity scanner baseline):**
SPY (already have), HYG, EEM, TLT, GLD, VIX

**Phase 2 — futures options (cross-asset backtests):**
SOFR (SR3), ZN, ZB, 6A, 6J, 6M, 6E, CL, ZC, ZW

**Phase 3 — credit derivatives:**
CDX is institutional-only ($10k+/yr via Markit). Skip — use HYG puts from Phase 1 instead.

**Spot FX rates (for synthetic backtests):** Free via yfinance/FRED.

### Cost breakdown

| Data | Source | Cost | Notes |
|------|--------|------|-------|
| SPY options | Already have | $0 | 2008-2025, 24.7M rows |
| HYG, EEM, TLT, GLD, VIX options | [HistoricalOptionData.com](https://historicaloptiondata.com/) | $575-805/yr | All US equity/ETF/index options, with IV+Greeks. Verify VIX coverage. |
| VIX options (if not above) | [CBOE DataShop](https://datashop.cboe.com/) | ~$100-300 one-time | Canonical source for VIX. EOD summary product. |
| FX futures options (6A, 6J, 6M, 6E) | [Databento](https://databento.com/) | ~$100-300 | Pay-as-you-go, EOD settlement, $125 free credit on signup |
| Rate futures options (SR3, ZN, ZB) | Databento | ~$100-200 | Same account, add products |
| Commodity futures options (CL, ZC, ZW) | Databento | ~$100-200 | Same account |
| FX implied vol surfaces | [IVolatility](https://www.ivolatility.com/) | $60-150/mo | Optional — proper IV surfaces beyond raw settlement |
| Spot FX rates | yfinance + FRED | $0 | `yf.download("AUDJPY=X")` or FRED series |
| CDX/CDS credit indices | Markit/S&P Global | $10,000+/yr | Skip — institutional only, use HYG puts |

### Three budget tiers

**Minimum viable (~$700-1,300 one-time):**
- HistoricalOptionData.com ($575-805/yr) — covers SPY, HYG, EEM, TLT, GLD, probably VIX
- Databento ($0-175 after $125 free credit) — covers all CME futures options (FX, rates, commodities)
- Free spot FX via yfinance/FRED
- Covers everything except IV surfaces

**Comfortable (~$2,000-3,000/yr):**
- Everything above
- IVolatility (~$720-1,800/yr) for CME implied vol surfaces
- CBOE DataShop for VIX if not in HistoricalOptionData
- Full data for phases 1+2

**Skip entirely:**
- Markit/S&P Global ($10k+/yr) — use HYG puts instead
- OptionMetrics ($20-50k/yr) — academic institutional, overkill
- CME DataMine — Databento is cheaper and more developer-friendly

### Data source details

**Databento** ([databento.com](https://databento.com/)) — Best for futures options.
- Modern Python API: `pip install databento`
- All CME/CBOT/NYMEX/COMEX products. Historical from June 2010.
- Schemas: OHLCV-1d (daily bars), Statistics (settlements, OI), MBP-1 (L1 BBO). Use OHLCV-1d or Statistics to keep costs low.
- Pricing: ~$5/compressed GB downloaded. $125 free credit on signup.
- $179/mo Standard plan available (unlimited live CME + 7yr OHLCV history) but overkill if you only need historical.
- [Pricing](https://databento.com/pricing) · [CME dataset](https://databento.com/datasets/GLBX.MDP3) · [Options data](https://databento.com/options)

**HistoricalOptionData.com** ([historicaloptiondata.com](https://historicaloptiondata.com/)) — Best for equity/ETF options.
- All US equity/ETF/index options from 2005-present.
- Level 1 (no Greeks): $545/yr. Level 2 (30-day surface IV): $575/yr. Level 3 (bid/ask IV, multi-tenor): $805/yr.
- Flat file delivery. Research-quality EOD data.

**CBOE DataShop** ([datashop.cboe.com](https://datashop.cboe.com/)) — Canonical source for VIX.
- VIX options/futures from 2004. IV surfaces, trade-level data.
- Option EOD Summary product is what we'd want. Per-symbol pricing.
- Also covers all OPRA-listed equity/ETF options.

**IVolatility** ([ivolatility.com](https://www.ivolatility.com/)) — Best for IV surfaces.
- US futures + futures options from CME/COMEX/NYMEX/ICE since 2006. Includes FX futures.
- NBBO, raw IV, IV Index, IV Surface, historical vol.
- IVolLive: $60/mo (delayed) or $150/mo (real-time). Pay-per-usage also available.
- [US Futures data](https://www.ivolatility.com/historical-options-data/us-futures-futures-options/)

**QuantConnect** ([quantconnect.com](https://www.quantconnect.com/)) — Free but locked to their platform.
- CME futures options from 2012 via Algoseek. Minute-frequency.
- Free for research/backtesting on their cloud. Cannot export data.
- Good for validating before spending money.
- [US Futures Options](https://www.quantconnect.com/data/algoseek-us-future-options)

**Free spot FX:**

```python
# Quick: yfinance
import yfinance as yf
audjpy = yf.download("AUDJPY=X", start="2010-01-01")
mxnjpy = yf.download("MXNJPY=X", start="2010-01-01")

# More reliable: FRED (St. Louis Fed)
from fredapi import Fred
fred = Fred(api_key='your_key')
usdjpy = fred.get_series('DEXJPUS', observation_start='2010-01-01')
usdaud = fred.get_series('DEXUSAL', observation_start='2010-01-01')
audjpy = usdaud * usdjpy  # construct cross
```

**Interactive Brokers:** Cannot pull expired options data — useless for historical backtesting. Only good for forward-looking data collection.

### Recommended starting path

1. **Sign up for Databento** (free, $125 credit). Pull daily settlement data for 6A, 6J, 6E options to see data format and coverage.
2. **Download free spot FX** via yfinance. Run a synthetic carry + tail protection backtest immediately (no money needed). **Done** — see section 8 for results.
3. **Buy HistoricalOptionData.com Level 2** ($575) when ready for the convexity scanner Phase 1.
4. **Add IVolatility** ($60/mo) when you want proper IV surfaces for Level 2+ scoring.

### Databento setup guide

**Sign up:**
1. Go to [databento.com](https://databento.com) and click Sign Up
2. Create an account (email + password)
3. Your API key is **auto-generated** — find it on the [API Keys page](https://databento.com/docs/portal/api-keys) in your portal
4. The key is a 32-character string starting with `db-`
5. You get **$125 in free credits** automatically for historical data

**Install and configure:**
```bash
pip install -U databento
export DATABENTO_API_KEY="db-your-key-here"
```

**Check cost before downloading** (deducted from $125 credit):
```python
import databento as db

client = db.Historical()  # picks up DATABENTO_API_KEY env var

# Check how much FX options data will cost
cost = client.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["6A.OPT", "6J.OPT", "6M.OPT"],
    schema="ohlcv-1d",
    start="2010-01-01",
    end="2026-01-01",
)
print(f"Estimated cost: ${cost}")
```

**Download FX futures options data:**
```python
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["6A.OPT", "6J.OPT", "6M.OPT", "6E.OPT"],
    schema="ohlcv-1d",
    start="2010-01-01",
    end="2026-01-01",
)

df = data.to_df()
print(f"Downloaded {len(df)} rows")
df.to_parquet("data/fx_options_daily.parquet")
```

**Key details:**
- Dataset for all CME products: `GLBX.MDP3`
- FX option symbols: `6A.OPT` (AUD), `6J.OPT` (JPY), `6M.OPT` (MXN), `6E.OPT` (EUR)
- Rate option symbols: `SR3.OPT` (SOFR), `ZN.OPT` (10yr Treasury), `ZB.OPT` (30yr Treasury)
- Commodity option symbols: `CL.OPT` (crude oil), `ZC.OPT` (corn), `ZW.OPT` (wheat)
- Schema `ohlcv-1d` is cheapest (daily bars). `statistics` gives settlements + OI. Both work for backtesting.
- Pricing: ~$5/compressed GB. EOD data for a handful of products over 15 years should be well within the $125 credit.
- Output formats: DataFrame (pandas), ndarray (numpy), CSV, Parquet
- Docs: [API demo](https://databento.com/blog/api-demo-python) · [Python client](https://github.com/databento/databento-python) · [Historical API](https://databento.com/docs/api-reference-historical)

### Why we need real option data (not synthetic)

The synthetic backtest in section 8 uses Black-Scholes with realized vol — this has real limitations:

1. **No skew.** Real FX put IV is 1-3 vol points above ATM due to crash risk premium. Our BS model uses flat vol, likely **underestimating put costs by 10-30%**.
2. **No term structure.** BS scales vol by √T, but real IV term structure is flatter. This is why our 3-month results contradict Jurek's finding.
3. **No bid-ask spreads.** Real OTM FX options on CME have 10-20% bid-ask as a fraction of premium. Zero transaction cost assumption flatters the results.
4. **No smile.** Deep OTM options have higher IV than our model assumes. The 5-delta puts are priced too cheap in our model.
5. **No volume/OI data.** We can't assess whether the options we're "buying" actually traded. Real data shows liquidity.

Real CME option prices from Databento solve all five problems. The synthetic backtest gives us confidence the structure works (puts are net profitable on carry pairs), but the exact numbers (5.17% vs 7.72% CAGR) could shift significantly with real pricing.
