# Notebooks

All notebooks use real SPY options data (2008-2025) and produce executed outputs with charts and tables.

## Run all notebooks

```bash
make notebooks
```

## Notebook Guide

### Getting Started
- **[quickstart.ipynb](quickstart.ipynb)** — Load data, define a strategy, run a backtest, plot results. Start here.

### Research Findings
- **[findings.ipynb](findings.ipynb)** — Full allocation sweep (puts vs calls), macro signal analysis, crash-period breakdown. Key finding: calls add modest alpha, puts drag, macro signals don't help timing.
- **[paper_comparison.ipynb](paper_comparison.ipynb)** — 10 strategies tested against academic paper claims. Styled comparison tables, risk/return scatter by category, crash heatmap.

### The Tail Hedge Debate
- **[spitznagel_case.ipynb](spitznagel_case.ipynb)** — The main analysis. Tests both AQR framing (no leverage, always loses) and Spitznagel framing (100% equity + puts on top, outperforms). Multi-dimensional parameter sweep across DTE, delta, exit timing, budget, and rebalance frequency. Implementation guide included. **Conclusion: Spitznagel is right.**

### Strategy Showcases
- **[strategies.ipynb](strategies.ipynb)** — 4 strategies head-to-head: OTM puts, OTM calls, long straddle, short strangle.
- **[volatility_premium.ipynb](volatility_premium.ipynb)** — Sell vol vs buy vol deep dive. Tests the Variance Risk Premium (Carr & Wu 2009).
- **[iron_condor.ipynb](iron_condor.ipynb)** — 4-leg iron condor income strategy.
- **[ivy_portfolio.ipynb](ivy_portfolio.ipynb)** — Endowment-style portfolio with long straddle hedge.

### Deep Dives
- **[trade_analysis.ipynb](trade_analysis.ipynb)** — Per-trade P&L analysis of the put hedge: bar charts, cumulative P&L, crash breakdowns, winner vs loser comparison.
- **[gold_sp500.ipynb](gold_sp500.ipynb)** — Multi-asset portfolio with cash/gold proxy + options overlay.

### Styling
All notebooks import `nb_style.py` for FT-inspired warm cream backgrounds, teal accents, and styled tables. Charts use crash-period shading (GFC, COVID, 2022).
