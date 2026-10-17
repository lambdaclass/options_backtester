"""Cross-underlying robustness check for the Spitznagel tail-hedge article.

The published tables are validated on SPY only. This script runs the article's
exact configuration (strike-based 40-45% OTM puts, DTE 90-180 entry / exit at
DTE 30, bi-monthly rebalance, daily exit checks, monetize-and-reinvest,
externally-funded annual budget) on QQQ and IWM — same window, same engine,
runtime invariants armed — and prints overlay-vs-buy&hold per underlying.

If the excess-return + drawdown-protection result is SPY-specific, that is
critical context for the article; if it generalizes, the thesis strengthens.

Data (per-symbol paths so the canonical SPY files are untouched):
    python scripts/fetch_data.py all --symbols QQQ --start 2008-01-02 --end 2024-12-31 \
        --stocks-output data/processed/qqq_stocks.csv --options-output data/processed/qqq_options.csv
    python scripts/fetch_data.py all --symbols IWM --start 2008-01-02 --end 2024-12-31 \
        --stocks-output data/processed/iwm_stocks.csv --options-output data/processed/iwm_options.csv
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from options_portfolio_backtester import (
    BacktestEngine,
    Direction,
    OptionType,
    Stock,
    Strategy,
    StrategyLeg,
)
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData,
    TiingoData,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
DATA = REPO_ROOT / "data" / "processed"

UNDERLYINGS = {
    "SPY": (DATA / "options.parquet", DATA / "stocks.csv"),
    "QQQ": (DATA / "qqq_options.parquet", DATA / "qqq_stocks.csv"),
    "IWM": (DATA / "iwm_options.parquet", DATA / "iwm_stocks.csv"),
}
BUDGETS = [0.005, 0.010, 0.020, 0.033]
INITIAL_CAPITAL = 1_000_000
WINDOW_END = pd.Timestamp("2024-12-31")


def _ann(s):
    years = (s.index[-1] - s.index[0]).days / 365.25
    return ((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100


def _maxdd(s):
    cm = s.cummax()
    return ((s - cm) / cm).min() * 100


def _sharpe(s):
    d = s.pct_change().dropna()
    if len(d) < 2 or d.std() == 0:
        return 0.0
    return _ann(s) / (d.std() * math.sqrt(252) * 100)


def _strategy(schema, symbol):
    """The article's strike-based 40-45% OTM put leg."""
    leg = StrategyLeg("leg_1", schema, option_type=OptionType.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == symbol)
        & (schema.dte >= 90)
        & (schema.dte <= 180)
        & (schema.strike <= schema.underlying_last * 0.60)
        & (schema.strike >= schema.underlying_last * 0.55)
    )
    leg.entry_sort = ("strike", False)
    leg.exit_filter = schema.dte <= 30
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def _run(symbol, opts, stx, budget):
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget_annual_pct = budget
    bt.options_budget_fresh_spend = True  # spend the full budget each rebalance
    bt.check_exits_daily = True
    bt.rebalance_stocks_on_exit = True
    bt.assert_invariants = True  # cash-flow + valuation guards armed
    bt.stocks = [Stock(symbol, 1.0)]
    bt.stocks_data = stx
    bt.options_data = opts
    bt.options_strategy = _strategy(opts.schema, symbol)
    bt.run(rebalance_freq=2, rebalance_unit="BMS")
    s = bt.balance["total capital"].copy()
    s.index = pd.to_datetime(s.index)
    n_trades = len(bt.trade_log)
    return s, n_trades


def _load(symbol, opts_path, stx_path, window_start=None):
    """Load and window-clamp a (options, stocks) pair; returns None if missing.

    IWM's release parquet ships adjClose as all-NaN; fall back to the raw
    close there (no dividend adjustment — buy&hold CAGR is understated by
    the dividend yield, but overlay-vs-hold uses the same series on both
    sides, so the *excess* comparison stays internally consistent).
    """
    if not opts_path.exists() or not stx_path.exists():
        return None
    opts = HistoricalOptionsData(str(opts_path))
    stx = TiingoData(str(stx_path))
    ocol, scol = opts.schema["date"], stx.schema["date"]
    opts._data = opts._data[opts._data[ocol] <= WINDOW_END]
    stx._data = stx._data[stx._data[scol] <= WINDOW_END]
    if window_start is not None:
        opts._data = opts._data[opts._data[ocol] >= window_start]
        stx._data = stx._data[stx._data[scol] >= window_start]
    # keep the engine's date-alignment assert satisfied after clamping:
    # intersect the date sets AND date-sort both frames — the assert compares
    # unique() arrays, which preserve order of appearance.
    shared = set(stx._data[scol]) & set(opts._data[ocol])
    stx._data = stx._data[stx._data[scol].isin(shared)].sort_values(scol, kind="stable")
    opts._data = opts._data[opts._data[ocol].isin(shared)].sort_values(ocol, kind="stable")
    if stx._data["adjClose"].isna().all():
        print(f"  [note] {symbol}: adjClose all-NaN in release data; using raw close "
              f"(no dividend adjustment — excess vs hold still like-for-like)")
        stx._data["adjClose"] = stx._data["close"]
    return opts, stx


def _table(symbol, opts, stx, label=None):
    scol = stx.schema["date"]
    df = stx._data.sort_values(scol)
    df = df[df[stx.schema["symbol"]] == symbol]
    bh = df.set_index(scol)["adjClose"]
    print(f"\n=== {label or symbol} ({bh.index[0].date()} → {bh.index[-1].date()}) ===")
    print(f"  {'config':24s} {'CAGR%':>8s} {'Excess':>8s} {'MaxDD%':>8s} {'Sharpe':>8s} {'trades':>7s}")
    print(f"  {'buy & hold':24s} {_ann(bh):>8.2f} {'—':>8s} {_maxdd(bh):>8.1f} {_sharpe(bh):>8.3f} {'—':>7s}")
    for budget in BUDGETS:
        s, n = _run(symbol, opts, stx, budget)
        print(
            f"  {'+ ' + f'{budget*100:.1f}%/yr deep OTM':24s} "
            f"{_ann(s):>8.2f} {_ann(s) - _ann(bh):>+8.2f} "
            f"{_maxdd(s):>8.1f} {_sharpe(s):>8.3f} {n:>7d}"
        )
    return bh.index[0]


def main():
    spy_paths = UNDERLYINGS["SPY"]

    # Full-window tables per underlying
    starts = {}
    for symbol, (opts_path, stx_path) in UNDERLYINGS.items():
        pair = _load(symbol, opts_path, stx_path)
        if pair is None:
            print(f"\n=== {symbol}: data missing ({opts_path.name}) — skipped ===")
            continue
        starts[symbol] = _table(symbol, *pair)

    # Window-matched SPY controls: an underlying whose chain starts later
    # (e.g. QQQ from 2011 — no GFC) must be compared against SPY over the
    # SAME window, otherwise "the hedge didn't pay on QQQ" is confounded
    # with "the window contained no crash that pays".
    for symbol in ("QQQ", "IWM"):
        if symbol not in starts or starts[symbol] <= starts.get("SPY", pd.Timestamp.min):
            continue
        pair = _load("SPY", *spy_paths, window_start=starts[symbol])
        if pair is not None:
            _table("SPY", *pair, label=f"SPY control over {symbol}'s window")


if __name__ == "__main__":
    main()
