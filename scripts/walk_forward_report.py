"""Walk-forward / OOS harness with aggregated report table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from options_portfolio_backtester.analytics.optimization import walk_forward
from options_portfolio_backtester.analytics.stats import BacktestStats
from backtester import Backtest
from backtester.datahandler import TiingoData
from backtester.enums import Stock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward OOS harness for stock-only benchmark.")
    p.add_argument("--stocks-file", default="data/processed/stocks.csv")
    p.add_argument("--symbols", default="SPY")
    p.add_argument("--weights", default=None)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--rebalance-months", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--in-sample-pct", type=float, default=0.70)
    p.add_argument("--output", default="data/processed/walk_forward_report.csv")
    return p.parse_args()


def normalize_weights(symbols: list[str], raw_weights: str | None) -> list[float]:
    if raw_weights is None:
        return [1.0 / len(symbols)] * len(symbols)
    vals = [float(x) for x in raw_weights.split(",")]
    if len(vals) != len(symbols):
        raise ValueError("--weights length must match --symbols length")
    total = float(sum(vals))
    if total <= 0:
        raise ValueError("--weights must sum to > 0")
    return [v / total for v in vals]


def run_stock_only(
    stocks_file: str,
    symbols: list[str],
    weights: list[float],
    initial_capital: float,
    rebalance_months: int,
) -> pd.Series:
    stocks_data = TiingoData(stocks_file)
    stocks = [Stock(sym, w) for sym, w in zip(symbols, weights)]
    bt = Backtest({"stocks": 1.0, "options": 0.0, "cash": 0.0}, initial_capital=int(initial_capital))
    bt.stocks = stocks
    bt.stocks_data = stocks_data
    bt.run(rebalance_freq=rebalance_months, rebalance_unit="BMS")
    return bt.balance["total capital"].dropna()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    weights = normalize_weights(symbols, args.weights)

    prices = pd.read_csv(args.stocks_file, parse_dates=["date"])
    prices = prices[prices["symbol"].isin(symbols)]
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))

    def _run_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[BacktestStats, pd.DataFrame]:
        sliced = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
        tmp_path = Path("data/processed/_walk_forward_tmp_stocks.csv")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        sliced.to_csv(tmp_path, index=False)
        equity = run_stock_only(
            stocks_file=str(tmp_path),
            symbols=symbols,
            weights=weights,
            initial_capital=args.initial_capital,
            rebalance_months=args.rebalance_months,
        )
        bal = pd.DataFrame({"total capital": equity})
        bal["% change"] = bal["total capital"].pct_change()
        stats = BacktestStats.from_balance(bal)
        return stats, bal

    wf = walk_forward(
        run_fn=_run_window,
        dates=dates,
        in_sample_pct=args.in_sample_pct,
        n_splits=args.n_splits,
    )

    rows: list[dict[str, float | int | str]] = []
    for is_res, oos_res in wf:
        split = int(is_res.params["split"])
        rows.append({
            "split": split,
            "in_sample_sharpe": is_res.stats.sharpe_ratio,
            "in_sample_return": is_res.stats.total_return,
            "oos_sharpe": oos_res.stats.sharpe_ratio,
            "oos_return": oos_res.stats.total_return,
        })
    out = pd.DataFrame(rows).sort_values("split")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\noutput: {out_path}")


if __name__ == "__main__":
    main()
