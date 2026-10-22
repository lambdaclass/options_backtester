"""One definition per metric, annualized from the data's own timestamps.

Sortino used the sample std of only the negative returns (equal losses gave
8.3e14); CAGR assumed 252 rows a year (a weekly balance reported 8,020%
instead of 148.5%); BacktestResults.sharpe_ratio was CAGR / volatility while
BacktestStats used mean / std.
"""

import math

import numpy as np
import pandas as pd
import pytest

from options_portfolio_backtester import BacktestResults, BacktestStats


def _balance(returns, freq="B", start="2020-01-01"):
    idx = pd.date_range(start, periods=len(returns) + 1, freq=freq)
    total = 100_000 * np.cumprod([1.0, *(1 + np.asarray(returns))])
    return pd.DataFrame({"total capital": total}, index=idx)


def test_sortino_uses_downside_deviation_over_all_periods():
    rets = np.array([0.01, -0.01, 0.02, -0.01, 0.015, -0.01] * 50)
    stats = BacktestStats.from_balance(_balance(rets))
    ppy = len(rets) / ((pd.bdate_range("2020-01-01", periods=len(rets) + 1)[-1]
                        - pd.Timestamp("2020-01-01")).days / 365.25)
    downside = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert stats.sortino_ratio == pytest.approx(rets.mean() / downside * math.sqrt(ppy), rel=1e-6)
    assert stats.sortino_ratio < 1e3


def test_weekly_balance_annualizes_on_calendar_time():
    # 104 weekly steps of +1%: two years, (1.01^104)^(1/2) - 1 = 67.8%/yr.
    stats = BacktestStats.from_balance(_balance([0.01] * 104, freq="W"))
    years = (104 * 7) / 365.25
    assert stats.annualized_return == pytest.approx(1.01 ** (104 / years) - 1, rel=1e-6)


def test_results_and_stats_agree_on_sharpe():
    rng = np.random.default_rng(1)
    bal = _balance(rng.normal(0.0005, 0.01, 750))
    results = BacktestResults(balance=bal, trade_log=None, config={})
    stats = BacktestStats.from_balance(bal)
    assert results.sharpe_ratio == pytest.approx(stats.sharpe_ratio, rel=1e-3)
