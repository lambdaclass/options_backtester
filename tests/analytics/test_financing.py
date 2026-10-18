"""Post-hoc interest on cash and borrowing."""

import pandas as pd
import pytest

from options_portfolio_backtester import apply_financing

IDX = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])


def _balance(cash, total):
    return pd.DataFrame({"cash": cash, "total capital": total}, index=IDX)


def test_no_cash_no_interest():
    bal = _balance([0.0, 0.0, 0.0], [100.0, 110.0, 120.0])
    out = apply_financing(bal, pd.Series(0.05, index=IDX), borrow_spread=0.01)
    assert out.tolist() == pytest.approx([100.0, 110.0, 120.0])


def test_positive_cash_earns_rate_on_calendar_days():
    bal = _balance([1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0])
    out = apply_financing(bal, pd.Series(0.0365, index=IDX))
    day1 = 1 + 1000 * 0.0365 / 365 / 1000           # 1 day
    day2 = 1 + 1000 * 0.0365 * 2 / 365 / 1000       # Jan 2 -> Jan 4
    assert out.tolist() == pytest.approx([1000.0, 1000.0 * day1, 1000.0 * day1 * day2])


def test_borrowing_pays_rate_plus_spread():
    bal = _balance([-500.0, -500.0, -500.0], [1000.0, 1000.0, 1000.0])
    out = apply_financing(bal, pd.Series(0.03, index=IDX), borrow_spread=0.0065)
    assert out.iloc[1] == pytest.approx(1000.0 * (1 - 500 * 0.0365 / 365 / 1000))


def test_rate_is_forward_filled():
    bal = _balance([1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0])
    sparse = pd.Series([0.0365], index=pd.to_datetime(["2019-12-01"]))
    dense = pd.Series(0.0365, index=IDX)
    assert apply_financing(bal, sparse).tolist() == pytest.approx(apply_financing(bal, dense).tolist())
