"""Entry-calendar robustness helpers."""

import math

import pandas as pd
import pytest

from options_portfolio_backtester import calendar_summary, entry_calendars, run_over_calendars

DAYS = pd.bdate_range("2020-01-01", "2020-12-31")


class TestEntryCalendars:
    def test_bimonthly_phases_and_offsets(self):
        cals = entry_calendars(DAYS, every_n_months=2, day_offsets=(0, 9))
        assert list(cals) == ["m1/d1", "m1/d10", "m2/d1", "m2/d10"]
        # Cycle month 1 = Jan, Mar, ...; first trading day of each.
        assert cals["m1/d1"][:3] == [pd.Timestamp(d) for d in ("2020-01-01", "2020-03-02", "2020-05-01")]
        assert len(cals["m1/d1"]) == 6

    def test_every_calendar_starts_on_day_one(self):
        for dates in entry_calendars(DAYS, every_n_months=2).values():
            assert dates[0] == DAYS[0]
        m2 = entry_calendars(DAYS, every_n_months=2, day_offsets=(0,))["m2/d1"]
        assert m2[:2] == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-03")]

    def test_tenth_trading_day(self):
        cal = entry_calendars(DAYS, every_n_months=1, day_offsets=(9,))["m1/d10"]
        assert cal[1] == DAYS[DAYS.month == 1][9]
        assert len(cal) == 13  # day one + 12 months

    def test_offset_beyond_month_clamps_to_last_day(self):
        cal = entry_calendars(DAYS, every_n_months=1, day_offsets=(40,))["m1/d41"]
        assert cal[1] == DAYS[DAYS.month == 1][-1]

    def test_unsorted_duplicate_input(self):
        shuffled = list(DAYS[::-1]) + list(DAYS[:5])
        assert entry_calendars(shuffled) == entry_calendars(DAYS)

    def test_bad_input(self):
        with pytest.raises(ValueError):
            entry_calendars(DAYS, every_n_months=0)
        with pytest.raises(ValueError):
            entry_calendars([])


class TestRunOverCalendars:
    def test_scores_series_results(self):
        idx = pd.bdate_range("2020-01-01", "2022-01-01")
        growth = {"a": 1.10, "b": 1.21}

        def run(dates):
            g = growth["a"] if len(dates) % 2 else growth["b"]
            return pd.Series(100 * g ** (pd.Series(range(len(idx))) / (len(idx) - 1)).values, index=idx)

        cals = {"odd": [1, 2, 3], "even": [1, 2]}
        out = run_over_calendars(run, cals)
        assert list(out.index) == ["odd", "even"]
        assert out.loc["odd", "final"] == pytest.approx(110)
        assert out.loc["even", "max_drawdown"] == pytest.approx(0)
        assert math.isnan(out.loc["odd", "fill_rate"])

    def test_summary(self):
        df = pd.DataFrame({"cagr": [1.0, -1.0, 3.0]}, index=["x", "y", "z"])
        s = calendar_summary(df)
        assert (s["mean"], s["worst"], s["best"], s["spread"], s["calendars"]) == (1.0, -1.0, 3.0, 4.0, 3.0)

    def test_engine_results(self, tmp_path):
        """End to end with the golden-scenario data: an engine per calendar."""
        from tests.oracles.test_daily_exit_accounting import _quoted_rows, _run
        rows = _quoted_rows()
        calls, engines = [], []

        def run(dates):
            calls.append(dates)
            engines.append(_run(tmp_path, rows, dte_exit=28))
            return engines[-1]

        out = run_over_calendars(run, {"one": ["2020-06-01"]})
        assert calls == [["2020-06-01"]]
        eng = engines[0]
        assert out.loc["one", "fill_rate"] == eng.option_fill_rate
        assert out.loc["one", "final"] == eng.balance["total capital"].iloc[-1]
