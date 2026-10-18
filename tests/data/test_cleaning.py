"""Each chain-cleaning check flags its hazard and nothing else."""

import pandas as pd
import pytest

from options_portfolio_backtester import clean_options_chain, flag_bad_quotes


def _row(strike, bid, ask, root="SPX", typ="put", spot=1000.0, exp="2020-03-20"):
    return {"quotedate": "2020-01-02", "expiration": exp, "type": typ, "strike": strike,
            "bid": bid, "ask": ask, "underlying_last": spot, "optionroot": root}


def _chain(rows):
    return pd.DataFrame(rows)


def test_clean_chain_flags_nothing():
    # Prices change with the strike; a two-strike penny plateau is normal.
    chain = _chain([_row(700, 1.0, 1.2), _row(725, 1.4, 1.6), _row(750, 1.9, 2.1),
                    _row(500, 0.05, 0.10), _row(525, 0.05, 0.10)])
    assert not flag_bad_quotes(chain).any().any()
    assert len(clean_options_chain(chain)) == 5


def test_roots_only_when_asked():
    chain = _chain([_row(700, 1.0, 1.2), _row(700, 0.19, 0.25, root="BSK")])
    assert not flag_bad_quotes(chain)["roots"].any()
    flags = flag_bad_quotes(chain, drop_roots=("BSK", "BSF"))
    assert flags["roots"].tolist() == [False, True]


def test_flat_print_across_strikes():
    # Strikes 650-725 at one price with spot 909: a 75-point (8%) plateau.
    flat = [_row(k, 64.30, 75.10, spot=909.0) for k in (650, 675, 700, 725)]
    other = [_row(800, 70.0, 72.0, spot=909.0)]
    flags = flag_bad_quotes(_chain(flat + other))
    assert flags["flat"].tolist() == [True] * 4 + [False]


def test_flat_needs_run_span_and_price():
    two = [_row(k, 5.0, 5.5) for k in (700, 725)]                 # run of 2
    narrow = [_row(k, 5.0, 5.5) for k in (700, 705, 710)]         # spans 1% of spot
    penny = [_row(k, 0.05, 0.10) for k in (500, 550, 600)]        # ask < 1
    for rows in (two, narrow, penny):
        assert not flag_bad_quotes(_chain(rows))["flat"].any()


def test_flat_does_not_cross_expirations():
    rows = [_row(700, 5.0, 5.5, exp="2020-03-20"), _row(725, 5.0, 5.5, exp="2020-03-20"),
            _row(750, 5.0, 5.5, exp="2020-06-19")]
    assert not flag_bad_quotes(_chain(rows))["flat"].any()


def test_time_value_above_bound():
    # Put 900 with spot 1000: intrinsic 0; a 300 bid is 30% of spot.
    chain = _chain([_row(900, 300.0, 310.0), _row(1100, 120.0, 125.0),
                    _row(900, 300.0, 310.0, typ="call")])
    # The call is 100 in the money: 200 of time value, 20% of spot -- allowed.
    assert flag_bad_quotes(chain)["time_value"].tolist() == [True, False, False]


def test_crossed_market():
    chain = _chain([_row(700, 2.0, 1.5), _row(725, 1.5, 2.0)])
    assert flag_bad_quotes(chain)["crossed"].tolist() == [True, False]


def test_checks_select_columns_and_reject_unknown():
    chain = _chain([_row(700, 2.0, 1.5)])
    assert list(flag_bad_quotes(chain, checks=("crossed",)).columns) == ["crossed"]
    assert len(clean_options_chain(chain, checks=("roots",))) == 1
    with pytest.raises(ValueError):
        flag_bad_quotes(chain, checks=("nope",))


def test_clean_keeps_index_and_order():
    chain = _chain([_row(725, 1.5, 2.0), _row(700, 2.0, 1.5), _row(750, 2.5, 3.0)])
    chain.index = [10, 20, 30]
    assert clean_options_chain(chain).index.tolist() == [10, 30]
