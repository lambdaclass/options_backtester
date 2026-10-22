"""Drop SPX chain rows that are not tradeable SPX option quotes.

Found by tracing an impossible result (a hedged run with a WORSE GFC
drawdown than buy & hold) back to the trades:

1. Binary options. CBOE's SPX binaries (roots BSK, BSF; 2008-10 to 2010-04)
   sit in the same chain with ordinary strikes and expirations but are priced
   as a probability (a few dimes). An OTM-band filter cannot tell them apart:
   one run bought 361 BSK "puts" at $0.19 in Oct 2008 and marked them like
   real puts.
2. Flat prints. A whole expiry quoted at one identical price across a wide
   strike range (e.g. 2003-01-02 Jun-03 puts at 64.30/75.10 for strikes
   650-725 with SPX at 909; Dec-04 LEAPS at ~$94 for strikes 500-850).
   Real plateaus only span a couple of adjacent strikes.
3. Impossible time value: a put bid more than 25% of spot above intrinsic
   (all 2001-2004).

Placeholder asks (99999) are kept: the engine never buys at them and their
bids mark correctly. The checks now live in the library
(options_portfolio_backtester.data.cleaning); this keeps the study's settings.
"""

import pandas as pd

from options_portfolio_backtester import clean_options_chain, normalize_expirations

BINARY_ROOTS = ("BSK", "BSF")


def clean_spx_chain(opt: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return `opt` without binary, flat-printed or impossible-priced rows
    (the library cleaner with this study's settings; crossed markets are left
    in -- the SPX chain has none), with Saturday expirations (SPX monthlies
    until 2015) moved to the Friday before, so DTE rules mean the same
    trading day across the whole history."""
    cleaned = clean_options_chain(opt, verbose=verbose, drop_roots=BINARY_ROOTS,
                                  checks=("roots", "flat", "time_value"))
    return normalize_expirations(cleaned)
