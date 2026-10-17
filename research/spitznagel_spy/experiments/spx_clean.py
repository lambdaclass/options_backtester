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
bids mark correctly.
"""

import pandas as pd

BINARY_ROOTS = ("BSK", "BSF")
FLAT_MIN_RUN = 3          # identical quotes on >= this many consecutive strikes
FLAT_MIN_SPAN = 0.03      # ... spanning >= 3% of spot
FLAT_MIN_ASK = 1.0        # penny plateaus are normal
MAX_TIME_VALUE = 0.25     # of spot


def clean_spx_chain(opt: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return `opt` without binary, flat-printed or impossible-priced rows."""
    n0 = len(opt)
    binary = opt["optionroot"].astype(str).str.startswith(BINARY_ROOTS)

    o = opt.sort_values(["quotedate", "expiration", "type", "strike"], kind="stable")
    new_group = ((o["quotedate"] != o["quotedate"].shift())
                 | (o["expiration"] != o["expiration"].shift())
                 | (o["type"] != o["type"].shift()))
    same = ~new_group & (o["bid"] == o["bid"].shift()) & (o["ask"] == o["ask"].shift())
    run = (~same).cumsum()
    g = o.groupby(run)["strike"]
    span = (g.transform("max") - g.transform("min")) / o["underlying_last"]
    flat = ((g.transform("size") >= FLAT_MIN_RUN) & (span >= FLAT_MIN_SPAN)
            & (o["bid"] > 0) & (o["ask"] >= FLAT_MIN_ASK))
    flat = flat.reindex(opt.index)

    is_put = opt["type"].astype(str).str.lower().str[0] == "p"
    intrinsic = (opt["strike"] - opt["underlying_last"]).where(
        is_put, opt["underlying_last"] - opt["strike"]).clip(lower=0)
    absurd = (opt["bid"] - intrinsic) > MAX_TIME_VALUE * opt["underlying_last"]

    drop = binary | flat | absurd
    if verbose:
        print(f"clean_spx_chain: dropped {drop.sum():,}/{n0:,} rows "
              f"(binary {binary.sum():,}, flat {flat.sum():,}, absurd {absurd.sum():,})")
    return opt[~drop]
