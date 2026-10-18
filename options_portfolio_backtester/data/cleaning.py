"""Find and drop option-chain rows that are not tradeable quotes.

Vendor chains carry rows that look like ordinary options but are not, and an
entry filter (an OTM band, a DTE window) cannot tell them apart. Each hazard
below produced a wrong backtest result on real data before it was caught:

``roots``
    Other products listed under the same underlying. CBOE's SPX binary
    options (roots BSK, BSF; 2008-2010) sit in the SPX chain with normal
    strikes and expirations but are priced as a probability, a few dimes; a
    deep-OTM put strategy bought them as $0.19 "puts" in Oct 2008.
``flat``
    One identical bid/ask printed across a wide run of strikes of one
    expiry (e.g. a whole Jun-03 expiry quoted 64.30/75.10 for strikes
    650-725). Real quotes change with the strike; a plateau longer than a
    couple of strikes, above a penny, is a stale or filled-in print.
``time_value``
    A bid more than a fraction of spot above intrinsic value -- impossible
    for a listed equity-index option.
``crossed``
    Bid above ask.

Use :func:`flag_bad_quotes` to inspect, :func:`clean_options_chain` to drop.
Placeholder asks (e.g. 99999) are left alone: an engine never buys at them,
and their bids mark correctly.
"""

from __future__ import annotations

import pandas as pd

CHECKS = ("roots", "flat", "time_value", "crossed")


def flag_bad_quotes(
    chain: pd.DataFrame,
    *,
    drop_roots: tuple[str, ...] = (),
    flat_min_run: int = 3,
    flat_min_span: float = 0.03,
    flat_min_ask: float = 1.0,
    max_time_value: float = 0.25,
    checks: tuple[str, ...] = CHECKS,
) -> pd.DataFrame:
    """One boolean column per check in ``checks`` (True = bad row), aligned
    with ``chain``'s index.

    ``drop_roots``: ``optionroot`` prefixes to drop (``roots`` check).
    ``flat_*``: a run of >= ``flat_min_run`` consecutive strikes with an
    identical bid and ask, spanning >= ``flat_min_span`` of spot, with ask >=
    ``flat_min_ask``. ``max_time_value``: largest plausible bid above
    intrinsic, as a fraction of spot.

    Expects the default schema's column names (``quotedate``, ``expiration``,
    ``type``, ``strike``, ``bid``, ``ask``, ``underlying_last``,
    ``optionroot``).
    """
    unknown = set(checks) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown checks {sorted(unknown)}; choose from {CHECKS}")
    flags = pd.DataFrame(index=chain.index)

    if "roots" in checks:
        flags["roots"] = (chain["optionroot"].astype(str).str.startswith(tuple(drop_roots))
                          if drop_roots else False)

    if "flat" in checks:
        o = chain.sort_values(["quotedate", "expiration", "type", "strike"], kind="stable")
        new_group = ((o["quotedate"] != o["quotedate"].shift())
                     | (o["expiration"] != o["expiration"].shift())
                     | (o["type"] != o["type"].shift()))
        same = ~new_group & (o["bid"] == o["bid"].shift()) & (o["ask"] == o["ask"].shift())
        run = (~same).cumsum()
        g = o.groupby(run)["strike"]
        span = (g.transform("max") - g.transform("min")) / o["underlying_last"]
        flat = ((g.transform("size") >= flat_min_run) & (span >= flat_min_span)
                & (o["bid"] > 0) & (o["ask"] >= flat_min_ask))
        flags["flat"] = flat.reindex(chain.index)

    if "time_value" in checks:
        is_put = chain["type"].astype(str).str.lower().str[0] == "p"
        intrinsic = (chain["strike"] - chain["underlying_last"]).where(
            is_put, chain["underlying_last"] - chain["strike"]).clip(lower=0)
        flags["time_value"] = (chain["bid"] - intrinsic) > max_time_value * chain["underlying_last"]

    if "crossed" in checks:
        flags["crossed"] = chain["bid"] > chain["ask"]

    return flags.astype(bool)


def clean_options_chain(chain: pd.DataFrame, *, verbose: bool = False, **kwargs) -> pd.DataFrame:
    """``chain`` without the rows :func:`flag_bad_quotes` flags (same keyword
    arguments). With ``verbose``, prints how many rows each check dropped."""
    flags = flag_bad_quotes(chain, **kwargs)
    drop = flags.any(axis=1)
    if verbose:
        detail = ", ".join(f"{c} {flags[c].sum():,}" for c in flags.columns)
        print(f"clean_options_chain: dropped {drop.sum():,}/{len(chain):,} rows ({detail})")
    return chain[~drop]
