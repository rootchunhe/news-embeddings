"""CRSP daily panel from Data/raw/crsp: the one place where open-to-open returns are defined.
daily_panel() returns US common stocks on NYSE/AMEX/NASDAQ with absolute prices, numeric columns, one row per (permno, day),
the trading-day index `pos` and the trading calendar; o2o(df, k) is the split- and dividend-adjusted open-to-open return from
day pos+k-1 to pos+k, NaN unless the rows are exactly k trading days apart and both open prints are within a factor 2 of the
close; open_to_close(df) is the same-day open-to-close return."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_COLS = ["permno", "cusip", "dlycaldt", "dlyret", "dlyopen", "dlyclose", "dlycap", "primaryexch"]


def daily_panel(columns=BASE_COLS):
    files = sorted((ROOT / "Data/raw/crsp").glob("dsf_*.parquet"))
    cr = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    cr["dlycaldt"] = pd.to_datetime(cr.dlycaldt)
    cr["permno"] = cr.permno.astype(int)
    for c in ["dlyret", "dlyopen", "dlyclose", "dlycap"]:
        if c in cr:
            cr[c] = pd.to_numeric(cr[c], errors="coerce").astype(float)
    for c in ["dlyopen", "dlyclose"]:
        if c in cr:
            cr[c] = cr[c].abs()
    cr = cr.dropna(subset=["dlyret"]).drop_duplicates()                       # rows repeated across the yearly files, identical
    assert not cr.duplicated(["permno", "dlycaldt"]).any(), "two different CRSP rows for one permno-day"
    cr = cr.sort_values(["permno", "dlycaldt"]).reset_index(drop=True)
    tdays = np.unique(cr.dlycaldt.values.astype("datetime64[ns]"))
    cr["pos"] = np.searchsorted(tdays, cr.dlycaldt.values.astype("datetime64[ns]"))
    return cr, tdays


def open_ok(o, c):
    return (o / c).between(0.5, 2.0)


def o2o(cr, k, g):
    """open(pos+k-1) -> open(pos+k) return of each row (g: cr grouped by permno), adjusted with dlyret of day pos+k and the closes around it"""
    sh = lambda col, j: g[col].shift(-j) if j else cr[col]
    ok = (sh("pos", k - 1) == cr.pos + k - 1) & (sh("pos", k) == cr.pos + k)
    o0, o1, c0, c1, r1 = sh("dlyopen", k - 1), sh("dlyopen", k), sh("dlyclose", k - 1), sh("dlyclose", k), sh("dlyret", k)
    return (o1 / o0 * (1 + r1) * c0 / c1 - 1).where(ok & open_ok(o0, c0) & open_ok(o1, c1))


def open_to_close(cr):
    return (cr.dlyclose / cr.dlyopen - 1).where(open_ok(cr.dlyopen, cr.dlyclose))
