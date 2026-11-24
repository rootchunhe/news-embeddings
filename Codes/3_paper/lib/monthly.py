"""Firm-month helpers: the historical level, the JKP month-ahead return and the characteristic list of the Didisheim replication."""
import numpy as np
import pandas as pd

from lib import DIDISHEIM_RESULTS
from lib.constants import TEST_YEARS

from common.config import LEVEL_WINDOW, RET_CLIP
from common.jkp import month_ranks, monthly_panel
from common.runinfo import require_same_sample


def trailing_level(Z, M, lag=1):
    """the firm's mean of Z over its rows (months with news, equally weighted) of the LEVEL_WINDOW calendar months ending `lag` months
    before the row (M: permno, month as 'YYYY-MM' or Period[M]), 0 for a row without such months; returns (L, has)"""
    p = pd.PeriodIndex(M.month, freq="M")
    permno, ordinal = M.permno.values, (p.year * 12 + p.month).values
    key = pd.Series(np.arange(len(Z)), index=pd.MultiIndex.from_arrays([permno, ordinal]))
    L, cnt = np.zeros(Z.shape, np.result_type(Z.dtype, np.float32)), np.zeros(len(Z))       # float32 input stays float32 (Llama: 5,120 columns)
    for k in range(lag, lag + LEVEL_WINDOW):
        src = key.reindex(list(zip(permno, ordinal - k))).values
        ok = ~np.isnan(src)
        L[ok] += Z[src[ok].astype(int)]
        cnt[ok] += 1
    has = cnt > 0
    L[has] /= cnt[has, None]
    return L, has


def jkp_returns():
    """JKP month-ahead excess return: permno, month (Period[M], the signal month), ret (not winsorized), ret_fit (ret winsorized at
    RET_CLIP: the monthly ridge's training target), cap (market equity of the signal month)"""
    J = monthly_panel(columns=["permno", "eom", "ret_exc_lead1m", "market_equity"], end=f"{TEST_YEARS[-1]}-12")   # the same end as the replication
    J = J[J.ret_exc_lead1m.notna()].copy()
    J["permno"] = J.permno.astype("int64")
    J["month"] = pd.PeriodIndex(J.month, freq="M")
    J["ret"] = pd.to_numeric(J.ret_exc_lead1m, errors="coerce").astype("float64")
    J["ret_fit"] = J.ret.clip(*RET_CLIP)
    J["cap"] = pd.to_numeric(J.market_equity, errors="coerce").astype("float64")       # for the value-weighted sorts
    return J[["permno", "month", "ret", "ret_fit", "cap"]]


SIZE_OF_JKP = {"nano": "micro", "micro": "micro", "small": "small", "large": "big", "mega": "big"}   # JKP size_grp -> lib.panel.SIZE_GROUPS


def jkp_size_groups():
    """the Fama and French (2008) size group of every JKP firm-month from JKP's size_grp (NYSE breakpoints at the month end): micro =
    nano + micro (below the 20th percentile), small (20th percentile to the median), big = large + mega; permno, month (Period[M]), size"""
    J = monthly_panel(columns=["permno", "eom", "size_grp"], end=f"{TEST_YEARS[-1]}-12")
    J = J[J.size_grp.notna()].copy()
    unknown = set(J.size_grp.unique()) - set(SIZE_OF_JKP)
    assert not unknown, f"JKP size groups without a mapping: {unknown}"
    J["permno"] = J.permno.astype("int64")
    J["month"] = pd.PeriodIndex(J.month, freq="M")
    J["size"] = J.size_grp.map(SIZE_OF_JKP)
    assert not J.duplicated(["permno", "month"]).any(), "JKP firm-months appear twice"
    return J[["permno", "month", "size"]]


def jkp_ranks(chars):
    """permno, month (Period[M]) and the characteristics ranked across each month's JKP stocks (common.jkp.month_ranks)"""
    K = monthly_panel(columns=["permno", "eom"] + list(chars))
    K[chars] = month_ranks(K, chars)
    return K.assign(month=pd.PeriodIndex(K.month, freq="M"))[["permno", "month"] + list(chars)]


def replication_characteristics(model):
    """the JKP characteristics the Didisheim replication on the extended sample used (from its manifest)"""
    return require_same_sample(DIDISHEIM_RESULTS / model, f"Didisheim replication of {model}", needs=("monthly",))["characteristics"]
