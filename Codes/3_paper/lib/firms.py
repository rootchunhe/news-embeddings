"""The firms of the December windows (Table 4 and the firm networks, Table 5): every firm's December outcomes and characteristics, the windows of
news up to each December, and the firm-level inputs of a window's two-way projection."""
import numpy as np
import pandas as pd

from lib import ROOT
from lib.constants import POST_YEAR, PROJECTION_YEARS, SPLIT_YEAR
from lib.ridge import category_index, firm_years, rolling_windows

from common.config import FIRST_YEAR
from common.jkp import monthly_panel

OUTCOMES = ["log_pb", "log_pe", "ltg"]
CHARS = ["beta_60m", "log_be", "at_gr1", "gp_at", "div_at"]
FF49 = range(1, 50)


def winsorize(D, cols, ref, lower=True):
    """D[cols] clipped at the 1st and 99th percentiles of the rows ref (the fitting firms)"""
    q = D.loc[ref, cols].quantile([0.01, 0.99])
    return D[cols].clip(q.loc[0.01] if lower else None, q.loc[0.99], axis=1)


def ltg():
    """the IBES median long-term growth forecast of the last December statistics period (as Sarkar 2026; Bordalo et al. 2023), linked to
    permno by the IBES-CRSP link valid on that date (best score)"""
    T = pd.read_parquet(ROOT / "Data/raw/ibes/ltg.parquet")
    L = pd.read_parquet(ROOT / "Data/raw/ibes/link.parquet")
    T["statpers"] = pd.to_datetime(T.statpers)
    T = T[T.statpers.dt.month == 12].drop_duplicates(["ticker", "statpers"], keep=False)
    T = T[T.statpers == T.groupby(["ticker", T.statpers.dt.year]).statpers.transform("max")]
    M = T.merge(L, on="ticker")
    M = M[(M.statpers >= pd.to_datetime(M.sdate)) & (M.statpers <= pd.to_datetime(M.edate))]
    M = M[M.score == M.groupby(["ticker", "statpers"]).score.transform("min")].drop_duplicates(["ticker", "statpers", "permno"])
    M = M.assign(permno=M.permno.astype("int64"), year=M.statpers.dt.year, ltg=M.medest.astype(float))
    M = M[M.groupby(["ticker", "statpers"]).permno.transform("size") == 1]           # a ticker linked to two permnos: dropped
    M = M[M.score == M.groupby(["permno", "year"]).score.transform("min")]
    return M.drop_duplicates(["permno", "year"], keep=False)[["permno", "year", "ltg"]]   # two tickers on one permno, same score: dropped


def decembers(columns=()):
    """every firm at every December: permno, permco, year, ff49, the outcomes and the characteristics (plus the JKP `columns`, kept as
    they are)"""
    base = ["permno", "permco", "eom", "ff49", "be_me", "ni_me", "me_company", "beta_60m", "at_gr1", "gp_at", "div_at"]
    J = monthly_panel(columns=base + [c for c in columns if c not in base])
    J = J[J.month.str.endswith("-12")].assign(year=lambda d: d.month.str[:4].astype(int))
    pos = lambda x: x.where(x > 0).astype(float)
    J["log_pb"], J["log_pe"] = -np.log(pos(J.be_me)), -np.log(pos(J.ni_me))
    J["log_be"] = np.log(pos(J.be_me * J.me_company))                             # JKP ratios are over the company's market equity
    J = J.merge(ltg(), on=["permno", "year"], how="left")
    return J.dropna(subset=CHARS + ["ff49", "permco"])[["permno", "permco", "year", "ff49"] + OUTCOMES + CHARS
                                                        + [c for c in columns if c not in CHARS]]


def december_windows(model, yrs, firsts=None):
    """(first year, year T, summed statistics, firm statistics) of the window of the December of every year T of yrs: the news of the
    PROJECTION_YEARS years up to T, from each first year of `firsts` (default: FIRST_YEAR, the year after the model's pretraining cutoff
    (Table C6) and SPLIT_YEAR (Table 10)); firm statistics (lib.ridge.rolling_windows) of the same years. From FIRST_YEAR the
    window of December T is lib.ridge.firm_years(T + 1)"""
    cats = category_index()
    for first in sorted(firsts or {FIRST_YEAR, SPLIT_YEAR, POST_YEAR.get(model, FIRST_YEAR)}):
        span = lambda y, first=first: list(range(max(first, y - PROJECTION_YEARS + 1), y + 1))
        assert first != FIRST_YEAR or all(span(y) == firm_years(y + 1) for y in yrs), \
            "the December window of y is not the projection window of test year y + 1 (lib.constants.PROJECTION_YEARS)"
        for y, _, st, fst in rolling_windows(model, cats, [y for y in yrs if y >= first], firm=True, span=span):
            yield first, y, st, fst


def firm_inputs(f, fst, i):
    """the firm-level inputs of the firms at rows i of the window fit f (a WindowFit with firm statistics fst): F (the firm effect), the
    category mix (the firm's mean of the category component alpha_c - mu over the stories behind fst) and the average (the firm's mean of
    E - mu); firm effects are compared across firms, so every firm must lie in the window's largest connected component"""
    assert f.largest[i].all(), "a firm outside the largest connected component of the window: its F is not comparable"
    n = f.nF[i, None]
    return {"F": f.gamma[i], "category mix": fst[2][i] @ f.alpha / n - f.mu, "average": fst[1][i] / n - f.mu}
