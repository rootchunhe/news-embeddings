"""The decompositions of the replication forecast yhat, with the ridge coefficients (training years T-8..T-3) and the two-way
category/firm projection (the stories of the six years before T, lib.ridge.firm_years) of each test year T frozen and
applied to the stories of T and of T - 1 (the history the year needs; the stories of T - 1 are in the projection's sample):
  content   yhat - ybar = yC + yF + yU (event category, firm identity, idiosyncratic); yC is the category effect centered at the
            projection headlines' mean plus the offset (projection mean - training mean)' beta, one number per T;
  timing    yC = cL + cD and yU = uL + uD, with cL / uL the mean of the firm's monthly (trade-month) averages over its news months
            among the previous LEVEL_WINDOW months (0 without such a month: no history, all deviation); cL is the level of yC
            without the offset, so the offset is in cD for every headline; yF is constant within T, so it is all level. Every story of a month shares the level known at the start of the month;
  appendix  yU = yH + yUH with the firm's own HISTORY_DAYS-day mean.
    OMP_NUM_THREADS=8 ../../env/bin/python 3_decompose.py --model roberta-large      (Llama: on a compute node)
Output Results/paper/3_decompose/<model>/decomposition.csv (also on each segment of lib.constants.SEGMENTS) + run.json; the story-level parts in
Data/clean/decomposition_<model>.parquet"""
import numpy as np
import pandas as pd

from lib import ROOT
from lib.constants import CELLS, HISTORY_DAYS, LABELS, PAPER, PARTS, PERIODS, SEGMENT_PERIOD, SEGMENTS, TEST_YEARS, TIMING, TS_DAILY, REBUILD_TOL

from common.config import CHUNK
from common.runinfo import write_manifest
from lib.run import finish, save, start
from lib.evaluate import component_table
from lib.panel import firm_day, forecasts
from lib.monthly import trailing_level
from lib.ridge import WindowFit, category_index, firm_years, load_year, replication_fit, rolling_windows


KEYS = ["rp_story_id", "permno", "day0", "ts_ny", "day1_yr", "y", "y_fit", "cap", "small", "yr", "category", "group"]


def prior_sums(daily):
    """for every (permno, day) row of `daily` (columns permno, day0, s, n): the sums of s and n over the rows of the same firm in
    the prior HISTORY_DAYS calendar days (strictly before the day)"""
    kc = pd.factorize(daily.permno, sort=True)[0].astype("int64")
    day = daily.day0.values.astype("datetime64[D]").astype("int64")
    o = np.lexsort((day, kc))
    daily, kc, day = daily.iloc[o].reset_index(drop=True), kc[o], day[o]
    pos = kc * (1 << 20) + day                                             # one increasing integer per (firm, day)
    s, n = daily.s.values.astype(float), daily.n.values.astype(float)
    S, N = np.zeros(len(daily)), np.zeros(len(daily))
    for k in range(1, HISTORY_DAYS + 1):                                       # one row per (firm, day): the k-day lag is a lookup
        j = np.minimum(np.searchsorted(pos, pos - k), len(pos) - 1)
        hit = pos[j] == pos - k
        S[hit] += s[j[hit]]
        N[hit] += n[j[hit]]
    return daily.assign(ds=S, dn=N)


def own_mean(S, col):
    """the mean of the firm's own col over its stories of the prior HISTORY_DAYS calendar days (strictly before day0); 0 without any"""
    firm = prior_sums(S.groupby(["permno", "day0"]).agg(s=(col, "sum"), n=(col, "size")).reset_index())
    firm["h"] = np.where(firm.dn > 0, firm.ds / firm.dn.replace(0, np.nan), 0.0)
    return S[["permno", "day0"]].merge(firm[["permno", "day0", "h"]], on=["permno", "day0"], how="left").h.values


def parts(f, beta, X, c):
    """the content split of every story of (X, c) under the window fit f and the coefficients beta: yC, yF, yU and the forecast;
    yCp: the two-way category effect centered at the projection mean (0 for unseen categories), yC = yCp + (f.mu_proj - f.mu)' beta"""
    fi, seen = f.firm_index(c.permno.values)
    yCp = ((f.alpha - f.mu_proj) @ beta)[c.code.values]
    yC = yCp + (f.mu_proj - f.mu) @ beta
    yF = np.where(seen, (f.gamma @ beta)[fi], 0.0)
    yfull = np.concatenate([f.z(X[i:i + CHUNK]) @ beta for i in range(0, len(c), CHUNK)])
    return pd.DataFrame({"yCp": yCp, "yC": yC, "yF": yF, "yU": yfull - yC - yF, "yfull": yfull})


def with_level(H, cols):
    """for every row of H (permno, day0, cols): the mean of the firm's trade-month averages of cols over its news months among the
    previous LEVEL_WINDOW months (0 without such a month) and the flag hasL"""
    H = H.assign(month=H.day0.dt.to_period("M"))
    M = H.groupby(["permno", "month"])[cols].mean().reset_index()
    L, has = trailing_level(M[cols].values.astype(np.float64), M)
    M = M[["permno", "month"]].assign(hasL=has, **{f"L_{k}": L[:, j] for j, k in enumerate(cols)})
    return H.merge(M, on=["permno", "month"], how="left").drop(columns="month")


def split(f, beta, X, c, Xp, cp):
    """for the headlines of test year T (X, c), in the order of c: the content split (parts) and the levels cL, uL with the flag hasL
    (0 without history) and the own HISTORY_DAYS-day mean yH of yU, the history (Xp, cp) of T - 1 also under the fit f. cL is the
    level of yCp: the offset of yC stays out of the level"""
    cur, hist = parts(f, beta, X, c), parts(f, beta, Xp, cp)
    cols = ["yCp", "yU"]
    rows = lambda k, p, now: k[["permno", "day0"]].reset_index(drop=True).assign(_cur=now, **{j: p[j].values for j in cols})
    both = with_level(pd.concat([rows(cp, hist, False), rows(c, cur, True)], ignore_index=True), cols)
    both = both.assign(yH=own_mean(both, "yU"))                       # appendix: the own 20-day mean
    lv = both[both._cur].reset_index(drop=True)
    return cur.drop(columns="yCp").assign(hasL=lv.hasL.values, cL=lv.L_yCp.values, uL=lv.L_yU.values, yH=lv.yH.values)


def main():
    a, log, out = start("3_decompose")
    _, sd_T, fits = replication_fit(a.model)
    cats = category_index()
    rows, prev, diag = [], None, {}
    for T, yrs_tr, st, fst in rolling_windows(a.model, cats, TEST_YEARS, firm=True):
        f = WindowFit(st, fst)
        beta = f.replication_beta(fits[T])
        X, c = load_year(T, a.model, cats, need_return=False)               # every sample story enters the levels
        diag[str(T)] = f.diagnostics(fst, c.permno.values)
        Xp, cp = prev if prev is not None else load_year(T - 1, a.model, cats, need_return=False)
        d = pd.concat([c[KEYS].reset_index(drop=True), split(f, beta, X, c, Xp, cp)], axis=1)
        d["ybar"], d["yhat"] = f.ybar, f.ybar + d.yfull
        d = d[c.y.notna().values].reset_index(drop=True)                    # the decomposition itself: the stories with a next-day return
        d["day1_yr"] = d.day1_yr.astype("int64")
        off, yrs_p = float((f.mu_proj - f.mu) @ beta), firm_years(T)
        log(f"{T}: train {yrs_tr[0]}-{yrs_tr[-1]}, projection {yrs_p[0]}-{yrs_p[-1]}, firms {len(f.permno):,}, level for {100 * d.hasL.mean():.0f}% "
            f"of stories | var shares C {d.yC.var() / d.yfull.var():.2f} F {d.yF.var() / d.yfull.var():.2f} U {d.yU.var() / d.yfull.var():.2f} | "
            f"offset of yC {off:.2e} = {off / d.yC.std():.2f} sd of yC")
        rows.append(d.drop(columns="yfull"))
        prev = (X, c)
    S = forecasts(pd.concat(rows, ignore_index=True), {"yhat_rep": a.model})
    assert S.yhat_rep.notna().all(), "stories without a replication forecast"
    err = ((S.yhat - S.yhat_rep).abs() / S.yr.map(sd_T)).groupby(S.yr).max()
    log("max |rebuilt - replication| in forecast sd by year: " + " ".join(f"{T} {e:.1e}" for T, e in err.items()))
    assert err.max() < REBUILD_TOL, f"the projections do not rebuild the replication forecast (max error {err.max():.1e} forecast sd)"
    S = S.drop(columns="yhat_rep").sort_values(["permno", "day0"]).reset_index(drop=True)
    S["cD"], S["uD"], S["yUH"] = S.yC - S.cL, S.yU - S.uL, S.yU - S.yH
    for k, v in TIMING.items():
        S[k] = S[v].sum(axis=1)
    gap = (S[CELLS].sum(axis=1) - (S.yhat - S.ybar)).abs().max()
    assert gap < 1e-6 * (S.yhat - S.ybar).abs().max(), f"the cells do not add up to the forecast (max gap {gap:.1e})"
    facts = dict(stories=len(S), max_rebuild_error=float(err.max()), level_share=float(S.hasL.mean()), two_way=diag)
    fn = ROOT / f"Data/clean/decomposition_{a.model}.parquet"
    S.to_parquet(fn.with_suffix(".tmp.parquet"), index=False)
    fn.with_suffix(".tmp.parquet").replace(fn)
    write_manifest(fn.with_name(f"run_decomposition_{a.model}.json"), vars(a), spec=PAPER, rows=len(S), **facts)

    sigs = PARTS + TS_DAILY + ["yhat"]
    F = firm_day(S, sigs)
    periods = PERIODS | {SEGMENT_PERIOD[a.model][s]: v for s, v in SEGMENTS[a.model].items()}   # Tables C2, B2: the segments
    save(out, "decomposition", component_table(F, sigs, LABELS, periods), "content x timing decomposition of the forecast (firm-days)")
    finish(out, a, log, **facts)


if __name__ == "__main__":
    main()
