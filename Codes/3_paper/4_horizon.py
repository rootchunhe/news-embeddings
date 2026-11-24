"""Section 4, daily: the contribution of each part of the replication forecast (content x timing cells of 3_decompose) to its
within-day covariance with the single-day return at horizons 0..H (0: the immediate reaction before the trade, from the last close
or, for news during the session, from the day's open; h >= 1: the h-th open-to-open day), on the firm-days with all of them, with Newey-West t, the share of the total and the IC. The forecast is fixed at the story; only the horizon of the
return changes. Appendix: the level and the deviation residualized on the replication's JKP characteristics with the previous month's projection,
joint Fama-MacBeth by h.
Run from Codes/3_paper/ after 3_decompose.py:
    ../../env/bin/python 4_horizon.py --model roberta-large
Output Results/paper/4_horizon/<model>/{term_structure,jkp_fm}.csv, ic_daily.csv.gz (the daily rank ICs of the level and the
deviation by h), ic_daily_groups.csv.gz (their next-day rank ICs within each size group and news-intensity tercile, Table 11),
cov_daily.csv.gz (the daily covariances of the forecast and of every part with the return, by h) + run.json"""
import numpy as np
import pandas as pd

from lib.constants import CELLS, COVERAGE_GROUPS, H, LEVEL_WINDOW, NW_LAGS_DAILY, PARTS, SEGMENTS, TEST_YEARS, TIMING, TS_DAILY

from lib.evaluate import contribution_table, fama_macbeth, ic_series, predictive_cov
from lib.monthly import jkp_ranks, replication_characteristics
from lib.panel import decomposition, firm_day, sample_rows, segment, size_group
from lib.run import finish, save, start
from lib.timing import immediate_reaction

from common.portfolio import nw_tstat

SIGS = PARTS + TS_DAILY


def load(model):
    """story rows: the parts of the decomposition with the immediate reaction and the daily open-to-open returns"""
    D = decomposition(model, ["rp_story_id", "permno", "day0", "y", "cap", "small", "yr", "yhat"] + SIGS)
    o2o = [f"o2o_{h}" for h in range(1, H + 1)]
    R = sample_rows(TEST_YEARS, columns=["gap0"] + o2o)[["rp_story_id", "permno", "pub_day", "ts_ny", "gap0"] + o2o]
    S = immediate_reaction(D.merge(R, on=["rp_story_id", "permno"], how="left")).rename(columns={"r_imm": "r_0"})
    for h in range(1, H + 1):
        S[f"r_{h}"] = pd.to_numeric(S[f"o2o_{h}"], errors="coerce").astype(float)
    return S


def jkp_fm(F, fit, model):
    """the level and the deviation of the firm-days F residualized on the replication's JKP characteristics of the previous month-end
    (ranked across that month-end's JKP universe), with the projection fitted on the previous calendar month's firm-days of `fit`
    (every firm-day with a signal, whatever its later returns), so that every residual uses information known before its month (the
    first month has no projection and is left out); then their joint Fama-MacBeth on the return of every horizon h >= 1 (bp per sd)"""
    chars = replication_characteristics(model)
    K = jkp_ranks(chars)
    K["month"] += 1                                                         # known in the next month
    with_chars = lambda d: d.assign(month=d.day0.dt.to_period("M")).merge(K, on=["permno", "month"], how="inner")
    design = lambda d: np.column_stack([np.ones(len(d)), d[chars].values])
    G = with_chars(fit)
    X, Y = design(G), G[list(TIMING)].values.astype(np.float64)
    months = G.groupby("month").indices
    assert list(months) == list(pd.period_range(min(months), max(months), freq="M")), "the firm-day months are not contiguous"
    beta = {}
    for m, idx in months.items():
        assert len(idx) >= 2 * X.shape[1], f"{m}: {len(idx)} firm-days for {X.shape[1]} regressors"
        beta[m] = np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0]
    D = with_chars(F)
    X, Y = design(D), D[list(TIMING)].values.astype(np.float64)
    for m, idx in D.groupby("month").indices.items():
        Y[idx] = Y[idx] - X[idx] @ beta[m - 1] if m - 1 in beta else np.nan
    D[list(TIMING)] = Y
    D = D.dropna(subset=list(TIMING))
    rows = []
    for h in range(1, H + 1):
        t, _ = fama_macbeth(D, f"r_{h}", list(TIMING))
        rows += [{"h": h, "x": r.x, "coef_bp_per_sd": r.coef, "t": r.t, "days": r.periods} for r in t.itertuples()]
    return pd.DataFrame(rows)


def term_structure(F, log):
    """every part's contribution to the forecast's within-day covariance with the return h days after the trade, and its IC; and
    the daily ICs of the level and the deviation (day0, h, lev, dev) and the daily covariances of the forecast and every part (day0, h,
    yhat, parts)"""
    rows, daily, cov = [], [], []
    for h in range(0, H + 1):
        C = predictive_cov(F, SIGS + ["yhat"], ycol=f"r_{h}")
        T = contribution_table(C, SIGS, "yhat", NW_LAGS_DAILY, scale=1e6)
        ic = {k: ic_series(F, k, ycol=f"r_{h}") for k in T.part}
        T["IC_%"], T["t_IC"] = [100 * ic[k].mean() for k in T.part], [nw_tstat(ic[k].values, NW_LAGS_DAILY) for k in T.part]
        rows.append(T.assign(h=h))
        daily.append(pd.DataFrame({k: ic[k] for k in TIMING}).rename_axis("day0").reset_index().assign(h=h))
        cov.append(C[["yhat", *SIGS]].rename_axis("day0").reset_index().assign(h=h))
        log(f"h={h}: " + " ".join(f"{r.part} {r.share:.2f}" for r in T.itertuples() if r.part in CELLS))
    return (pd.concat(rows, ignore_index=True), pd.concat(daily, ignore_index=True)[["day0", "h", *TIMING]],
            pd.concat(cov, ignore_index=True)[["day0", "h", "yhat", *SIGS]])


def coverage_group(F):
    """the news-intensity tercile of each firm-day among the day's firm-days: the firm's sample stories (with or without a next-day
    return) in the LEVEL_WINDOW calendar months before the firm-day's month; tied firms share a tercile"""
    C = sample_rows(range(TEST_YEARS[0] - 1, TEST_YEARS[-1] + 1), need_return=False)[["permno", "day0"]]
    mi = lambda d: d.dt.year * 12 + d.dt.month
    n = C.groupby(["permno", mi(C.day0).rename("mi")]).size().unstack(fill_value=0)
    n = n.reindex(columns=range(n.columns.min(), n.columns.max() + 1), fill_value=0)
    past = n.T.rolling(LEVEL_WINDOW, min_periods=1).sum().shift(1).fillna(0).T.stack().rename("stories")   # months mi-LEVEL_WINDOW .. mi-1
    k = pd.MultiIndex.from_arrays([F.permno.values, mi(F.day0).values])
    r = pd.Series(past.reindex(k).fillna(0).values, index=F.index).groupby(F.day0.values).rank(pct=True)
    return np.select([r <= 1 / 3, r <= 2 / 3], list(COVERAGE_GROUPS[:2]), COVERAGE_GROUPS[2])


def group_ics(F):
    """Table 11: the next-day rank ICs of the level and the deviation within each Fama-French size group and news-intensity tercile
    (day0, group, lev, dev)"""
    out = []
    for g in (size_group(F.day0, F.cap), coverage_group(F)):
        for name in pd.unique(g):
            d = F[g == name]
            out.append(pd.DataFrame({k: ic_series(d, k, ycol="r_1") for k in TIMING}).rename_axis("day0").reset_index().assign(group=name))
    return pd.concat(out, ignore_index=True)[["day0", "group", *TIMING]]


def main():
    a, log, out = start("4_horizon")
    S = load(a.model)
    F0 = firm_day(S, SIGS + ["yhat"], extra={f"r_{h}": (f"r_{h}", "mean") for h in range(0, H + 1)})  # stories of a firm-day: signals and windows averaged
    F = F0.dropna(subset=[f"r_{h}" for h in range(0, H + 1)])                # one sample for every horizon: the firm-days with the whole path
    log(f"{len(S):,} stories, {len(F):,} firm-days")
    T, I, V = term_structure(F, log)
    save(out, "term_structure", T, "contribution to the within-day covariance (x 1e6), t, share, IC")
    I.to_csv(out / "ic_daily.csv.gz", index=False)
    V.to_csv(out / "cov_daily.csv.gz", index=False)
    group_ics(F).to_csv(out / "ic_daily_groups.csv.gz", index=False)
    for s, (lo, hi) in SEGMENTS[a.model].items():
        log(f"signal years {lo}-{hi}:")
        save(out, f"term_structure_{s}", term_structure(segment(F, lo, hi), log)[0], f"the same on the signal years {lo}-{hi} (Tables C3, B3)")
    print(T.pivot(index="part", columns="h", values="share").round(2).to_string())
    save(out, "jkp_fm", jkp_fm(F, F0, a.model), "level and deviation after the JKP residualization: Fama-MacBeth, bp per sd")
    finish(out, a, log, stories=len(S))


if __name__ == "__main__":
    main()
