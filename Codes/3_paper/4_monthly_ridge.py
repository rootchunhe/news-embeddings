"""Section 4, monthly: two linear predictors on the firm-month average embedding (all sample stories of the trade month):
  bridge   the replication's daily ridge coefficients of test year T applied to the firm-month average (= the firm-month mean of the
           centered daily forecasts): a fixed-coefficient bridge between daily and monthly prediction;
  ridge    a monthly ridge with the replication's estimator: centered average embedding, penalty grid lambda = c / P, train T-8..T-3,
           validation T-2..T-1 by the mean monthly rank correlation (labels realized before T), target the JKP month-ahead excess return
           winsorized at RET_CLIP on the training rows (lib.monthly.jkp_returns: ret_fit); validation and evaluation use the
           unwinsorized return (ret).
Both are split with the daily decomposition's mapping of year T (two-way category and firm effects; the history of T - 1 under the
fit of T) into the cells cL, cD, yF, uL, uD (level: mean over the firm's news months among the previous LEVEL_WINDOW months, 0
without such a month). At every delay tau the score of signal month m - tau predicts the return of month m + 1 (horizon h = tau + 1 in the paper): the
contribution of each cell to the within-month covariance (Newey-West t, share) at every delay of TAUS_FIG (also on the signal years of a segment), IC and EW decile long-short Sharpe at every delay of TAUS_FIG. Appendix: Fama-MacBeth of level and deviation after the replication's JKP residualization; the bridge's contributions within each Fama-French size group of the signal month (lib.monthly.jkp_size_groups).
    OMP_NUM_THREADS=8 ../../env/bin/python 4_monthly_ridge.py --model roberta-large      (Llama: on a compute node)
Output Results/paper/4_monthly_ridge/<model>/{contributions,contributions_size,performance,jkp_fm}.csv + run.json"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from lib.constants import CELLS, FIRST_YEAR, MIN_MONTH_STOCKS, NW_LAGS_MONTHLY, SEGMENTS, TAUS, TAUS_FIG, TEST_YEARS, TIMING
from lib.evaluate import contribution_table, fama_macbeth, ic_series, ls_series, predictive_cov
from lib.monthly import jkp_ranks, jkp_returns, jkp_size_groups, replication_characteristics, trailing_level
from lib.panel import SIZE_GROUPS
from lib.ridge import WindowFit, category_index, load_year, replication_fit, rolling_windows, windows
from lib.run import finish, save, start

from common.config import CHUNK, lambdas
from common.jkp import monthly_sharpe
from common.portfolio import nw_tstat
from common.ridge import coefficients, moments
from newsshock import residualize

PREDICTORS = ["bridge", "ridge"]
PARTS = CELLS + list(TIMING)


def firm_months(model, cats, J):
    """per clean year: firm-month rows (permno, month Period[M], n stories, ret = JKP month-ahead return or NaN and ret_fit, the same
    winsorized: the ridge's training target), the average embedding (float32) and the firm-month x category count matrix"""
    out = {}
    for y in range(FIRST_YEAR, TEST_YEARS[-1] + 1):
        X, c = load_year(y, model, cats, need_return=False)                 # every sample story (no return rule)
        code, uniq = pd.factorize(pd.MultiIndex.from_arrays([c.permno.values, c.day0.dt.to_period("M")]))
        G = csr_matrix((np.ones(len(c)), (code, np.arange(len(c)))), shape=(len(uniq), len(c)))
        S = np.zeros((len(uniq), X.shape[1]))
        for i in range(0, len(c), CHUNK):
            S += G[:, i:i + CHUNK] @ np.asarray(X[i:i + CHUNK], np.float64)
        n = np.asarray(G.sum(1)).ravel()
        rows = pd.DataFrame({"permno": uniq.get_level_values(0).astype("int64"), "month": uniq.get_level_values(1), "n": n})
        rows = rows.merge(J[["permno", "month", "ret", "ret_fit"]], on=["permno", "month"], how="left")
        out[y] = (rows, (S / n[:, None]).astype(np.float32), csr_matrix((np.ones(len(c)), (code, c.code.values)), shape=(len(uniq), len(cats))))
    return out


def ridge_stats(rows, Xb):
    """a training year's moments of the average embedding and the winsorized return (ret_fit)"""
    ok = rows.ret_fit.notna().values
    X, y = Xb[ok].astype(np.float64), rows.ret_fit.values[ok]
    return ok.sum(), X.sum(0), X.T @ X, X.T @ y, y.sum()


def monthly_ridge(fm, stats, T, log):
    """(mu, beta) of test year T: training years' moments (target ret_fit), validation by the mean monthly Spearman correlation
    with ret"""
    tr, va = windows(T)
    for y in tr:
        if y not in stats:
            stats[y] = ridge_stats(*fm[y][:2])
    S = [stats[y] for y in tr]
    mu, _, G, g = moments(sum(s[0] for s in S), *(sum(s[k] for s in S) for k in range(1, 5)))
    lams = lambdas(len(mu))
    W = coefficients(G, g, lams)
    V = pd.concat([fm[y][0] for y in va], ignore_index=True)
    Xv = np.concatenate([fm[y][1] for y in va])
    keep = (V.ret.notna() & (V.month < pd.Period(f"{T - 1}-12", "M"))).values       # the return of December T-1 is realized in T
    V = V[keep].reset_index(drop=True).assign(**{f"s{j}": col for j, col in enumerate(((Xv[keep] - mu) @ W).T)})
    score = [ic_series(V, f"s{j}", ycol="ret", by="month", min_names=MIN_MONTH_STOCKS).mean() for j in range(len(lams))]
    j = int(np.nanargmax(score))
    log(f"{T}: monthly ridge train {tr[0]}-{tr[-1]} ({sum(s[0] for s in S):,} firm-months), penalty {lams[j]:.3g}, validation IC {100 * score[j]:.2f}%")
    return mu, W[:, j]


def cells(fm, years, f, mu, beta):
    """the cells of the score (X - mu)' beta for the firm-months of `years` (the last one is the test year) under the window fit f: the
    components are centered at the projection mean f.mu_proj, and the offset (f.mu_proj - mu)' beta, common to all rows, goes to cD"""
    a_cat = (f.alpha - f.mu_proj) @ beta                                            # centered category effect in score units
    g_firm, off = f.gamma @ beta, float((f.mu_proj - mu) @ beta)
    parts = []
    for y in years:
        rows, Xb, N = fm[y]
        fi, seen = f.firm_index(rows.permno.values)
        s = (Xb.astype(np.float64) - mu) @ beta
        sC = (N @ a_cat) / rows.n.values
        sF = np.where(seen, g_firm[fi], 0.0)
        parts.append(rows[["permno", "month", "n", "ret"]].assign(total=s, yC=sC, yF=sF, yU=s - off - sC - sF, test=y == years[-1]))
    D = pd.concat(parts, ignore_index=True)
    L, has = trailing_level(D[["yC", "yU"]].values, D)
    D = D.assign(hasL=has, cL=L[:, 0], uL=L[:, 1])                       # 0 without history
    D["cD"], D["uD"] = D.yC - D.cL + off, D.yU - D.uL
    for k, v in TIMING.items():
        D[k] = D[v].sum(axis=1)
    return D[D.test].drop(columns=["test", "yC", "yU"]).reset_index(drop=True)


def evaluate(P, J, log, taus, performance=True):
    """contributions of every predictor at every delay of taus and (performance) its IC and EW decile long-short at the same delays"""
    contrib, perf = [], []
    for tau in taus:
        Q = P.drop(columns="ret").assign(month=P.month + tau).merge(J, on=["permno", "month"], how="inner")
        for p in PREDICTORS:
            cols = [f"{p}_{k}" for k in PARTS]
            C = predictive_cov(Q, cols + [f"{p}_total"], ycol="ret", by="month", min_names=MIN_MONTH_STOCKS)
            T = contribution_table(C, cols, f"{p}_total", NW_LAGS_MONTHLY, scale=1e6)
            contrib.append(T.assign(predictor=p, tau=tau, part=T.part.str.replace(f"{p}_", "")))
            if not performance:
                continue
            for k in ["total", "lev", "dev"]:
                ic = ic_series(Q, f"{p}_{k}", ycol="ret", by="month", min_names=MIN_MONTH_STOCKS)
                ls = ls_series(Q, f"{p}_{k}", ycol="ret", by="month", min_names=MIN_MONTH_STOCKS)
                perf.append({"predictor": p, "tau": tau, "signal": k, "IC_%": 100 * ic.mean(), "t_IC": nw_tstat(ic.values, NW_LAGS_MONTHLY), "EW_LS_%/month": 100 * ls.mean(),
                             "EW_Sharpe": monthly_sharpe(ls), "months": len(ls)})
        if performance and tau in TAUS:
            log(f"tau={tau}: " + " | ".join(f"{p} IC {r['IC_%']:.2f} SR {r['EW_Sharpe']:.2f}" for p in PREDICTORS for r in perf[-6:] if r["predictor"] == p and r["signal"] == "total"))
    return pd.concat(contrib, ignore_index=True), pd.DataFrame(perf)


def size_contributions(P, J, Z, log):
    """the bridge's contributions to the within-month covariance among the firm-months of each size group of the signal month (Z:
    lib.monthly.jkp_size_groups) at every delay of TAUS_FIG, as in evaluate: covariance x 1e6, Newey-West t, share of the group's total"""
    P = P.merge(Z, on=["permno", "month"], how="left", validate="m:1")
    log("size groups of the firm-months: " + ", ".join(f"{g} {100 * (P['size'] == g).mean():.1f}%" for g in SIZE_GROUPS)
        + f", none {int(P['size'].isna().sum()):,} (left out)")
    cols = [f"bridge_{k}" for k in PARTS]
    rows = []
    for tau in TAUS_FIG:
        Q = P.drop(columns="ret").assign(month=P.month + tau).merge(J, on=["permno", "month"], how="inner")
        for g in SIZE_GROUPS:
            C = predictive_cov(Q[Q["size"] == g], cols + ["bridge_total"], ycol="ret", by="month", min_names=MIN_MONTH_STOCKS)
            T = contribution_table(C, cols, "bridge_total", NW_LAGS_MONTHLY, scale=1e6)
            rows.append(T.assign(predictor="bridge", tau=tau, size=g, part=T.part.str.replace("bridge_", "")))
    return pd.concat(rows, ignore_index=True)


def jkp_fm(P, J, model):
    """appendix: level and deviation residualized month by month on the replication's JKP characteristics, joint Fama-MacBeth
    of the month-ahead return (% per sd) at every delay"""
    chars = replication_characteristics(model)
    K = jkp_ranks(chars)                                                     # ranked across the month's JKP stocks, as newsshock.load_jkp
    rows = []
    for p in PREDICTORS:
        D = P[["permno", "month", f"{p}_lev", f"{p}_dev"]].merge(K[["permno", "month"] + chars], on=["permno", "month"], how="inner")
        D["t"] = pd.factorize(D.month, sort=True)[0]
        R = residualize(D[[f"{p}_lev", f"{p}_dev"]].values.astype(np.float64), D, chars, True, False, fit=False)[0]
        D = D[["permno", "month"]].assign(lev=R[:, 0], dev=R[:, 1])
        for tau in TAUS_FIG:
            Q = D.assign(month=D.month + tau).merge(J, on=["permno", "month"], how="inner")
            t, _ = fama_macbeth(Q, "ret", ["lev", "dev"], by="month", min_names=MIN_MONTH_STOCKS, scale=100, lags=NW_LAGS_MONTHLY)
            rows += [{"predictor": p, "tau": tau, "x": r.x, "coef_pct_per_sd": r.coef, "t": r.t, "months": r.periods} for r in t.itertuples()]
    return pd.DataFrame(rows)


def main():
    a, log, out = start("4_monthly_ridge")
    _, _, fits = replication_fit(a.model)
    cats = category_index()
    J = jkp_returns()
    fm = firm_months(a.model, cats, J)
    J = J.drop(columns="ret_fit")                                            # below: the unwinsorized return
    log(f"firm-months {sum(len(v[0]) for v in fm.values()):,} in {FIRST_YEAR}-{TEST_YEARS[-1]}")
    stats, panels, cols, diag = {}, [], ["total"] + PARTS, {}
    for T, _, st, fst in rolling_windows(a.model, cats, TEST_YEARS, firm=True):
        f = WindowFit(st, fst)
        diag[str(T)] = f.diagnostics(fst)
        mu_r, beta_r = monthly_ridge(fm, stats, T, log)
        B = cells(fm, [T - 1, T], f, f.mu, f.replication_beta(fits[T]))
        R = cells(fm, [T - 1, T], f, mu_r, beta_r)
        panels.append(B[["permno", "month", "n", "ret", "hasL"]].join(B[cols].add_prefix("bridge_")).join(R[cols].add_prefix("ridge_")))
    P = pd.concat(panels, ignore_index=True)
    gap = max((P[f"{p}_total"] - P[[f"{p}_{k}" for k in CELLS]].sum(axis=1)).abs().max() for p in PREDICTORS)
    assert gap < 1e-8, f"the cells do not add up to the score (max gap {gap:.1e})"
    log(f"panel {len(P):,} firm-months, {100 * P.hasL.mean():.0f}% with a level")
    contrib, perf = evaluate(P, J, log, TAUS_FIG)
    save(out, "contributions", contrib, "contribution to the within-month covariance (x 1e6), t, share", digits=3)
    save(out, "performance", perf, "IC and EW decile long-short of each predictor by delay")
    for s, (lo, hi) in SEGMENTS[a.model].items():
        log(f"signal years {lo}-{hi}:")
        seg = evaluate(P[P.month.dt.year.between(lo, hi)], J, log, TAUS_FIG, performance=False)[0]
        save(out, f"contributions_{s}", seg, f"the same on the signal months of {lo}-{hi} (Tables C3, B3)", digits=3)
    save(out, "jkp_fm", jkp_fm(P, J, a.model), "level and deviation after the JKP residualization: Fama-MacBeth, % per sd")
    save(out, "contributions_size", size_contributions(P, J, jkp_size_groups(), log), "the bridge within each size group of the signal month (x 1e6)", digits=3)
    finish(out, a, log, firm_months=len(P), level_share=float(P.hasL.mean()), two_way=diag)


if __name__ == "__main__":
    main()
