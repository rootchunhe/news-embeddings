"""Evaluation of signals on firm-day (or firm-month) panels with columns day0, permno, y and the signal columns:
daily rank correlation, decile long-short, performance tables, increments between signals, Fama-MacBeth, factor alphas,
within-period covariances and the additive contributions of a forecast's parts to its covariance with the return."""
import numpy as np
import pandas as pd

from lib.constants import NW_LAGS_DAILY
from lib.panel import segment

from common.config import MIN_NAMES
from common.factors import nw_ols
from common.portfolio import ANN, nw_tstat, sharpe


def _enough(F, cols, by, min_names):
    d = F.dropna(subset=cols)
    return d[d.groupby(by).permno.transform("size") >= min_names]


def ic_series(F, sig, ycol="y", by="day0", min_names=MIN_NAMES):
    """Spearman correlation between the signal and the return within each period with >= min_names names"""
    d = _enough(F, [sig, ycol], by, min_names)
    g = d.groupby(by)
    rs, ry = g[sig].rank(), g[ycol].rank()
    x = pd.DataFrame({"by": d[by].values, "xy": (rs * ry).values, "x": rs.values, "y": ry.values, "xx": (rs ** 2).values, "yy": (ry ** 2).values})
    a = x.groupby("by").agg(n=("x", "size"), xy=("xy", "sum"), x=("x", "sum"), y=("y", "sum"), xx=("xx", "sum"), yy=("yy", "sum"))
    cov = a.xy / a.n - (a.x / a.n) * (a.y / a.n)
    vx, vy = a.xx / a.n - (a.x / a.n) ** 2, a.yy / a.n - (a.y / a.n) ** 2
    return (cov / np.sqrt(vx * vy)).replace([np.inf, -np.inf], np.nan).dropna()


def _ranks(F, sig, by):
    """ranks within the period for the long leg (ascending signal, the larger firm higher among ties) and for the short leg (ascending
    signal, the larger firm lower among ties), the permno last; the period's size; whether the signal varies within the period"""
    d = F[[by, sig, "permno"]].reset_index(drop=True)
    cap = F["cap"].fillna(0).values if "cap" in F else 0.0
    rank = lambda c: (d.assign(_c=c).sort_values([by, sig, "_c", "permno"]).groupby(by).cumcount() + 1).sort_index().values
    g = d.groupby(by)[sig]
    return rank(cap), rank(-cap), g.transform("size").values, (g.transform("nunique") > 1).values


def bucket(r, n, q=10):
    """q-tile 1..q of rank r (1..n) among n: floor(n / q) names in each end bucket (the legs), ceil(q r / n) in between"""
    k = np.floor(n / q)
    return np.where(r > n - k, q, np.where(r <= k, 1, np.clip(np.ceil(q * r / n), 2, q - 1)))


def legs(F, sig, q=10, by="day0"):
    """membership of the top and the bottom q-tile within each period: ties go to the larger firm in either leg, a firm that would
    enter both (one tie block across both cutoffs) enters neither; a period with a constant signal has no legs"""
    up, dn, n, live = _ranks(F, sig, by)
    k = np.floor(n / q)                                                  # names per leg
    top, bot = live & (up > n - k), live & (dn <= k)
    return top & ~bot, bot & ~top


def decile(F, sig, q=10, by="day0", min_names=MIN_NAMES):
    """1..q within the period, the end buckets exactly the legs of ls_series; NaN on rows without a signal and in periods with fewer
    than min_names names or a constant signal"""
    d = _enough(F, [sig], by, min_names)
    up, dn, n, live = _ranks(d, sig, by)
    top, bot = legs(d, sig, q, by)
    out = pd.Series(np.where(bot, 1, np.where(top, q, np.clip(np.ceil(up / n * q), 2, q - 1))), index=d.index).where(live)
    return out.reindex(F.index).astype("Int64")


def ls_weights(F, sig, q=10, vw=False, ycol="y", by="day0", min_names=MIN_NAMES):
    """the rows of ls_series with their weight in the long and in the short q-tile (each leg sums to one per period): columns
    of F plus w_long, w_short; the portfolio of ls_series, e.g. for common.portfolio.close_valued and ewct_table"""
    d = _enough(F, [sig, ycol], by, min_names)
    top, bot = legs(d, sig, q, by)
    both = (pd.Series(top).groupby(d[by].values).transform("any") & pd.Series(bot).groupby(d[by].values).transform("any")).values
    d, top, bot = d[both], top[both], bot[both]                       # a period with an empty leg has no portfolio
    w = d["cap"].fillna(0).clip(lower=0) if vw else pd.Series(1.0, index=d.index)
    for leg, member in zip(["long", "short"], [top, bot]):
        x = w.where(member, 0.0)
        d = d.assign(**{f"w_{leg}": (x / x.groupby(d[by]).transform("sum").replace(0, np.nan)).fillna(0.0)})
    return d


def ls_series(F, sig, q=10, vw=False, ycol="y", by="day0", min_names=MIN_NAMES):
    """top-minus-bottom q-tile return per period (equal- or cap-weighted)"""
    d = ls_weights(F, sig, q, vw, ycol, by, min_names)
    r = d[ycol].values
    a = pd.DataFrame({"by": d[by].values, "rt": r * d.w_long, "rb": r * d.w_short, "wt": d.w_long, "wb": d.w_short}).groupby("by").sum()
    return (a.rt / a.wt - a.rb / a.wb).replace([np.inf, -np.inf], np.nan).dropna()


def perf(F, sigs, labels, periods):
    """one row per (signal, period): IC with its Newey-West t, EW and VW decile long-short mean (bp) and Sharpe"""
    rows = []
    for s in sigs:
        for name, (lo, hi) in periods.items():
            d = segment(F, lo, hi)
            ic, ew, vw = ic_series(d, s), ls_series(d, s), ls_series(d, s, vw=True)
            rows.append({"signal": labels[s], "period": name, "days": len(ew), "IC_%": 100 * ic.mean(), "t_IC": nw_tstat(ic.values, NW_LAGS_DAILY),
                         "EW_LS_bp": 1e4 * ew.mean(), "EW_Sharpe": sharpe(ew), "VW_LS_bp": 1e4 * vw.mean(), "VW_Sharpe": sharpe(vw)})
    return pd.DataFrame(rows)


def mse_gain(F, a, b, ycol="y", by="day0", min_names=MIN_NAMES, lags=NW_LAGS_DAILY):
    """out-of-sample R2 of forecast a relative to forecast b, 1 - SSE_a / SSE_b, and of each against a zero forecast, 1 - SSE / sum y^2
    (y: the unmodified return), on the periods with >= min_names rows complete in both, and the Newey-West t of the per-period
    difference of the summed squared errors"""
    d = _enough(F, [a, b, ycol], by, min_names)
    e = pd.DataFrame({"by": d[by].values, "dse": ((d[ycol] - d[b]) ** 2 - (d[ycol] - d[a]) ** 2).values,
                      "se_a": ((d[ycol] - d[a]) ** 2).values, "se_b": ((d[ycol] - d[b]) ** 2).values, "yy": (d[ycol] ** 2).values}).groupby("by").sum()
    return {"R2_vs_bp": 1e4 * e.dse.sum() / e.se_b.sum(), "R2_a_bp": 1e4 * (1 - e.se_a.sum() / e.yy.sum()), "R2_b_bp": 1e4 * (1 - e.se_b.sum() / e.yy.sum()),
            "t_mse": nw_tstat(e.dse.values, lags), "periods": len(e)}


def jobson_korkie(a, b):
    """z-statistic for the difference of the Sharpe ratios of two aligned series"""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    s1, s2 = a.mean() / a.std(), b.mean() / b.std()
    rho = np.corrcoef(a, b)[0, 1]
    return (s1 - s2) / np.sqrt((2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / n)


def increments(F, pairs, labels):
    """one row per (a, b): difference of the mean IC (Diebold-Mariano t) and of the EW and VW decile long-short Sharpe ratios
    (Jobson-Korkie z), on the common periods"""
    sigs = sorted({s for pair in pairs for s in pair})
    IC = {s: ic_series(F, s) for s in sigs}
    LS = {(s, vw): ls_series(F, s, vw=vw) for s in sigs for vw in (False, True)}
    rows = []
    for a, b in pairs:
        ra, rb = IC[a].align(IC[b], join="inner")
        row = {"A": labels[a], "B": labels[b], "d_IC_%": 100 * (ra.mean() - rb.mean()), "DM_t": nw_tstat((ra - rb).values, NW_LAGS_DAILY)}
        for vw, tag in [(False, ""), (True, "_VW")]:
            la, lb = LS[(a, vw)].align(LS[(b, vw)], join="inner")
            row.update({f"d_Sharpe{tag}": sharpe(la) - sharpe(lb), f"JK_z{tag}": jobson_korkie(la, lb)})
        row["days"] = len(ra)
        rows.append(row)
    return pd.DataFrame(rows)


def fama_macbeth(D, ycol, xcols, by="day0", min_names=MIN_NAMES, scale=1e4, lags=NW_LAGS_DAILY, standardize=True):
    """period-by-period OLS of scale * y on a constant and the regressors (within-period standardized unless standardize=False), on
    the rows complete in all of them; periods with fewer than min_names complete rows are skipped (never recorded as zero). Returns
    (table with one row per regressor, coefficients by period)"""
    d = D.dropna(subset=[ycol] + list(xcols))
    g = d.groupby(by)
    if standardize:
        X = {c: ((d[c] - g[c].transform("mean")) / g[c].transform("std").replace(0, np.nan)) for c in xcols}
    else:
        X = {c: d[c].astype(float) for c in xcols}
    d = d.assign(**{f"_x_{c}": X[c] for c in xcols}).dropna(subset=[f"_x_{c}" for c in xcols])
    d = d[d.groupby(by).permno.transform("size") >= max(min_names, len(xcols) + 2)]
    coefs = {}
    for key, idx in d.groupby(by).indices.items():
        Xm = np.column_stack([np.ones(len(idx))] + [d[f"_x_{c}"].values[idx] for c in xcols])
        coefs[key] = np.linalg.lstsq(Xm, scale * d[ycol].values[idx], rcond=None)[0]
    C = pd.DataFrame(coefs, index=["const"] + list(xcols)).T.sort_index()[list(xcols)]
    rows = [{"x": c, "coef": C[c].mean(), "t": nw_tstat(C[c].values, lags), "periods": len(C)} for c in xcols]
    return pd.DataFrame(rows), C


def factor_alpha(r, factors, lags=NW_LAGS_DAILY):
    """OLS of a return series on factor returns (inner join on the index): annualized alpha, its Newey-West t, R2 and loadings"""
    d = pd.concat([r.rename("r"), factors], axis=1, join="inner").dropna()
    b, t, r2 = nw_ols(d.r.values, d[factors.columns].values, lags)            # HAC standard errors of the full regression
    return {"periods": len(d), "alpha_%/yr": 100 * ANN * b[0], "t_alpha": t[0], "R2": r2, **dict(zip(factors.columns, b[1:]))}


def component_table(F, sigs, labels, periods):
    """one row per (period, component): share of the within-day covariance with the return (contribution_table), of the full
    forecast's variance and of its score, all on the days with >= MIN_NAMES firms complete in every component; IC, EW and VW decile long-short mean (bp) and Sharpe. Each share is the component's own over the
    forecast's, so the covariance and score shares of an additive split of yhat minus its mean add up to one (the variance shares do
    not: the components are correlated)"""
    rows = []
    for name, (lo, hi) in periods.items():
        d = _enough(segment(F, lo, hi), list(sigs) + ["yhat", "y"], "day0", MIN_NAMES)
        parts = [k for k in sigs if k != "yhat"]
        A = contribution_table(predictive_cov(d, parts + ["yhat"]), parts, "yhat", NW_LAGS_DAILY).set_index("part").share
        var_full = xs_cov(d, "yhat", "yhat")
        for k in sigs:
            ic, ls, vw = ic_series(d, k), ls_series(d, k), ls_series(d, k, vw=True)
            rows.append({"period": name, "component": k, "label": labels[k], "A_j": A[k], "score_share": xs_cov(d, k, "yhat") / var_full,
                         "var_share": xs_cov(d, k, k) / var_full, "IC_%": 100 * ic.mean(), "EW_Sharpe": sharpe(ls), "EW_LS_bp": 1e4 * ls.mean(),
                         "VW_Sharpe": sharpe(vw), "VW_LS_bp": 1e4 * vw.mean()})
    return pd.DataFrame(rows)


def xs_cov(F, a, b):
    """pooled within-period (cross-sectional) covariance of columns a and b: period means removed, so intercepts drop out"""
    d = F[["day0", a] + ([b] if b != a else [])].dropna()
    g = d.groupby("day0")
    x = d[a] - g[a].transform("mean")
    y = x if b == a else d[b] - g[b].transform("mean")
    return (x * y).sum() / len(d)


def predictive_cov(F, cols, ycol="y", by="day0", min_names=MIN_NAMES):
    """the within-period covariance of each column with the return as a series over periods (rows complete in every column, periods
    with >= min_names of them): sum over the period's rows of the demeaned products, divided by the average rows per period, so
    that the mean over periods is the pooled within-period covariance of xs_cov. Additive across columns that add up."""
    d = _enough(F, list(cols) + [ycol], by, min_names)
    g = d.groupby(by)
    y = d[ycol] - g[ycol].transform("mean")
    n_bar = g.size().mean()
    return pd.DataFrame({c: ((d[c] - g[c].transform("mean")) * y).groupby(d[by]).sum() / n_bar for c in cols})


def contribution_table(C, parts, total, lags, scale=1.0):
    """mean contribution of each part (column of C, periods x columns) and of the total, Newey-West t, and the share of the total"""
    tot = C[total]
    rows = []
    for k in list(parts) + [total]:
        rows.append({"part": k, "mean": scale * C[k].mean(), "t": nw_tstat(C[k].values, lags), "share": C[k].mean() / tot.mean(), "periods": len(C)})
    return pd.DataFrame(rows)
