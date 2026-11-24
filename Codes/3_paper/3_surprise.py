"""Does the headline embedding contain the IBES earnings surprise, and does the forecast use it?
    OMP_NUM_THREADS=8 ../../env/bin/python 3_surprise.py --model roberta-large      (IBES from Codes/1_data/01_pull.py, block ibes)
Output Results/paper/3_surprise/<model>/{panel_a,panel_b}.csv + run.json; Data/clean/story_surprise.parquet
(rebuilt when a file they are built from was added, removed or changed: IBES, the clean files, CRSP)"""
import os

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from lib import ROOT
from lib.constants import NW_LAGS_DAILY, PAPER, SURPRISE, SURPRISE_DUMMY_STORIES, SURPRISE_RANK_PRIOR, TEST_YEARS
from lib.run import finish, save, start
from lib.evaluate import ic_series
from lib.monthly import trailing_level
from lib.panel import firm_day, forecasts, sample_rows

from common.config import lambdas
from common.crsp import daily_panel
from common.emb import load_emb
from common.portfolio import nw_tstat
from common.runinfo import require_same_sample, write_manifest

from newsshock import load_embeddings

RAW = ROOT / "Data/raw/ibes"
TABLES = ["actuals", "surprise", "link"]
EVAL = range(SURPRISE["eval"][0], SURPRISE["eval"][1] + 1)     # test years of the tables
MIN_TRAIN = SURPRISE["min_train"]                               # stories the training part and the validation year of a Panel A fit each need
TARGET = "es_rank"                          # point-in-time centered percentile rank of the price-scaled surprise
SPECS_A = ["C", "C + forecast", "C + E (ridge)", "C + ED (ridge)", "C + EL (ridge)", "C + ED + EL (ridge)"]   # ED + EL: both side by side, own coefficients
OLS_B = {"C": [], "C + forecast": ["yhat"], "C + ES": [TARGET], "C + ES + forecast": [TARGET, "yhat"]}
RIDGE_B = {"C + {} (ridge)": 0, "C + ES + {} (ridge)": 1}       # how many of the columns after the dummies and the no-level indicator (the surprise) enter unpenalized
SPECS_B = ["C", "C + forecast", "C + E (ridge)", "C + ED + EL (ridge)", "C + ES", "C + ES + forecast", "C + ES + E (ridge)",
           "C + ES + ED + EL (ridge)"]
BLOCKS = {"E": 1, "ED": 1, "EL": 1, "ED + EL": 2}         # embedding blocks of each ridge input, each with its own penalty
ESTIMABLE_TOL = 1e-8                                      # require_estimable: relative size of a row's unidentified part that raises


def point_in_time(M, col):
    """winsorized value and centered percentile rank of col against the announcements of the previous twelve months"""
    M = M.sort_values("anndats")
    month = M.anndats.dt.to_period("M")
    by_month = {m: v.values for m, v in M[col].groupby(month)}
    w, r = np.full(len(M), np.nan), np.full(len(M), np.nan)
    for m, idx in month.groupby(month).indices.items():
        prior = np.concatenate([by_month[k] for k in by_month if m - 12 <= k < m] or [np.array([])])
        prior = np.sort(prior[np.isfinite(prior)])
        if len(prior) < SURPRISE_RANK_PRIOR:
            continue
        lo, hi = np.quantile(prior, [0.01, 0.99])
        v = M[col].values[idx].clip(lo, hi)
        w[idx] = v
        r[idx] = np.searchsorted(prior, v) / len(prior) - 0.5
    return M.assign(**{col: w, f"{col}_rank": r})


def announcements(log, cr):
    """one row per (permno, announcement day): fiscal period, announcement time, actual, consensus, dispersion, price, es and its
    point-in-time rank"""
    P, A, L = (pd.read_parquet(RAW / f"{k}.parquet") for k in ["surprise", "actuals", "link"])
    for d, cols in [(P, ["anndats"]), (A, ["anndats", "pends"]), (L, ["sdate", "edate"])]:
        for c in cols:
            d[c] = pd.to_datetime(d[c])
    for c in ["actual", "surpmean", "surpstdev", "suescore"]:
        P[c] = pd.to_numeric(P[c], errors="coerce").astype("float64")
    # the fiscal period (year and month of the period end) is part of the announcement key: a firm can announce several periods
    # on one day, and the surprise and the actual of different periods must not be matched to each other
    P["fp"] = pd.to_numeric(P.pyear, errors="coerce") * 100 + pd.to_numeric(P.pmon, errors="coerce")
    A["fp"] = A.pends.dt.year * 100 + A.pends.dt.month
    P = P.dropna(subset=["actual", "surpmean", "anndats", "fp"]).assign(fp=lambda d: d.fp.astype("int64"))
    A = A[(A.curr_act == "USD") & A.fp.notna()].assign(fp=lambda d: d.fp.astype("int64"))
    key = ["ticker", "anndats", "fp"]
    amb_p, amb_a = P.duplicated(key, keep=False), A.duplicated(key, keep=False)
    log(f"ambiguous IBES rows (same ticker, day and fiscal period): {amb_p.sum():,} surprises, {amb_a.sum():,} actuals; dropped")
    M = P[~amb_p].merge(A[~amb_a][key + ["anntims"]], on=key, how="inner")
    tm = pd.to_timedelta(M.anntims.astype(str), errors="coerce")
    tm = tm.where(tm > pd.Timedelta(0))                                  # 00:00:00 is IBES's placeholder for an unknown time
    log(f"{tm.isna().sum():,} of {len(M):,} announcements without an announcement time (missing or 00:00); dropped")
    M = M.assign(ann_dt=M.anndats + tm)[tm.notna()]
    M = M.merge(L, on="ticker", how="inner")
    M = M[(M.anndats >= M.sdate) & (M.anndats <= M.edate)]
    M = M[M.score == M.groupby(key).score.transform("min")].assign(permno=lambda d: d.permno.astype("int64")).drop_duplicates(key + ["permno"])
    amb = M.duplicated(key, keep=False)
    log(f"{len(M[amb].drop_duplicates(key)):,} announcements whose best-scored links name several PERMNOs: ambiguous, dropped")
    M = M[~amb]
    amb = M.duplicated(["permno", "anndats"], keep=False)
    log(f"{amb.sum():,} announcements of several fiscal periods (or tickers) of one firm on one day: ambiguous, dropped")
    M = M[~amb]
    cr = cr.rename(columns={"dlycaldt": "pday"}).sort_values("pday")
    cr["permno"] = cr.permno.astype("int64")
    M = M.sort_values("anndats")
    M["before"] = M.anndats - pd.Timedelta(days=1)
    M = pd.merge_asof(M, cr[["permno", "pday", "dlyclose"]].rename(columns={"dlyclose": "price"}), left_on="before", right_on="pday", by="permno", tolerance=pd.Timedelta(days=10))
    M["price"] = pd.to_numeric(M.price, errors="coerce").astype("float64")
    M = M[M.price.abs() >= 5].copy()
    M["es"] = (M.actual - M.surpmean) / M.price.abs()
    M = point_in_time(M, "es")
    return M[["permno", "anndats", "ann_dt", "fp", "actual", "surpmean", "surpstdev", "price", "es", "es_rank"]].reset_index(drop=True)


CACHE_COLS = ["rp_story_id", "permno", "hid", "day0", "anndats", "category", "y", "y_fit", "cap", "small", "yr", "day1_yr", "es_rank"]   # cap, small: firm_day


def stories(M, tdays):
    """earnings-group sample stories published between the firm's announcement and the first open after it: the return window of
    every story (that open to the next) is the first trading day after the announcement, and the surprise and the headline are public
    before it starts. The announcement's first open follows the sample's rule for day0 (a time from 09:00 trades the next day) on the
    CRSP trading days tdays."""
    S = sample_rows(TEST_YEARS)
    S = S[S.group == "earnings"].sort_values("ts_ny")
    A = M.sort_values("ann_dt")
    S = pd.merge_asof(S, A, left_on="ts_ny", right_on="ann_dt", by="permno", direction="backward")
    S = S[S.ann_dt.notna() & S.es_rank.notna()].reset_index(drop=True)
    t = S.ann_dt.dt.hour + S.ann_dt.dt.minute / 60
    first = S.ann_dt.dt.normalize() + pd.to_timedelta((t >= 9).astype(int), unit="D")
    S["ann_day0"] = tdays[np.minimum(np.searchsorted(tdays, first.values.astype(tdays.dtype)), len(tdays) - 1)]
    return S[S.day0 == S.ann_day0][CACHE_COLS].reset_index(drop=True)


def story_cache(a, log, cr, tdays):
    """Data/clean/story_surprise.parquet, rebuilt unless its manifest matches the sample definition and the IBES raw files"""
    fn, fn_json = ROOT / "Data/clean/story_surprise.parquet", ROOT / "Data/clean/run_story_surprise.json"
    files = [RAW / f"{k}.parquet" for k in TABLES] + sorted((ROOT / "Data/clean").glob("clean_*.parquet")) + [ROOT / "Data/clean/nyse_breakpoints.parquet"] + sorted((ROOT / "Data/raw/crsp").glob("dsf_*.parquet"))
    inputs = {str(f.relative_to(ROOT)): [f.stat().st_size, f.stat().st_mtime_ns] for f in files}       # every input file: name, size, time
    try:
        m = require_same_sample(fn_json, "earnings stories", needs=PAPER)
        if fn.exists() and m.get("input_files") == inputs and m.get("columns") == CACHE_COLS:
            return pd.read_parquet(fn)
        log("an input of the story cache (IBES, clean files, CRSP or the columns) changed since it was built; rebuilding")
    except RuntimeError as e:
        log(f"{e}; rebuilding")
    M = announcements(log, cr)
    log(f"{len(M):,} announcements with a consensus, a time, a CRSP link and a price")
    S = stories(M, tdays)
    tmp = fn.with_suffix(f".{os.uname().nodename}.{os.getpid()}.tmp.parquet")
    S.to_parquet(tmp, index=False)
    tmp.replace(fn)
    write_manifest(fn_json, vars(a), spec=PAPER, announcements=len(M), stories=len(S), input_files=inputs, columns=CACHE_COLS)
    return S


def dummies(cat, levels):
    return (np.asarray(cat, dtype=object)[:, None] == np.array(levels, dtype=object)[None, :]).astype(np.float64)


def r2(y, p, ybar):
    return 1 - ((y - p) ** 2).sum() / ((y - ybar) ** 2).sum()


def firmday_r2(c, pred, T):
    """out-of-sample R2 of the return, a zero forecast as benchmark, on the firm-days of the rows of c whose return is realized
    before year T"""
    v = c[c.day1_yr < T]
    F = firm_day(v.assign(p=np.asarray(pred)[v.index.values]), ["p"])
    return r2(F.y, F.p, 0.0)


def design(S, nolevel):
    """the unpenalized columns every fit draws from: intercept, a dummy for each category of S, the no-level indicator (nolevel: the
    stories without a level), the surprise last; and each category's column. With the indicator, a fit is the same for any constant
    in place of the zero level of those stories (require_estimable)"""
    cats = sorted(S.category.unique())
    W = np.column_stack([np.ones(len(S)), dummies(S.category, cats), np.asarray(nolevel, np.float64), S[TARGET].values])
    return W, {k: i + 1 for i, k in enumerate(cats)}


def unpen_rcond(p):
    """relative cutoff of the eigenvalues of a p x p Gram matrix of unpenalized columns: below it a direction counts as unidentified"""
    return p * np.finfo(np.float64).eps


def require_estimable(X, j, fit, new, what):
    """raises unless every row of X[new] lies in the row space of X[fit] (X: a fit's unpenalized columns; j: the no-level
    indicator, after the category dummies), without the dummies that are all zero among fit (such a dummy has the coefficient 0;
    the indicator and the regressors always stay): the null directions of the Gram matrix (eigenvalues up
    to unpen_rcond) are orthogonal to every row of X[new], up to ESTIMABLE_TOL times the row's norm. Then the fit's predictions on
    new do not depend on the pseudo-inverse's choice among the unidentified coefficients, and are the same for any constant in
    place of the zero level"""
    use = (X[fit] != 0).any(0)
    use[j:] = True
    Xf, Xn = X[fit][:, use], X[new][:, use]
    e, V = np.linalg.eigh(Xf.T @ Xf)
    N = V[:, e <= unpen_rcond(len(e)) * e.max()]
    bad = np.abs(Xn @ N).max(1, initial=0.0) > ESTIMABLE_TOL * np.linalg.norm(Xn, axis=1)
    if bad.any():
        raise RuntimeError(f"{what}: the unpenalized columns of {int(bad.sum()):,} headlines are outside the row space of the "
                           f"{int(np.sum(fit)):,} fitted headlines (a combination of intercept, category dummies, no-level indicator "
                           "or regressors that the fitted headlines do not identify)")


def small_categories(S, tr, te):
    """log text: the categories with 1 to SURPRISE_DUMMY_STORIES - 1 stories among tr
    (no dummy: they share the intercept), their stories among tr and among te, and the stories of te in categories absent from tr"""
    n = S.category[tr].value_counts()
    small = S.category.isin(n.index[n < SURPRISE_DUMMY_STORIES]).values
    return (f"{int((n < SURPRISE_DUMMY_STORIES).sum())} of {len(n)} categories without a dummy ({int((small & tr).sum()):,} training, "
            f"{int((small & te).sum()):,} test stories), {int((~S.category.isin(n.index).values & te).sum()):,} test stories in new categories")


def sums(W, Z, y, rows, c):
    """the sums over the rows that a window's ridge moments are built from (float64): W'W, W'(Z - c), (Z - c)'(Z - c), W'y, (Z - c)'y;
    the shift c only keeps the centering accurate (ridge_gram centers on the window's own mean)"""
    w, z, t = W[rows], Z[rows].astype(np.float64) - c, y[rows]
    return [w.T @ w, w.T @ z, z.T @ z, w.T @ t, z.T @ t]


def ridge_gram(M, c, cols):
    """the moments of a ridge of y on the unpenalized columns cols of W and the centered embedding, from the window's sums M: mean
    of Z, X'X and X'y of X = [W[:, cols], Z - mean], n (column 0 of W is the intercept)"""
    ww, wz, zz, wy, zy = M
    n = ww[0, 0]
    m = wz[0] / n                                                     # the window's mean of Z - c
    dz = wz[cols] - np.outer(ww[cols, 0], m)
    G = np.block([[ww[np.ix_(cols, cols)], dz], [dz.T, zz - n * np.outer(m, m)]])
    return m + c, G, np.concatenate([wy[cols], zy - m * wy[0]]), int(round(n))


def ridge_fits(gram, nd, lams, keep=None):
    """ridge from ridge_gram's moments: the first nd columns (D: intercept, category dummies, no-level indicator, regressors) unpenalized, the embedding
    penalized with lam * n (the replication's per-observation penalty); keep: the columns of D to use (all by default), so that
    several column sets of D share one Gram matrix. D is profiled out (Frisch-Waugh; its pseudo-inverse gives a dummy level absent
    from a fit subset the coefficient 0), so one eigendecomposition serves every penalty. Returns {lam: predict(Dn, Zn)} with Dn
    holding all nd columns"""
    mu, G, b, n = gram
    k, z = (np.arange(nd) if keep is None else np.asarray(keep)), np.arange(nd, len(b))
    Ai, C = np.linalg.pinv(G[np.ix_(k, k)], rcond=unpen_rcond(len(k))), G[np.ix_(k, z)]   # explicit rank cutoff, as require_estimable
    e, V = np.linalg.eigh(G[np.ix_(z, z)] - C.T @ Ai @ C)
    r = V.T @ (b[z] - C.T @ Ai @ b[k])
    W = {lam: V @ (r / (e + lam * n)) for lam in lams}
    return {lam: (lambda Dn, Zn, a=Ai @ (b[k] - C @ w), w=w: Dn[:, k] @ a + (Zn - mu) @ w) for lam, w in W.items()}


def block_ridge(gram, nd, keep=None):
    """ridge_fits with one penalty per embedding column (pen * n), for the joint input ED + EL whose blocks each get their own penalty:
    one Cholesky solve per penalty vector (the single eigendecomposition of ridge_fits serves every penalty only when it is a scalar).
    Returns fit(pen) -> predict(Dn, Zn)"""
    mu, G, b, n = gram
    k, z = (np.arange(nd) if keep is None else np.asarray(keep)), np.arange(nd, len(b))
    Ai, C = np.linalg.pinv(G[np.ix_(k, k)], rcond=unpen_rcond(len(k))), G[np.ix_(k, z)]   # explicit rank cutoff, as require_estimable
    M, r = G[np.ix_(z, z)] - C.T @ Ai @ C, b[z] - C.T @ Ai @ b[k]

    def fit(pen):
        w = cho_solve(cho_factor(M + np.diag(pen * n)), r)
        return lambda Dn, Zn, a=Ai @ (b[k] - C @ w): Dn[:, k] @ a + (Zn - mu) @ w
    return fit


def group_penalty(fit, lams, blocks, score):
    """the group ridge of Table 4 (Sarkar 2026, Appendix C.3): a common penalty from lams, then each block's in turn with the others
    held, each maximizing score(predict); returns the per-column penalties and the block penalties"""
    pen = lambda ls: np.concatenate([np.full(len(c), l) for c, l in zip(blocks, ls)])
    best = [max(lams, key=lambda l: score(fit(pen([l] * len(blocks)))))] * len(blocks)
    for j in range(len(blocks)):
        best[j] = max(lams, key=lambda l: score(fit(pen(best[:j] + [l] + best[j + 1:]))))
    return pen(best), best


def level(S, model):
    """the level of every story and whether it has one: the mean of the firm's trade-month average embeddings over its news months
    among the previous LEVEL_WINDOW months (lib.monthly.trailing_level; zero without such a month: E^L = 0, E^D = E;
    design's no-level indicator)"""
    require_same_sample(ROOT / f"Data/clean/run_firm_month_{model}_trade.json", f"trade-month firm-month embeddings of {model}")
    keys, E = load_embeddings(model, "_trade", "0000-00", "9999-99")
    L, has = trailing_level(E, keys)
    pos = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_arrays([keys.permno, keys.month]))
    i = pos.loc[list(zip(S.permno.astype(int), S.day0.dt.strftime("%Y-%m")))].values
    return L[i], has[i]


def panel_a(S, Z, W, col, yhat, log):
    """rolling probes of the surprise rank on EVAL, each fitted on the earlier years: OLS on the category dummies and the no-level
    indicator (and the forecast), ridge on the same columns and each embedding of Z (penalty by the OOS R2 of the year before the test
    year; ED + EL: one per block, group_penalty); OOS R2 and Spearman by year.
    W, col: design(S); each embedding's sums cover the years before T - 1 and then add year T - 1; every fit passes
    require_estimable"""
    from scipy.stats import spearmanr
    rows = []
    y, yr = S[TARGET].values, S.yr.values
    shift = {e: Ze[yr < EVAL[0] - 1].mean(0, dtype=np.float64) for e, Ze in Z.items()}
    acc = {e: sums(W, Ze, y, yr < EVAL[0] - 1, shift[e]) for e, Ze in Z.items()}
    for T in EVAL:
        tr, te, last = yr < T, yr == T, yr == T - 1
        assert (tr & ~last).sum() >= MIN_TRAIN and last.sum() >= MIN_TRAIN, f"{T}: fewer than {MIN_TRAIN:,} training or validation stories"
        levels = [c for c, n in S.category[tr].value_counts().items() if n >= SURPRISE_DUMMY_STORIES]
        cols = [0] + [col[c] for c in levels] + [W.shape[1] - 2]            # the no-level indicator after the dummies
        D = W[:, cols]
        require_estimable(D, len(cols) - 1, tr & ~last, last, f"panel A {T}, validation")
        for name, X in [("C", D), ("C + forecast", np.column_stack([D, yhat]))]:      # the ridges' unpenalized columns are those of C
            require_estimable(X, len(cols) - 1, tr, te, f"panel A {T}, test, {name}")
        ybar, preds = y[tr].mean(), {}
        for e, Ze in Z.items():
            blocks = np.array_split(np.arange(Ze.shape[1]), BLOCKS[e])
            if len(blocks) > 1:                                              # ED + EL: a penalty per block
                pen, _ = group_penalty(block_ridge(ridge_gram(acc[e], shift[e], cols), len(cols)), lambdas(len(blocks[0])), blocks,
                                       lambda f: r2(y[last], f(D[last], Ze[last]), y[tr & ~last].mean()))
                acc[e] = [s + t for s, t in zip(acc[e], sums(W, Ze, y, last, shift[e]))]
                preds[f"C + {e} (ridge)"] = block_ridge(ridge_gram(acc[e], shift[e], cols), len(cols))(pen)(D[te], Ze[te])
                continue
            val = ridge_fits(ridge_gram(acc[e], shift[e], cols), len(cols), lambdas(Ze.shape[1]))
            lam = max(val, key=lambda l: r2(y[last], val[l](D[last], Ze[last]), y[tr & ~last].mean()))
            acc[e] = [s + t for s, t in zip(acc[e], sums(W, Ze, y, last, shift[e]))]
            preds[f"C + {e} (ridge)"] = ridge_fits(ridge_gram(acc[e], shift[e], cols), len(cols), [lam])[lam](D[te], Ze[te])
        for name, X in [("C", D), ("C + forecast", np.column_stack([D, yhat]))]:
            preds[name] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        for name in SPECS_A:
            p = preds[name]
            rows.append({"year": T, "spec": name, "n_test": int(te.sum()), "R2_oos": r2(y[te], p, ybar), "spearman": spearmanr(p, y[te])[0]})
        log(f"panel A {T}: " + " | ".join(f"{s} R2 {rows[-len(SPECS_A) + i]['R2_oos']:.3f}" for i, s in enumerate(SPECS_A)) + f" | {small_categories(S, tr, te)}")
    return pd.DataFrame(rows)


def panel_b(S, Z, W, col, log):
    """the next-day return (y_fit) fitted on the stories whose labels are realized before the test year: OLS on the category dummies,
    the no-level indicator and the regressors of OLS_B; ridge on the dummies, the indicator, the surprise (RIDGE_B; all unpenalized)
    and each embedding of Z, the penalty by the
    out-of-sample R2 (firmday_r2) on the stories of the year before the test year (fitted on the earlier ones; ED + EL: one per block,
    group_penalty). On the firm-days of
    EVAL: the rank IC of every fit and its out-of-sample R2 for the return, 1 - sum (y - fit)^2 / sum y^2 (Gu, Kelly and Xiu, 2020).
    W, col: design(S); each embedding's sums cover the stories realized before T - 1 and then add those realized in T - 1; every fit
    passes require_estimable"""
    yr, realized, y = S.yr.values, S.day1_yr.values, S.y_fit.values
    fitted = {k: np.full(len(S), np.nan) for k in SPECS_B}
    shift = {e: Ze[realized < EVAL[0] - 1].mean(0, dtype=np.float64) for e, Ze in Z.items()}
    acc = {e: sums(W, Ze, y, realized < EVAL[0] - 1, shift[e]) for e, Ze in Z.items()}
    for T in EVAL:
        tr, te, val = realized < T, yr == T, yr == T - 1
        levels = [c for c, n in S.category[tr].value_counts().items() if n >= SURPRISE_DUMMY_STORIES]
        cols = [0] + [col[c] for c in levels] + [W.shape[1] - 2, W.shape[1] - 1]       # the no-level indicator, then the surprise last
        D = W[:, cols[:-1]]
        for name, extra in OLS_B.items():                                  # the ridges' unpenalized columns are those of C and C + ES
            X = np.column_stack([D] + [S[e].values[:, None] for e in extra])
            require_estimable(X, D.shape[1] - 1, tr, te, f"panel B {T}, test, {name}")
            fitted[name][te] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        DS = W[:, cols]
        for name, X in [("C", D), ("C + ES", DS)]:
            require_estimable(X, D.shape[1] - 1, realized < T - 1, val, f"panel B {T}, validation, {name}")
        cv, best = S[val].reset_index(drop=True), {}
        for e, Ze in Z.items():
            lams = lambdas(Ze.shape[1])
            keep = {name.format(e): np.arange(D.shape[1] + extra) for name, extra in RIDGE_B.items()}
            g, Zv = ridge_gram(acc[e], shift[e], cols), Ze[val]
            blocks = np.array_split(np.arange(Ze.shape[1]), BLOCKS[e])
            if len(blocks) > 1:                                              # ED + EL: a penalty per block
                pen = {}
                for name, k in keep.items():
                    pen[name], best[name] = group_penalty(block_ridge(g, DS.shape[1], k), lambdas(len(blocks[0])), blocks,
                                                          lambda f: firmday_r2(cv, f(DS[val], Zv), T))
                acc[e] = [s + t for s, t in zip(acc[e], sums(W, Ze, y, realized == T - 1, shift[e]))]
                g, Zt = ridge_gram(acc[e], shift[e], cols), Ze[te]
                for name, k in keep.items():
                    fitted[name][te] = block_ridge(g, DS.shape[1], k)(pen[name])(DS[te], Zt)
                continue
            for name, k in keep.items():
                f = ridge_fits(g, DS.shape[1], lams, k)
                score = {lam: firmday_r2(cv, f[lam](DS[val], Zv), T) for lam in lams}
                best[name] = max(score, key=score.get)
            acc[e] = [s + t for s, t in zip(acc[e], sums(W, Ze, y, realized == T - 1, shift[e]))]
            g, Zt = ridge_gram(acc[e], shift[e], cols), Ze[te]
            for name, k in keep.items():
                fitted[name][te] = ridge_fits(g, DS.shape[1], [best[name]], k)[best[name]](DS[te], Zt)
        log(f"panel B {T}: training {tr.sum():,} stories, penalty " + " ".join(f"{n} " + "/".join(f"{x:g}" for x in np.atleast_1d(l)) for n, l in best.items()))
    for name in SPECS_B:
        S[name] = fitted[name]
    ev = S[S.yr.isin(EVAL)]
    F = firm_day(ev, SPECS_B)
    rows = []
    for name in SPECS_B:
        ic, d = ic_series(F, name), F[[name, "y"]].dropna()
        rows.append({"spec": name, "firm_days": len(d), "R2_%": 100 * r2(d.y, d[name], 0.0),
                     "IC_%": 100 * ic.mean(), "t_IC": nw_tstat(ic.values, NW_LAGS_DAILY)})
    return pd.DataFrame(rows)


def main():
    a, log, out = start("3_surprise")
    cr, tdays = daily_panel(["permno", "dlycaldt", "dlyret", "dlyclose"])     # the CRSP closes (announcements) and trading days
    S = story_cache(a, log, cr, tdays)
    del cr
    log(f"{len(S):,} earnings stories matched to {S.groupby(['permno', 'anndats']).ngroups:,} announcements")
    S = forecasts(S, {"yhat": a.model})
    assert S.yhat.notna().all(), "earnings stories without a forecast"
    for c in ["y", "y_fit", "es_rank", "yhat"]:
        S[c] = pd.to_numeric(S[c], errors="coerce").astype("float64")        # plain numpy floats (no nullable dtypes)
    S["category"] = S.category.astype(object)
    E, (L, has) = load_emb(a.model, S.hid.values.astype("int64"), np.float32), level(S, a.model)
    Z, nolevel = {"E": E, "ED": E - L, "EL": L}, ~has
    log(f"embeddings {E.shape}, level for {100 * has.mean():.1f}% of stories")
    W, col = design(S, nolevel)
    Z["ED + EL"] = np.hstack([Z["ED"], Z["EL"]])
    A = panel_a(S, Z, W, col, S.yhat.values, log)
    summ = A.groupby("spec")[["R2_oos", "spearman"]].mean().reindex(SPECS_A)
    summ["n_test"] = A.groupby("spec").n_test.sum()
    save(out, "panel_a", summ, f"Panel A: OOS prediction of the earnings-surprise rank (mean over test years {EVAL[0]}-{EVAL[-1]})", index=True)
    save(out, "panel_b", panel_b(S, {e: Z[e] for e in ["E", "ED + EL"]}, W, col, log), "Panel B: the first trading day after the announcement, rolling fitted values")
    finish(out, a, log, stories=len(S), level_share=float(has.mean()))


if __name__ == "__main__":
    main()
