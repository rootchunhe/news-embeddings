"""The CKX ridge from per-year sufficient statistics: the centered embedding, the paper's penalty grid (lambda = c / P per observation),
train T-8..T-3, validation T-2..T-1 (replication 2_replication/ckx/01_regress.py, whose fitted coefficients are read here; the kernel
is common/ridge.py), and the two-way category/firm projection of the decompositions, estimated on the stories of the PROJECTION_YEARS
years before the test year (firm_years: T-6..T-1). Per-year statistics (n, sum x x', sum x y, sum y, per-category counts and sums; per-firm sums and firm x
category counts) are cached in <lib.WORK>/ridge_stats/<model>/ under a key over the category list, the sample rule, the clean file,
the breakpoints and the embedding chunks."""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components

import lib
from lib import ROOT
from lib.constants import BANDS, MIN_FIRM_YEARS, MIN_STORIES, PROJECTION_YEARS
from lib.evaluate import ic_series
from lib.panel import firm_day, sample_rows

from common.config import CHUNK, FIRST_YEAR, TRAIN_YEARS, VAL_YEARS, Y_CLIP, lambdas
from common.emb import load_emb, manifests
from common.ridge import coefficients, moments
from common.runinfo import require_same_sample
from common.universe import DESCRIPTION, UNIVERSE_PCT

STATS_VERSION = 2                                            # bump when year_stats changes what it computes


def category_index():
    """sorted RavenPack categories of all clean files; cached in <lib.WORK>/categories.txt"""
    fn = lib.WORK / "categories.txt"
    if not fn.exists():
        cats = set()
        for f in sorted((ROOT / "Data/clean").glob("clean_*.parquet")):
            cats |= set(pd.read_parquet(f, columns=["category"]).category.dropna())
        tmp = fn.with_suffix(f".{os.uname().nodename}.{os.getpid()}.tmp")  # per process: parallel jobs on several nodes build the same files
        tmp.write_text("\n".join(sorted(cats)))
        tmp.replace(fn)
    return pd.Index(fn.read_text().split("\n"))


def windows(T):
    """(training years, validation years) of test year T"""
    return list(range(max(FIRST_YEAR, T - TRAIN_YEARS - VAL_YEARS), T - VAL_YEARS)), list(range(T - VAL_YEARS, T))


def firm_years(T):
    """the years of the two-way category/firm projection of test year T: the PROJECTION_YEARS years before T, from FIRST_YEAR. The projection uses
    the sample stories' embeddings, categories and firms, no returns (sample membership requires a next-day return)"""
    return list(range(max(FIRST_YEAR, T - PROJECTION_YEARS), T))


def replication_pred(model):
    """the replication's forecast file after the sample check"""
    require_same_sample(ROOT / f"Data/clean/run_pred_{model}.json", f"forecasts of {model}", needs=("ridge",))
    pred = pd.read_parquet(ROOT / f"Data/clean/pred_{model}.parquet", columns=["rp_story_id", "permno", "test_year", "yhat", "lam_r"])
    pred["permno"] = pred.permno.astype("int64")
    return pred


def replication_fit(model):
    """the replication's ridge by test year: (lam_T, sd_T, fits) with the penalty, the forecast dispersion and, per test year, the
    fitted coefficients and training count {mu, ybar, beta, lam, n_train} that 01_regress.py saved (Data/clean/ridge_fit_<model>.npz)"""
    g = replication_pred(model).groupby("test_year")
    z = np.load(ROOT / f"Data/clean/ridge_fit_{model}.npz")
    fits = {int(T): {"mu": z["mu"][i], "ybar": float(z["ybar"][i]), "beta": z["beta"][i], "lam": float(z["lam"][i]), "n_train": int(z["n_train"][i])} for i, T in enumerate(z["test_year"])}
    return g.lam_r.first(), g.yhat.std(), fits


def load_year(y, model, cats, columns=(), need_return=True):
    """embeddings and the sample rows of year y (plus `columns`; need_return=False: also the stories without a next-day return) with
    the category code into cats"""
    c = sample_rows([y], columns, need_return)
    c["code"] = cats.get_indexer(c.category)
    assert (c.code >= 0).all(), "category outside Data/clean/categories.txt; delete the file to rebuild it"
    return load_emb(model, c.hid.values, np.float16), c


# ---- per-year statistics cache ----------------------------------------------------------------------------------------------

def _key(model, y, cats, kind):
    files = [ROOT / f"Data/clean/clean_{y}.parquet", ROOT / "Data/clean/nyse_breakpoints.parquet"]
    stamp = "|".join(f"{f.stat().st_mtime_ns}:{f.stat().st_size}" for f in files if f.exists())
    emb = json.dumps(manifests(model), sort_keys=True, default=str)
    text = "\n".join(cats) + DESCRIPTION + f"|{UNIVERSE_PCT}|{Y_CLIP}|{STATS_VERSION}|" + stamp + emb + kind
    return hashlib.sha1(text.encode()).hexdigest()


def group_sums(idx, X, K):
    """sum of the rows of X by group index (K groups)"""
    S = np.zeros((K, X.shape[1]))
    for i in range(0, len(X), CHUNK):
        Z = np.asarray(X[i:i + CHUNK], dtype=np.float64)
        S += csr_matrix((np.ones(len(Z)), (idx[i:i + CHUNK], np.arange(len(Z)))), shape=(K, len(Z))) @ Z
    return S


def year_stats(X, c, C):
    """n, sum x x', sum x y, sum y and per category: counts, sum x (C x d), sum y"""
    y, code = c.y_fit.values, c.code.values
    d = X.shape[1]
    sxx, sxy = np.zeros((d, d)), np.zeros(d)
    for i in range(0, len(X), CHUNK):
        Z = np.asarray(X[i:i + CHUNK], dtype=np.float64)
        sxx += Z.T @ Z
        sxy += Z.T @ y[i:i + CHUNK]
    return {"n": len(X), "sxx": sxx, "sxy": sxy, "sy": y.sum(), "nc": np.bincount(code, minlength=C).astype(float),
            "S": group_sums(code, X, C), "syc": np.bincount(code, weights=y, minlength=C)}


def year_firm_stats(X, c, C):
    """per-firm sums of the embedding (permno sorted, SF: firms x d) and the firm x category count matrix N"""
    permno, fidx = np.unique(c.permno.values, return_inverse=True)
    N = np.zeros((len(permno), C), np.int32)
    np.add.at(N, (fidx, c.code.values), 1)
    return {"permno": permno, "SF": group_sums(fidx, X, len(permno)), "N": N}


def _cached(fn, key, keys):
    if fn.exists():
        z = np.load(fn)
        if str(z["key"]) == key:
            return {k: z[k] for k in keys}
    return None


def _store(fn, key, st):
    fn.parent.mkdir(parents=True, exist_ok=True)
    tmp = fn.with_suffix(f".{os.uname().nodename}.{os.getpid()}.tmp.npz")
    np.savez(tmp, key=key, **st)
    tmp.replace(fn)


def stats_year(model, y, cats, firm=False, loaded=None):
    """cached year statistics; with firm=True also the firm statistics (the embeddings are read once for both).
    Returns st or (st, fst). `loaded` = (X, c) from load_year if already in memory"""
    d = lib.WORK / "ridge_stats" / model
    k_s, k_f = _key(model, y, cats, "stats"), _key(model, y, cats, "firm")
    st = _cached(d / f"{y}.npz", k_s, ["n", "sxx", "sxy", "sy", "nc", "S", "syc"])
    fst = _cached(d / f"{y}_firm.npz", k_f, ["permno", "SF", "N"]) if firm else None
    if st is None or (firm and fst is None):
        X, c = loaded if loaded is not None else load_year(y, model, cats)
        if st is None:
            st = year_stats(X, c, len(cats))
            _store(d / f"{y}.npz", k_s, st)
        if firm and fst is None:
            fst = year_firm_stats(X, c, len(cats))
            _store(d / f"{y}_firm.npz", k_f, fst)
    if firm:
        assert (fst["N"].sum(0) == st["nc"]).all(), f"{model} {y}: the firm and category caches disagree; delete them"
        return st, fst
    return st


def sum_firm_stats(fstats, years):
    """union of the firms of the years; summed SF and N; the number of the years with news of each firm"""
    permno = np.unique(np.concatenate([fstats[y]["permno"] for y in years]))
    SF = np.zeros((len(permno), fstats[years[0]]["SF"].shape[1]))
    N = np.zeros((len(permno), fstats[years[0]]["N"].shape[1]), np.int64)
    nY = np.zeros(len(permno), np.int64)
    for y in years:
        idx = np.searchsorted(permno, fstats[y]["permno"])
        SF[idx] += fstats[y]["SF"]
        N[idx] += fstats[y]["N"]
        nY[idx] += 1
    return permno, SF, N, nY


def identified(nF, nY, years):
    """the firms with a firm effect: at least MIN_STORIES stories and news in at least MIN_FIRM_YEARS of the window's `years` years
    (the per-year statistics' years; all of them in a shorter window): a persistent identity needs more than one period. The others
    are treated as unseen"""
    return (nF >= MIN_STORIES) & (nY >= min(MIN_FIRM_YEARS, years))


def kept_effects(gamma, nF, comp, keep):
    """the firm effects of the firms with a firm effect, re-centered to average zero over their stories within each component (as
    two_way's are over all firms); 0 for the others. The shift stays in the kept firms' idiosyncratic part, the category effects unchanged"""
    out = np.zeros_like(gamma)
    for k in np.unique(comp[keep]):
        i = keep & (comp == k)
        out[i] = gamma[i] - nF[i] @ gamma[i] / nF[i].sum()
    return out


def rolling_windows(model, cats, years, firm=False, span=None, firm_span=None):
    """for each T of `years`: (T, its years, summed statistics, firm statistics or None); its years are the training years of test
    year T, or span(T). With firm=True the firm statistics are (permno, SF, N, S): the firms, their sums, the firm x category counts
    and the category sums of the stories of firm_span(T), the years of the two-way projection (default: firm_years(T), or span(T)
    when a span is given), then each firm's years with news and the window's number of years (fst = permno, SF, N, S, nY, years).
    The per-year caches are loaded once and dropped when no later window needs them"""
    firm_span = (firm_span or span or firm_years) if firm else (lambda T: [])
    stats, fstats = {}, {}
    for T in years:
        yrs, yrs_f = span(T) if span else windows(T)[0], firm_span(T)
        if not yrs or (firm and not yrs_f):
            continue
        for y in sorted(set(yrs) | set(yrs_f)):
            if y in yrs_f and y not in fstats:
                stats[y], fstats[y] = stats_year(model, y, cats, firm=True)
            elif y not in stats:
                stats[y] = stats_year(model, y, cats)
        for y in [y for y in stats if y < min(yrs + yrs_f)]:
            del stats[y]
        for y in [y for y in fstats if y < yrs_f[0]]:
            del fstats[y]
        fst = None
        if firm:
            permno, SF, N, nY = sum_firm_stats(fstats, yrs_f)
            fst = (permno, SF, N, sum(stats[y]["S"] for y in yrs_f), nY, len(yrs_f))
        yield T, yrs, {k: sum(stats[y][k] for y in yrs) for k in stats[yrs[0]]}, fst


# ---- the fit of one window ----------------------------------------------------------------------------------------------------

def two_way(SC, SF, N, mu):
    """the two-way projection of a set of stories' embeddings on category and firm indicators, E = alpha_c + gamma_f + residual, by
    least squares from their category sums SC (C x d), firm sums SF (F x d) and counts N (F x C). Within each connected component of the
    firm-category graph the effects are identified up to a shift between its categories and its firms; the firm effects average zero
    over the component's stories, the offset going to its category effects (a firm alone in its component has effect 0). Categories
    without stories get alpha = mu, a centered effect of 0; the other rows of alpha and all of gamma do not depend on mu. Returns alpha (C x d, raw), gamma (F x d) and the component of
    every firm"""
    Nf = N.astype(float)
    nC, nF = Nf.sum(0), Nf.sum(1)
    Sc, Sf = SC - np.outer(nC, mu), SF - np.outer(nF, mu)                  # centered sums
    alpha, gamma = np.zeros_like(Sc), np.zeros_like(Sf)
    A = csr_matrix(Nf > 0)
    _, comp = connected_components(bmat([[None, A], [A.T, None]]), directed=False)
    comp_f, comp_c = comp[:len(nF)], comp[len(nF):]
    for k in np.unique(comp_f):
        fi, ci = np.flatnonzero(comp_f == k), np.flatnonzero(comp_c == k)
        Nk, w = Nf[np.ix_(fi, ci)], 1 / nF[fi]
        M = np.diag(nC[ci]) - Nk.T @ (w[:, None] * Nk)                     # Schur complement; its null space is the constant vector
        a = np.linalg.solve(M + np.full(M.shape, nC[ci].mean() / len(ci)), Sc[ci] - Nk.T @ (w[:, None] * Sf[fi]))
        g = w[:, None] * (Sf[fi] - Nk @ a)
        shift = nF[fi] @ g / nF[fi].sum() if len(fi) > 1 else g[0]
        alpha[ci], gamma[fi] = a + shift, g - shift
    return alpha + mu, gamma, comp_f


def firm_index(permno_sorted, permno):
    """(row of each permno in the sorted firm list, whether it is there); absent firms point to row 0"""
    fi = np.minimum(np.searchsorted(permno_sorted, permno), len(permno_sorted) - 1)
    seen = permno_sorted[fi] == permno
    return np.where(seen, fi, 0), seen


def firm_effects(model, cats, T):
    """(permno sorted, gamma) of the firms with a firm effect (identified) in the two-way projection of the stories of firm_years(T),
    T >= FIRST_YEAR + 1; equal to the WindowFit of test year T on those firms (gamma does not depend on the centering, two_way)"""
    yrs = firm_years(T)
    assert yrs, f"no year before {T}"
    S, fstats = 0, {}
    for y in yrs:
        st, fstats[y] = stats_year(model, y, cats, firm=True)
        S = S + st["S"]
    permno, SF, N, nY = sum_firm_stats(fstats, yrs)
    nF = N.sum(1).astype(float)
    keep = identified(nF, nY, len(yrs))
    _, gamma, comp = two_way(S, SF, N, S.sum(0) / N.sum())
    return permno[keep], kept_effects(gamma, nF, comp, keep)[keep]


class WindowFit:
    """one training window in the ridge's coordinates z = E - mu: G = Cov(z), g = Cov(z, y), the categories' centered centroids zc
    (0 for categories without training stories) and, with firm statistics fst = (permno, SF, N, S, nY, years) (rolling_windows), the
    two-way effects of the stories behind fst, which need not be the training stories: mu_proj (their mean embedding; mu when they are
    the training stories), alpha (C x d, raw; mu_proj, an effect of 0 centered at mu_proj, for categories without such stories), gamma
    (F x d; kept_effects: 0 for a firm without a firm effect, see identified), has_F (identified), gamma_ls (the least-squares firm
    effects of all firms), the firms' components (two_way) numbered, their stories, whether a firm lies in the component with the most stories,
    permno, nF (the firm's stories behind fst)"""

    def __init__(self, st, fst=None):
        n, nc, S = st["n"], st["nc"], st["S"]
        self.n, self.nc = int(n), nc
        self.mu, self.ybar, self.G, self.g = moments(n, S.sum(0), st["sxx"], st["sxy"], st["sy"])
        Sz = S - np.outer(nc, self.mu)
        self.zc = Sz / np.maximum(nc, 1)[:, None]
        self.mu32 = self.mu.astype(np.float32)
        if fst is not None:
            self.permno, SF, N, S_two_way, nY, years = fst
            self.mu_proj = S_two_way.sum(0) / N.sum()
            self.alpha, self.gamma_ls, comp = two_way(S_two_way, SF, N, self.mu_proj)   # on all stories: alpha does not depend on the rule
            self.nF = N.sum(1).astype(float)
            self.has_F = identified(self.nF, nY, years)
            self.gamma = kept_effects(self.gamma_ls, self.nF, comp, self.has_F)   # an unidentified firm's effect stays in the idiosyncratic part
            _, self.comp_idx = np.unique(comp, return_inverse=True)            # the firms' components, numbered 0, 1, ...
            self.comp_stories = np.bincount(self.comp_idx, self.nF)
            self.largest = self.comp_idx == np.argmax(self.comp_stories)       # firms in the component with the most stories

    def firm_index(self, permno):
        """firm_index on this fit's firms, seen only for a firm with a firm effect (has_F)"""
        fi, seen = firm_index(self.permno, permno)
        return fi, seen & self.has_F[fi]

    def diagnostics(self, fst, permno=None, firm_mask=None):
        """checks of the two-way fit (gamma_ls): the largest normal-equation residual of categories and firms relative to the largest
        sum, the largest story-weighted mean firm effect of a component relative to the largest firm effect, the number of components, the
        share of the projection's stories (those behind fst) and the firms (or those of firm_mask) outside the largest component, the
        firms and stories without a firm effect and, with permno (the stories the fit is applied to), the share of those stories whose
        firm lies outside the largest component and of those without a firm effect"""
        _, SF, N, SC = fst[:4]
        Nf = N.astype(float)
        scale = max(np.abs(SC).max(), np.abs(SF).max())
        idx, sizes, outside = self.comp_idx, self.comp_stories, ~self.largest
        g = self.gamma_ls
        means = np.stack([self.nF[idx == j] @ g[idx == j] / sizes[j] for j in range(len(sizes))])
        out = {"components": int(len(sizes)),
               "normal_residual_categories": float(np.abs(SC - Nf.sum(0)[:, None] * self.alpha - Nf.T @ g).max() / scale),
               "normal_residual_firms": float(np.abs(SF - Nf @ self.alpha - self.nF[:, None] * g).max() / scale),
               "centering_residual": float(np.abs(means).max() / max(np.abs(g).max(), 1e-300)),
               "stories_outside_largest": float(self.nF[outside].sum() / self.nF.sum()),
               "firms_outside_largest": int((outside if firm_mask is None else outside & firm_mask).sum()),
               "firms_without_F": int((~self.has_F).sum()), "stories_without_F": float(self.nF[~self.has_F].sum() / self.nF.sum())}
        if permno is not None:
            fi, seen = firm_index(self.permno, permno)
            out["applied_stories_outside_largest"] = float(np.mean(seen & outside[fi]))
            out["applied_stories_without_F"] = float(np.mean(~(seen & self.has_F[fi])))
        return out

    def replication_beta(self, fit):
        """the replication's coefficients of this window (fit = replication_fit(...)[2][T]); its centring must be this window's"""
        assert np.abs(fit["mu"] - self.mu).max() <= 1e-6 * np.abs(self.mu).max(), "the replication's window mean differs from the cached statistics"
        return fit["beta"]

    def z(self, X):
        """centered rows (float32)"""
        return np.asarray(X, dtype=np.float32) - self.mu32


# ---- the replication's ridge on affine feature sets (2_probe) ----------------------------------------------------------------

def transformed(st, shift):
    """statistics (n, sum x, sum x x', sum x y, sum y) of the features E (shift None), t[code] ("only") or E - t[code] ("minus")"""
    n, sy, nc = st["n"], st["sy"], st["nc"]
    if shift is None:
        return n, st["S"].sum(0), st["sxx"], st["sxy"], sy
    mode, t = shift
    tt, ty, sx_t = t.T @ (nc[:, None] * t), t.T @ st["syc"], nc @ t
    if mode == "only":
        return n, sx_t, tt, ty, sy
    cross = t.T @ st["S"]
    return n, st["S"].sum(0) - sx_t, st["sxx"] - cross - cross.T + tt, st["sxy"] - ty, sy


def fit_ridge(n, sx, sxx, sxy, sy):
    """ridge from summed statistics on the centered features, one coefficient vector per penalty of the grid:
    (mu float32, ybar, W of shape d x L, penalties)"""
    mu, ybar, G, g = moments(n, sx, sxx, sxy, sy)
    lams = lambdas(len(mu))
    return mu.astype(np.float32), ybar, coefficients(G, g, lams).astype(np.float32), lams


def features(X, code, shift):
    Z = np.asarray(X, dtype=np.float32)
    if shift is None:
        return Z
    mode, t = shift
    t_rows = t[code].astype(np.float32)
    return t_rows if mode == "only" else Z - t_rows


def predict(X, code, shift, fit):
    """forecasts for every penalty of the grid at once: (n, L)"""
    mu, ybar, W, _ = fit
    return np.concatenate([(features(X[i:i + CHUNK], code[i:i + CHUNK], shift) - mu) @ W for i in range(0, len(X), CHUNK)]) + ybar


def validation_score(c, score, T):
    """firm-day rank correlation on the validation rows of test year T (the replication's penalty criterion); rows whose return
    is realized in year T are excluded as in the replication"""
    v = c[c.day1_yr < T]
    v = v.assign(s=np.asarray(score)[v.index.values])                   # y: the raw return, as the replication ranks it
    return ic_series(firm_day(v, ["s"]), "s").mean()


def band_index(d):
    """band (lib.constants.BANDS) of each of d principal directions and the band labels"""
    idx, labels = np.zeros(d, int), []
    for b, (lo, hi) in enumerate(BANDS):
        if lo > d:
            break
        idx[lo - 1:min(hi, d)] = b
        labels.append(f"{lo}-{min(hi, d)}")
    return idx, labels
