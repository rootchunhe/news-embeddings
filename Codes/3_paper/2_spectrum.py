"""Where in the embedding's principal directions the content of Table 6 (category C, firm F, idiosyncratic I) and the event cells
(event group x headline direction, split into the group, the direction within it and the rest within the cell) sit, what the ridge
keeps of each band, which bands carry the forecast's out-of-sample covariance with the return, and where the rewrites move. C, F and
I: the two-way effects of the training window (the stories behind the ridge's covariance matrix), not of Table 6's projection years
(lib.ridge.firm_years). Run from Codes/3_paper/:
    OMP_NUM_THREADS=8 ../../env/bin/python 2_spectrum.py --model roberta-large [--perturb]
Output Results/paper/2_spectrum/<model>/{bands,by_k,axes,dof}.csv + run.json (bands: by band of directions and over all; by_k: every
principal direction k, with cum_<x>: the share of the window's variance and of each part's variance along directions 1..k; dof: the
ridge's effective number of parameters by test year, the sum of its squared shrinkage factors, for Framework/calibration.py;
run.json unseen_cells: by test year, the share of the test stories and of their absolute covariance weights in cells and in groups
the training window lacks)"""
from lib.constants import LARGE, SUBSPACE

from lib.run import start, save, finish, years
from lib.ridge import WindowFit, group_sums, band_index, category_index, load_year, replication_fit, rolling_windows, windows
from lib.panel import rewrite_pairs, rewrites as load_rewrites
from lib.category import bucket

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from common.config import CHUNK, MIN_NAMES

SPLITS = ["C", "F", "I", "G", "D", "W"]                                       # content of Table 6; group, direction, within cell
COV = ["cov_All"] + [f"cov_{j}" for j in SPLITS]


def top_directions(M, r):
    """the r leading right singular directions of M (d x r)"""
    if min(M.shape) > LARGE:
        return randomized_svd(M, r, random_state=0)[2].T
    return np.linalg.svd(M, full_matrices=False)[2][:r].T


def group_of(keys):
    """the event group of each cell key (group/direction)"""
    return pd.Series(keys).str.rsplit("/", n=1).str[0].values


def year_cells(X, c):
    """the stories of (X, c) by cell (event group x headline direction, as Table 1): keys, counts, sums of the embedding"""
    keys, idx = np.unique(bucket(c.group, c.headline).values, return_inverse=True)
    return keys, np.bincount(idx).astype(float), group_sums(idx, X, len(keys))


def window_cells(cells, yrs, mu):
    """the cells of the training years: keys, story counts, centered centroids, group of each cell, the group centroids (the
    story-weighted means of their cells') and the groups' names"""
    keys = np.unique(np.concatenate([cells[y][0] for y in yrs]))
    n, S = np.zeros(len(keys)), np.zeros((len(keys), len(mu)))
    for y in yrs:
        k, a, s = cells[y]
        i = np.searchsorted(keys, k)
        n[i], S[i] = n[i] + a, S[i] + s
    zc = (S - np.outer(n, mu)) / n[:, None]
    grp, groups = pd.factorize(group_of(keys))
    Ng = np.bincount(grp, weights=n)
    zg = np.zeros((len(Ng), len(mu)))
    np.add.at(zg, grp, n[:, None] * zc)
    return keys, n, zc, grp, zg / Ng[:, None], groups


def day_weights(c):
    """story weights w such that sum_s w_s x_s is the within-day covariance (times the firm-days) of the firm-day mean of x with the
    next-day return, on the days with >= MIN_NAMES firms: Table 6's predictive covariance"""
    F = c.groupby(["day0", "permno"]).agg(y=("y", "first"), n=("y", "size")).reset_index()
    F = F[F.groupby("day0").permno.transform("size") >= MIN_NAMES]
    F["w"] = (F.y - F.groupby("day0").y.transform("mean")) / F.n
    return c[["day0", "permno"]].merge(F[["day0", "permno", "w"]], on=["day0", "permno"], how="left").w.fillna(0.0).values


def test_moments(f, X, c, cells):
    """the test year's covariance vectors h_j = sum_s w_s x_s (day_weights) of the centered embedding (All) and of its parts under the
    window's fit: the two-way category and firm effects (C, F; zero for a firm without a firm effect in the training window) and the cell and group
    centroids. A test story's centroids: its cell in the window: the cell's and its group's; only its group in the window: the group's
    for both (adds to G, nothing to D = cell - G); its group absent: zero for both (adds only to W = All - cell). Also returns the
    shares of the test stories, and of their absolute weights, in the last two cases"""
    w = day_weights(c)
    h = {"All": sum(np.asarray(X[i:i + CHUNK], np.float64).T @ w[i:i + CHUNK] for i in range(0, len(X), CHUNK)) - f.mu * w.sum()}
    h["C"] = (f.alpha - f.mu).T @ np.bincount(c.code.values, w, len(f.alpha))
    fi, seen = f.firm_index(c.permno.values)
    h["F"] = f.gamma.T @ np.bincount(fi, w * seen, len(f.gamma))
    keys, _, zc, _, zg, groups = cells
    k = bucket(c.group, c.headline).values
    ci, gi = pd.Index(keys).get_indexer(k), pd.Index(groups).get_indexer(group_of(k))       # -1: the window lacks the cell, the group
    new_cell, new_group = (ci < 0) & (gi >= 0), gi < 0
    assert not ((ci >= 0) & new_group).any(), "a cell of the window whose group is not"
    wg = np.bincount(gi[~new_group], w[~new_group], len(zg))                 # every story of a group the window has
    h["G"] = zg.T @ wg
    h["cell"] = zc.T @ np.bincount(ci[ci >= 0], w[ci >= 0], len(keys)) + zg.T @ np.bincount(gi[new_cell], w[new_cell], len(zg))
    aw = np.abs(w)
    return h, {"stories_new_cell": float(new_cell.mean()), "stories_new_group": float(new_group.mean()),
               "weight_new_cell": float(aw[new_cell].sum() / aw.sum()), "weight_new_group": float(aw[new_group].sum() / aw.sum())}


def spectrum(f, st, fst, lam, cells, h):
    """per-direction table of one window: the variance lambda_k and the variances of the parts along v_k (C, F, I of the two-way
    projection of the training stories, which fst must hold; between groups, the direction within a group and within cells), the
    shrinkage, and the direction's contribution
    w_k v_k'h_j to the test year's covariance of the forecast and of each part with the return (C + F + I = G + D + W = All, direction
    by direction), plus the eigenvectors V and the ridge coefficients w_k along them"""
    lam_k, V = np.linalg.eigh(f.G)
    order = np.argsort(lam_k)[::-1]
    lam_k, V = np.maximum(lam_k[order], 1e-10), V[:, order]
    w_k = (V.T @ f.g) / (lam_k + lam)
    a, g = (f.alpha - f.mu) @ V, f.gamma @ V
    _, SF, N = fst[:3]
    sz, sf = (st["S"] - np.outer(f.nc, f.mu)) @ V, (SF - np.outer(f.nF, f.mu)) @ V
    var_C, var_F = (f.nc[:, None] * a ** 2).sum(0) / f.n, (f.nF[:, None] * g ** 2).sum(0) / f.n   # second moments about 0 (C and the least-squares F have mean 0; F is 0 for a firm without a firm effect); var_I below is exact either way
    cross = ((sz * a).sum(0) + (sf * g).sum(0) - (g * (N @ a)).sum(0)) / f.n
    _, n, zc, grp, zg, _ = cells
    gw = np.bincount(grp, n)
    var_cell, var_G = (n[:, None] * (zc @ V) ** 2).sum(0) / f.n, (gw[:, None] * (zg @ V) ** 2).sum(0) / f.n   # cell and group centroids: mean 0
    cov = {j: w_k * (V.T @ v) for j, v in h.items()}
    tab = pd.DataFrame({"k": np.arange(1, len(lam_k) + 1), "lambda": lam_k, "s": lam_k / (lam_k + lam),
                        "var_C": var_C, "var_F": var_F, "var_I": lam_k - 2 * cross + var_C + var_F,
                        "var_G": var_G, "var_D": var_cell - var_G, "var_W": lam_k - var_cell,
                        "cov_All": cov["All"], "cov_C": cov["C"], "cov_F": cov["F"], "cov_I": cov["All"] - cov["C"] - cov["F"],
                        "cov_G": cov["G"], "cov_D": cov["cell"] - cov["G"], "cov_W": cov["All"] - cov["cell"]})
    return tab, V, w_k


def summary(t, by):
    """shares of the directions grouped by `by`: the variance share (of the window's total) and the shrinkage, and each part's share
    of the group's variance, averaged over the test years; the covariance share (of the forecast's total) and each part's share of
    the group's covariance, pooled over the test years as in Table 6"""
    yr = t.groupby(["test_year", by], sort=False)
    v = yr[["lambda"] + [f"var_{j}" for j in SPLITS]].sum()
    out = pd.DataFrame({"var_share": v["lambda"] / v["lambda"].groupby(level=0).transform("sum"), "shrinkage": yr.s.mean(),
                        **{f"var_{j}": v[f"var_{j}"] / v["lambda"] for j in SPLITS}}).groupby(by, sort=False).mean()
    c = t.groupby(by, sort=False)[COV].sum()
    out.insert(1, "cov_share", c.cov_All / t.cov_All.sum())
    return out.join(c[COV[1:]].div(c.cov_All, axis=0)).assign(directions=t.groupby(by, sort=False).k.nunique())


def subspaces(f):
    """the category and the firm subspace of a window: the SUBSPACE leading directions of the category centroids and of the firm
    effects of the training stories, each weighted by the square root of its stories"""
    return top_directions(np.sqrt(f.nc)[:, None] * f.zc, SUBSPACE), top_directions(np.sqrt(f.nF)[:, None] * f.gamma, SUBSPACE)


def rewrite_axes(v, DZ, T, Qc, Qf):
    """one axis per rewrite type (mean displacement in the centred coordinates; flips sign-aligned towards the negative word, magnitude /k
    folded into xk) and its share in the category and the firm subspaces (squared projection)"""
    sign = np.ones(len(v))
    sign[(v.variant == "direction flipped").values & ~v.to_negative.values.astype(bool)] = -1
    shrink = v.variant.str.startswith("magnitude /").values
    sign[shrink] = -1
    axis_of = np.where(shrink, v.variant.str.replace("magnitude /", "magnitude x").values, v.variant.values)
    rows = []
    for name in pd.unique(axis_of):
        m = axis_of == name
        u = (DZ[m] * sign[m][:, None]).mean(0)
        u /= np.linalg.norm(u)
        rows.append({"test_year": T, "axis": name, "n": int(m.sum()), "in_category_subspace": float(((Qc.T @ u) ** 2).sum()),
                     "in_firm_subspace": float(((Qf.T @ u) ** 2).sum())})
    return pd.DataFrame(rows)


def main():
    a, log, out = start("2_spectrum", years="", perturb=False)
    lam_T, _, fits = replication_fit(a.model)
    cats = category_index()
    cells, diag, unseen = {}, {}, {}
    P = Zp = None
    if a.perturb:
        P, Zp = load_rewrites(a.model)
    tabs, axes = [], []
    for T, yrs_tr, st, fst in rolling_windows(a.model, cats, years(a.years), firm=True, firm_span=lambda T: windows(T)[0]):
        f = WindowFit(st, fst)                                            # the two-way effects of the training years
        diag[str(T)] = f.diagnostics(fst)
        cells.update({y: year_cells(*load_year(y, a.model, cats, ["headline"])) for y in yrs_tr if y not in cells})
        X, c = load_year(T, a.model, cats, ["headline"])
        cells[T] = year_cells(X, c)                                       # a later window's training year
        win = window_cells(cells, yrs_tr, f.mu)
        h, unseen[str(T)] = test_moments(f, X, c, win)
        del X, c
        tab, V, w_k = spectrum(f, st, fst, float(lam_T[T]), win, h)
        saved = f.replication_beta(fits[T])
        assert np.abs(V @ w_k - saved).max() <= 1e-4 * np.abs(saved).max(), f"{T}: the spectral ridge differs from the replication's coefficients"
        tabs.append(tab.assign(test_year=T))
        s = tab.sum()
        log(f"{T}: d {len(tab)} | test-year covariance shares C {s.cov_C / s.cov_All:.2f} F {s.cov_F / s.cov_All:.2f} I {s.cov_I / s.cov_All:.2f}"
            f" | variance shares C {s.var_C / s['lambda']:.2f} F {s.var_F / s['lambda']:.2f} I {s.var_I / s['lambda']:.2f}"
            f" | test stories in a cell / group the window lacks {100 * unseen[str(T)]['stories_new_cell']:.2f}% / {100 * unseen[str(T)]['stories_new_group']:.2f}%")
        if P is not None and (P.yr == T).any():                         # the geometry of the year's rewrites
            v, Zv, Zo = rewrite_pairs(P, Zp, T)
            axes.append(rewrite_axes(v, f.z(Zv) - f.z(Zo), T, *subspaces(f)))
    t = pd.concat(tabs, ignore_index=True)
    idx, labels = band_index(int(t.k.max()))
    t["band"] = np.array(labels)[idx[t.k.values - 1]]
    bands = save(out, "bands", pd.concat([summary(t, "band"), summary(t.assign(band="all"), "band")]).reset_index())
    save(out, "dof", t.groupby("test_year").s.apply(lambda s: float((s ** 2).sum())).rename("dof").reset_index(),
         "the ridge's effective number of parameters by test year (sum of squared shrinkage factors)")
    t["lambda_share"] = t["lambda"] / t.groupby("test_year")["lambda"].transform("sum")
    cum = {f"cum_{j}": t.groupby("test_year")[j].cumsum() / t.groupby("test_year")[j].transform("sum") for j in ["lambda"] + [f"var_{j}" for j in "CFI"]}
    t = t.rename(columns={"s": "shrinkage"}).assign(**{f"var_{j}": t[f"var_{j}"] / t["lambda"] for j in SPLITS}, **cum)
    by_k = t.groupby("k")[["lambda_share", "shrinkage"] + [f"var_{j}" for j in SPLITS] + list(cum)].mean()
    save(out, "by_k", by_k.join(t.groupby("k")[COV].sum() / t.cov_All.sum()).reset_index())
    print("\n==== bands ====")
    print(bands.round(3).to_string())
    if axes:
        ax = save(out, "axes", pd.concat(axes, ignore_index=True))
        print("\n==== rewrite axes, mean over years ====")
        print(ax.drop(columns=["test_year", "n"]).groupby("axis", sort=False).mean().round(3).to_string())
    finish(out, a, log, test_years=int(t.test_year.nunique()), two_way=diag, unseen_cells=unseen)


if __name__ == "__main__":
    main()
