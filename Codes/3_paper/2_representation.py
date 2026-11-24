"""Is the firm component F of the two-way projection a representation of the firm, in the sense of Sarkar (2026, Table 1)? In the
window of every December y (the news of the PROJECTION_YEARS years up to y, from FIRST_YEAR; also from each first year after a pretraining
cutoff and from SPLIT_YEAR, for the post-cutoff tables), F of each firm, its category mix (its mean category component), both together, and,
for comparison, its raw average centered embedding (which still carries the mix of its news categories) explain the firm's valuation and growth expectations at December y: log(P/B), log(P/E) for positive
earnings (JKP) and the IBES median long-term growth forecast (December), against and together with the controls: Fama-French 49
industry dummies and Sarkar's characteristics (CAPM beta, log book equity, asset growth, gross profits to assets, dividends to
assets); outcomes and characteristics winsorized at the 1st and 99th percentiles of each fold's fitting firms in the window
(dividends to assets at the upper tail only, as Sarkar).
Split sample as Sarkar: three folds of companies (permco), fixed across windows; a group ridge as Sarkar (Appendix C.3), one penalty
per group of inputs (the embedding unstandardized; the characteristics and the industry dummies standardized on the training firms),
tuned by leave-one-out on the other two folds; R2 of each held-out fold around its own mean. Also out of sample in time: the fit on
the other two folds of the previous window, applied to the held-out fold of the current one, its intercept
re-estimated on the current other two folds (a time fixed effect). Firms need a firm effect in the window (lib.ridge.identified: MIN_STORIES stories, news in MIN_FIRM_YEARS calendar years). From FIRST_YEAR, the
window of December y is the two-way projection window of test year y + 1 (lib.ridge.firm_years): its F is Table 6's firm
identity of year y + 1. Run from Codes/3_paper/:
    OMP_NUM_THREADS=8 ../../env/bin/python 2_representation.py --model roberta-large
Output Results/paper/2_representation/<model>/r2.csv + run.json"""
import numpy as np
import pandas as pd

from lib.firms import CHARS, FF49, OUTCOMES, decembers, december_windows, firm_inputs, winsorize
from lib.ridge import WindowFit
from lib.run import finish, save, start, years

from common.config import FIRST_YEAR

INPUTS = {"F": ["F"], "average": ["average"], "controls": ["chars", "industry"], "F + controls": ["F", "chars", "industry"],
          "average + controls": ["average", "chars", "industry"], "category mix": ["category mix"],
          "category mix + F": ["category mix", "F"], "category mix + F + controls": ["category mix", "F", "chars", "industry"]}
# groups of inputs, each with its own penalty, in Sarkar's tuning order
STANDARDIZED = {"chars", "industry"}                                         # standardized on the training firms (not the embedding)
ALPHAS = np.logspace(-3, 6, 19)


def blocks(f, fst, c):
    """the input blocks of the firms c (rows c.i of the window fit f)"""
    return firm_inputs(f, fst, c.i.values) | {"chars": c[CHARS].values.astype(float),
            "industry": pd.get_dummies(c.ff49.astype(int)).reindex(columns=FF49, fill_value=0).values.astype(float)}


def fit(X, std, y):
    """Sarkar's group ridge (2026, Appendix C.3): an intercept and one penalty per block of X (std: standardized on these firms), a
    common penalty first, then each block's in turn with the others held, each from ALPHAS by the exact leave-one-out error. Dual
    form on the centered blocks: K = sum_g X_g X_g' / a_g, G = (K + I)^-1, leave-one-out residual (G y)_i / (G_ii - 1/n). Returns the
    prediction of the blocks"""
    n = len(y)
    mu = [x.mean(0) for x in X]
    sd = [np.where(x.std(0) > 0, x.std(0), 1.0) if s else np.ones(x.shape[1]) for x, s in zip(X, std)]
    Z = [(x - m) / s for x, m, s in zip(X, mu, sd)]
    yc = y - y.mean()
    K = [z @ z.T for z in Z]

    def tune(W, w):                                                   # the penalty t minimizing the LOO error for G = W diag(1/(1 + w/t)) W'
        d = 1 / (1 + np.clip(w, 0, None)[:, None] / ALPHAS[None, :])
        Gy, Gd = W @ (d * (W.T @ yc)[:, None]), (W ** 2) @ d
        return ALPHAS[np.argmin(((Gy / (Gd - 1 / n)) ** 2).mean(0))]

    w, U = np.linalg.eigh(sum(K))
    a = [tune(U, w)] * len(K)
    if len(K) > 1:
        for g in range(len(K)):                                       # K + I = M^1/2 (I + B / a_g) M^1/2, M = I + the other blocks
            m, V = np.linalg.eigh(np.eye(n) + sum(K[h] / a[h] for h in range(len(K)) if h != g))
            R = (V / np.sqrt(m)) @ V.T
            s, Q = np.linalg.eigh(R @ K[g] @ R)
            a[g] = tune(R @ Q, s)
    alpha = np.linalg.solve(np.eye(n) + sum(k / t for k, t in zip(K, a)), yc)
    b = [z.T @ alpha / t / s for z, t, s in zip(Z, a, sd)]           # the blocks' coefficients on their raw scale
    c0 = y.mean() - sum(m @ v for m, v in zip(mu, b))
    return lambda X: c0 + sum(x @ v for x, v in zip(X, b))


def r2(y, pred):
    """R2 around the held-out fold's own mean"""
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    a, log, out = start("2_representation", years="")
    D = decembers()
    rows, prev, series, diag = [], None, None, {}
    for first, dec, st, fst in december_windows(a.model, years(a.years)):
        if first != series:                                             # "previous window": within the windows of one first year
            prev, series = None, first
        f = WindowFit(st, fst)
        keep = f.has_F                                                  # a firm effect: MIN_STORIES stories, news in MIN_FIRM_YEARS years
        diag[f"{first}-{dec}"] = f.diagnostics(fst, firm_mask=keep)
        c = pd.DataFrame({"permno": f.permno[keep], "i": np.where(keep)[0]}).merge(D[D.year == dec], on="permno")
        both = OUTCOMES + [x for x in CHARS if x != "div_at"]
        fold = (c.permco.values.astype("int64") * 2654435761 % 2 ** 32) % 3   # a company keeps its fold across windows
        fits = {}
        for k in range(3):
            fitting = fold != k
            ck = c.copy()                                                # winsorized at the fitting firms' percentiles
            ck[both], ck["div_at"] = winsorize(c, both, fitting), winsorize(c, ["div_at"], fitting, lower=False).div_at
            B = blocks(f, fst, ck)
            for o in OUTCOMES:
                y, ok = ck[o].values, ck[o].notna().values
                tr, te = ok & fitting, ok & ~fitting
                for name, groups in INPUTS.items():
                    std = [g in STANDARDIZED for g in groups]
                    Xtr, Xte = [B[g][tr] for g in groups], [B[g][te] for g in groups]
                    fits[o, name, k] = m = fit(Xtr, std, y[tr])
                    rows.append({"first_year": first, "december": dec, "sample": "split", "outcome": o, "inputs": name, "fold": k,
                                 "R2": r2(y[te], m(Xte)), "firms": int(te.sum())})
                    if prev is not None:                             # fitted on the previous window's other folds
                        m = prev[o, name, k]                         # its intercept re-estimated on the current other folds
                        pred = m(Xte) - m(Xtr).mean() + y[tr].mean()
                        rows.append({**rows[-1], "sample": "previous window", "R2": r2(y[te], pred)})
        prev = fits
        log(f"from {first}: December {dec}, {len(c):,} firms (P/E {int(c.log_pe.notna().sum()):,}, LTG {int(c.ltg.notna().sum()):,})")
    R = save(out, "r2", pd.DataFrame(rows), "R2 of each held-out fold: valuations and growth forecasts on each input")
    print((100 * R[R.first_year == FIRST_YEAR].groupby(["sample", "inputs", "outcome"], sort=False).R2.mean().unstack()).round(1).to_string())
    finish(out, a, log, decembers=int(R.december.nunique()), two_way=diag)


if __name__ == "__main__":
    main()
