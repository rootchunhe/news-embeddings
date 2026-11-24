"""Section 5 inputs on the Didisheim firm-months (month of arrival, the replication's keys): for every test year T and its firm-months,
the event-category part C (mean over the month's stories of the two-way category effect of the fit of T, centered at m_T, the
mean embedding of the projection's headlines (WindowFit.mu_proj), which is stored by test year as window_mean), the firm part F (the firm effect of the fit of T) and the level of C (its mean over the firm's news months among the previous LEVEL_WINDOW months, the
months of T - 1 also under the fit of T; 0 without such a month). The idiosyncratic component is U = E - m_T - C - F, formed from the
replication's E by the reader, so the parts add up to E - m_T exactly. Also stored, outside that sum: the firm identity LEVEL_LAG
months earlier, F_lag with its flag seen_lag: the firm effect of test year T - LEVEL_LAG / 12 (lib.ridge.firm_effects for the years
before the first test year), whose stories (lib.ridge.firm_years of that year) end more than LEVEL_LAG months before the firm-month;
0 with seen_lag false for a firm without one. --template instead builds the counterfactual firm-month embedding
from the category template vectors alone (from 2000). No story embedding is read.
Run from Codes/3_paper/:  ../../env/bin/python 4_monthly.py --model roberta-large [--template]
Output Data/clean/firm_month_parts_<model>.npz (or firm_month_emb_<model>_T.npz) + manifest; run.json under Results/paper/4_monthly[_T]/<model>/"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from lib import ROOT
from lib.constants import FIRST_YEAR, LEVEL_LAG, PAPER, TEST_YEARS
from lib.monthly import trailing_level
from lib.panel import sample_rows
from lib.ridge import WindowFit, category_index, firm_effects, firm_index, firm_years, rolling_windows
from lib.run import finish, start

from common.runinfo import require_same_sample, write_manifest


def stories(cats):
    """every sample story (no return rule, as the replication's firm-month embedding) with its arrival month and category code"""
    S = sample_rows(range(FIRST_YEAR, TEST_YEARS[-1] + 2), need_return=False)      # clean files are by trade year: the next one holds December's late arrivals
    S = S[S.ts_ny.dt.year <= TEST_YEARS[-1]].reset_index(drop=True)
    code = cats.get_indexer(S.category)
    assert (code >= 0).all(), "category outside Data/clean/categories.txt; delete the file to rebuild it"
    return pd.DataFrame({"permno": S.permno.values, "month": S.ts_ny.dt.strftime("%Y-%m").values, "code": code})


def firm_month_counts(S, n_cat):
    """firm-month keys (permno, month), story counts and the firm-month x category count matrix"""
    kf, uniq = pd.factorize(pd.MultiIndex.from_arrays([S.permno.values, S.month.values]))
    keys = pd.DataFrame({"permno": uniq.get_level_values(0).astype(np.int64), "month": uniq.get_level_values(1).astype(str)})
    N = csr_matrix((np.ones(len(S)), (kf, S.code.values)), shape=(len(uniq), n_cat))
    return keys, np.asarray(N.sum(1)).ravel(), N


def save_npz(a, log, t0, name, arrays, keys, n, **facts):
    fn = ROOT / f"Data/clean/{name}.npz"
    np.savez(fn.with_suffix(".tmp.npz"), permno=keys.permno.values, month=keys.month.values.astype(str), n_stories=n.astype(int), **arrays)
    fn.with_suffix(".tmp.npz").replace(fn)
    write_manifest(fn.with_name(f"run_{name}.json"), vars(a), spec=PAPER, output=str(fn), firm_months=len(keys), seconds=round(time.time() - t0),
                   **facts)
    log(f"{len(keys):,} firm-months -> {fn.name}")


def check_keys(model, keys, n):
    """the firm-months and story counts of the replication's firm-month embedding on the same months"""
    require_same_sample(ROOT / f"Data/clean/run_firm_month_{model}.json", f"firm-month embedding of {model}")
    z = np.load(ROOT / f"Data/clean/firm_month_emb_{model}.npz")
    base = pd.Series(z["n_stories"], index=pd.MultiIndex.from_arrays([z["permno"].astype(np.int64), z["month"].astype(str)]))
    m = base.index.get_level_values(1)
    base = base[(m >= keys.month.min()) & (m <= keys.month.max())].sort_index()
    ours = pd.Series(n.astype(int), index=pd.MultiIndex.from_frame(keys)).sort_index()
    assert base.index.equals(ours.index) and (base.values == ours.values).all(), \
        f"the parts ({len(ours):,} firm-months) do not match the replication's firm-month embedding ({len(base):,})"


def check_lag(months, last_year):
    """months from December of last_year (the last month of the stories behind a lagged firm effect) to the earliest of `months`
    ('YYYY-MM'); asserts more than LEVEL_LAG"""
    first = min(months)
    gap = 12 * (int(first[:4]) - last_year) + int(first[5:7]) - 12
    assert gap >= LEVEL_LAG + 1, f"the lagged firm effect of {first} uses stories of {last_year}, {gap} months earlier (needs more than {LEVEL_LAG})"
    return gap


def lagged_firm(permno, months, T, vintage):
    """the firm identity LEVEL_LAG months before the firm-months (permno, months) of test year T: the firm effect of test year
    T - LEVEL_LAG / 12, from vintage(year) -> (sorted permno, gamma); returns (F_lag, seen_lag), 0 and False for a firm outside it"""
    assert LEVEL_LAG % 12 == 0, "F is constant within a test year: the lag must be whole years"
    L = T - LEVEL_LAG // 12
    check_lag(months, firm_years(L)[-1])
    p, g = vintage(L)
    fi, seen = firm_index(p, permno)
    return np.where(seen[:, None], g[fi], 0.0), seen


def parts(a, log, t0, cats):
    S = stories(cats)
    out, means, diag = [], {}, {}
    fits = {}                                                                # (permno, gamma) of the earlier test years until their lagged use
    vintage = lambda L: fits.pop(L) if L in fits else firm_effects(a.model, cats, L)   # before the first test year: the projection alone
    for T, _, st, fst in rolling_windows(a.model, cats, TEST_YEARS, firm=True):
        f = WindowFit(st, fst)
        fits[T] = (f.permno[f.has_F], f.gamma[f.has_F])   # the firms with a firm effect, as firm_effects
        d = S[S.month.str[:4].astype(int).isin([T - 1, T])]
        keys, n, N = firm_month_counts(d, len(cats))
        C = np.asarray(N @ (f.alpha - f.mu_proj)) / n[:, None]               # the two-way category effect, centered at the projection mean
        means[T], diag[str(T)] = f.mu_proj, f.diagnostics(fst, d.permno.values)
        fi, seen = f.firm_index(keys.permno.values)
        F = np.where(seen[:, None], f.gamma[fi], 0.0)
        LC, has = trailing_level(C, keys)
        cur = (keys.month.str[:4].astype(int) == T).values
        F_lag, seen_lag = lagged_firm(keys.permno.values[cur], keys.month.values[cur], T, vintage)
        out.append((keys[cur], n[cur], C[cur].astype(np.float32), F[cur].astype(np.float32), LC[cur].astype(np.float32), has[cur], seen[cur],
                    F_lag.astype(np.float32), seen_lag))
        log(f"{T}: {cur.sum():,} firm-months, {100 * has[cur].mean():.0f}% with a level, {100 * seen_lag.mean():.0f}% with a firm effect {LEVEL_LAG} months earlier")
    keys = pd.concat([o[0] for o in out], ignore_index=True)
    n = np.concatenate([o[1] for o in out])
    order = np.lexsort((keys.month.values, keys.permno.values))
    keys, n = keys.iloc[order].reset_index(drop=True), n[order]
    check_keys(a.model, keys, n)
    C, F, LC, F_lag = (np.concatenate([o[k] for o in out])[order] for k in (2, 3, 4, 7))
    years = np.array(sorted(means))
    save_npz(a, log, t0, f"firm_month_parts_{a.model}", {"C": C.astype(np.float16), "LC": LC.astype(np.float16), "F": F.astype(np.float16),
             "F_lag": F_lag.astype(np.float16), "hasL": np.concatenate([o[5] for o in out])[order], "seen": np.concatenate([o[6] for o in out])[order],
             "seen_lag": np.concatenate([o[8] for o in out])[order], "window_year": years, "window_mean": np.stack([means[T] for T in years])},
             keys, n, two_way=diag)
    return len(keys)


def template(a, log, t0, cats):
    require_same_sample(ROOT / f"Data/clean/run_template_emb_{a.model}.json", f"category templates of {a.model}")
    z = np.load(ROOT / f"Data/clean/template_emb_{a.model}.npz")
    order = pd.Index(z["categories"].astype(str)).get_indexer(cats)
    assert (order >= 0).all(), "template embeddings do not cover every category; rerun 2_embed.py"
    keys, n, N = firm_month_counts(stories(cats), len(cats))
    E = np.asarray(N @ z["emb"][order].astype(np.float64)) / n[:, None]
    o = np.lexsort((keys.month.values, keys.permno.values))
    keys, n, E = keys.iloc[o].reset_index(drop=True), n[o], E[o]
    center = E.mean(0)
    save_npz(a, log, t0, f"firm_month_emb_{a.model}_T", {"E": (E - center).astype(np.float16), "center": center.astype(np.float32)}, keys, n)
    return len(keys)


def main():
    a, log, out = start(lambda a: "4_monthly_T" if a.template else "4_monthly", template=False)
    t0 = time.time()
    cats = category_index()
    n = template(a, log, t0, cats) if a.template else parts(a, log, t0, cats)
    finish(out, a, log, firm_months=n)


if __name__ == "__main__":
    main()
