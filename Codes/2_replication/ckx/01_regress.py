"""Rolling-window return-prediction (ridge) and sentiment (logistic) models on frozen headline embeddings (CKX Sec. 2.1, 3.2).
Run from Codes/2_replication/ckx/:
    ../../../env/bin/python 01_regress.py --model roberta-large [--logit] [--device cuda]
Sample: common/universe.in_sample. Window for test year T (years of the trade day): train T-8..T-3, validation T-2..T-1
restricted to stories whose return is realized before T starts, test T; data start in 2000, so 2004-2007 have 2-5 training years.
Ridge: closed form on the centered, unscaled embedding from per-year sufficient statistics (X'X, X'y), target winsorized at
+-Y_CLIP, per-observation penalty c / P for c in RIDGE_GRID, chosen by the validation firm-day Spearman correlation. Logistic
(v2024 sentiment model, --logit, a run of its own): label 1[ret3_c2c > 0], features standardized per dimension, penalty by
validation accuracy; it needs the training matrix. No refit on train+validation.
Out: the ridge, Data/clean/pred_<model>.parquet (rp_story_id, permno, day0, test_year, yhat, lam_r) and the fitted ridge of every
window Data/clean/ridge_fit_<model>.npz (test_year, lam, ybar, n_train, mu, beta; read by 3_paper) with the manifest
run_pred_<model>.json; --logit: Data/clean/sent_<model>.parquet (rp_story_id, permno, day0, test_year, p_sent, lam_s) with
run_sent_<model>.json (read by 02_tables only). A manifest is absent while its run is in progress"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Codes"))
from common.emb import load_emb
from common.config import CHUNK, FIRST_YEAR, MIN_NAMES, TRAIN_YEARS, VAL_YEARS, Y_CLIP, lambdas
from common.ridge import coefficients, moments
from common.runinfo import write_manifest
from common.universe import in_sample, sample_counts

LAMBDAS_S = [1e-2, 1e-1, 1, 10, 100]
COLS = ["rp_story_id", "permno", "day0", "day1", "hid", "o2o_1", "label_pos", "dlycap", "single_comp", "category", "dup_ckx"]
a = dev = None                                                  # command-line arguments and torch device, set by main()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--test_start", type=int, default=2004)
    p.add_argument("--test_end", type=int, default=2026)
    p.add_argument("--train_years", type=int, default=TRAIN_YEARS)
    p.add_argument("--val_years", type=int, default=VAL_YEARS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--logit", action="store_true", help="the logistic sentiment model instead of the ridge")
    p.add_argument("--suffix", default="", help="appended to the output file names (test runs)")
    a = p.parse_args()
    changed = [k for k in ["test_start", "test_end", "train_years", "val_years"] if getattr(a, k) != p.get_default(k)]
    assert not changed or a.suffix, f"non-default {changed}: give --suffix, the default files are the replication every paper script reads"
    return a


def load_year(y):
    """embeddings and labels of the sample rows of year y"""
    c = pd.read_parquet(ROOT / f"Data/clean/clean_{y}.parquet", columns=COLS)
    c = c[in_sample(c)].reset_index(drop=True)
    return load_emb(a.model, c.hid.values), c


def year_stats(cache, stats, y):
    """sufficient statistics of year y: n, sum x, sum x x' (the logistic run: sum x^2 only), sum x y, sum y (float64)"""
    if y not in stats:
        X, c = cache[y]
        yv = c.o2o_1.values.astype(np.float64).clip(-Y_CLIP, Y_CLIP)
        d = X.shape[1]
        sxx, sx, sxy = np.zeros(d if a.logit else (d, d)), np.zeros(d), np.zeros(d)
        for i in range(0, len(X), a.chunk):
            Z = np.asarray(X[i:i + a.chunk], dtype=np.float64)
            sxx += (Z * Z).sum(0) if a.logit else Z.T @ Z
            sx += Z.sum(0)
            sxy += Z.T @ yv[i:i + a.chunk]
        stats[y] = (len(X), sx, sxx, sxy, yv.sum())
    return stats[y]


def gpu(x):
    return torch.tensor(x, device=dev, dtype=torch.float32)


def fit_ridge(cache, stats, years):
    """the ridge run: mu on the device, n and the float64 fit {mu, ybar, W (d x L), lams}; the logistic run: mu and sd on the device
    (its standardization; the ridge is on the centered embedding) and n"""
    S = [year_stats(cache, stats, y) for y in years]
    n = sum(s[0] for s in S)
    if a.logit:
        mu = sum(s[1] for s in S) / n
        return gpu(mu), gpu(np.sqrt(np.maximum(sum(s[2] for s in S) / n - mu ** 2, 1e-8))), n, None
    mu, ym, G, g = moments(n, sum(s[1] for s in S), sum(s[2] for s in S), sum(s[3] for s in S), sum(s[4] for s in S))
    lams = lambdas(len(mu))
    return gpu(mu), None, n, {"mu": mu, "ybar": ym, "W": coefficients(G, g, lams), "lams": lams}


def predict(X, mu, sd, w, b, logit=False):
    out = []
    for i in range(0, len(X), a.chunk):
        z = (torch.from_numpy(np.ascontiguousarray(X[i:i + a.chunk])).to(dev).float() - mu) / sd @ w + b
        out.append(torch.sigmoid(z) if logit else z)
    return torch.cat(out).cpu().numpy()


def standardize(X, mu, sd):
    """(X - mu) / sd as one float32 tensor on the device, filled by chunks"""
    Z = torch.empty(X.shape, device=dev)
    for i in range(0, len(X), a.chunk):
        Z[i:i + a.chunk] = (torch.from_numpy(np.ascontiguousarray(X[i:i + a.chunk])).to(dev).float() - mu) / sd
    return Z


def fit_logit(Zs, y, lam, iters=60):
    """L2-regularized logistic regression on the standardized features Zs, full-batch L-BFGS with chunked loss and gradient"""
    d = Zs.shape[1]
    w = torch.zeros(d, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)
    yf = torch.tensor(y, device=dev, dtype=torch.float32)
    n = len(Zs)

    def closure():
        opt.zero_grad()
        total = 0.0
        for i in range(0, n, a.chunk):
            Z = Zs[i:i + a.chunk]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yf[i:i + a.chunk], reduction="sum") / n
            loss.backward()
            total += loss.item()
        reg = 0.5 * lam * (w * w).sum()
        reg.backward()
        return total + reg.item()

    opt.step(closure)
    return w.detach(), b.detach()


def firmday_rank_corr(c, scores):
    """for every column of scores (stories x k): the mean over the days with >= MIN_NAMES firm-days of the Spearman correlation
    between the firm-day mean score and the next-day o2o return"""
    S = np.asarray(scores, np.float64).reshape(len(c), -1)
    cols = list(range(S.shape[1]))
    d = pd.DataFrame(S).assign(day0=c.day0.values, permno=c.permno.values, y=c.o2o_1.values)
    f = d.groupby(["day0", "permno"]).agg({**{j: "mean" for j in cols}, "y": "first"}).reset_index()
    f = f[f.groupby("day0").permno.transform("size") >= MIN_NAMES]
    r = f.groupby("day0")[cols + ["y"]].rank()                          # average ranks within the day, as Series.corr(method="spearman")
    r = r - r.groupby(f.day0.values).transform("mean")
    num = r[cols].mul(r.y, axis=0).groupby(f.day0.values).sum()
    den = np.sqrt((r[cols] ** 2).groupby(f.day0.values).sum().mul((r.y ** 2).groupby(f.day0.values).sum(), axis=0))
    return (num / den).mean().values


def main():
    global a, dev                                               # read by the helpers above (chunk size, model, device)
    a = parse()
    dev = torch.device(a.device)
    torch.manual_seed(0)
    TAG = a.model.split("/")[-1]
    stem = "sent" if a.logit else "pred"
    OUT = ROOT / f"Data/clean/{stem}_{TAG}{a.suffix}.parquet"
    FITS = ROOT / f"Data/clean/ridge_fit_{TAG}{a.suffix}.npz"
    MANIFEST = OUT.with_name(f"run_{stem}_{TAG}{a.suffix}.json")
    TMP = OUT.with_suffix(".tmp.parquet")
    for f in [MANIFEST, TMP]:
        f.unlink(missing_ok=True)                                   # a run in progress has no valid manifest
    years = sorted(int(f.stem.split("_")[1]) for f in (ROOT / "Data/clean").glob("clean_*.parquet"))
    assert years == list(range(FIRST_YEAR, years[-1] + 1)), f"the clean files are not the years {FIRST_YEAR}-{years[-1]}"
    cache, stats, preds, saved = {}, {}, [], []
    t_all = time.time()
    for T in range(a.test_start, min(a.test_end, years[-1]) + 1):
        yrs_tr = list(range(max(FIRST_YEAR, T - a.train_years - a.val_years), T - a.val_years))
        yrs_va = list(range(T - a.val_years, T))
        t0 = time.time()
        for y in yrs_tr + yrs_va + [T]:
            if y not in cache:
                cache[y] = load_year(y)
        for y in list(cache):
            if y < yrs_tr[0]:
                del cache[y]
                stats.pop(y, None)
        # validation rows: stories whose return is realized before the test year starts
        cva = pd.concat([cache[y][1] for y in yrs_va], ignore_index=True)
        Xva = np.concatenate([cache[y][0] for y in yrs_va])
        keep = (pd.to_datetime(cva.day1).dt.year < T).values
        Xva, cva = Xva[keep], cva[keep].reset_index(drop=True)
        Xte, cte = cache[T]
        mu, sd, n_tr, fit = fit_ridge(cache, stats, yrs_tr)               # the logistic model standardizes with mu, sd
        keys = {"rp_story_id": cte.rp_story_id, "permno": cte.permno, "day0": cte.day0, "test_year": T}
        if not a.logit:
            ones, b = torch.ones_like(mu), gpu(fit["ybar"])
            val_rc = dict(zip(fit["lams"], firmday_rank_corr(cva, predict(Xva, mu, ones, gpu(fit["W"]), b))))   # every penalty in one pass
            lam_r = max(val_rc, key=val_rc.get)
            j = fit["lams"].index(lam_r)
            saved.append({"test_year": T, "lam": lam_r, "ybar": fit["ybar"], "n_train": n_tr, "mu": fit["mu"], "beta": fit["W"][:, j]})
            yhat = predict(Xte, mu, ones, gpu(fit["W"][:, j]), b)
            preds.append(pd.DataFrame({**keys, "yhat": yhat, "lam_r": lam_r}))
            msg = f"ridge lam {lam_r} val-rc {val_rc[lam_r]:.4f} OOS-rc {firmday_rank_corr(cte, yhat)[0]:.4f}"
        else:
            ctr = pd.concat([cache[y][1] for y in yrs_tr], ignore_index=True)
            mtr = ctr.label_pos.notna().values
            mva = cva.label_pos.notna().values
            Ztr = standardize(np.concatenate([cache[y][0] for y in yrs_tr])[mtr], mu, sd)   # once for every penalty
            val_acc, fits_s = {}, {}
            for lam in LAMBDAS_S:
                fits_s[lam] = fit_logit(Ztr, ctr.label_pos.values[mtr], lam)
                val_acc[lam] = ((predict(Xva, mu, sd, *fits_s[lam], logit=True)[mva] > 0.5) == cva.label_pos.values[mva]).mean()
            lam_s = max(val_acc, key=val_acc.get)
            p_sent = predict(Xte, mu, sd, *fits_s[lam_s], logit=True)
            mte = cte.label_pos.notna().values
            acc = ((p_sent[mte] > 0.5) == cte.label_pos.values[mte]).mean()
            preds.append(pd.DataFrame({**keys, "p_sent": p_sent, "lam_s": lam_s}))
            msg = f"logit lam {lam_s} val-acc {val_acc[lam_s]:.4f} OOS-acc {acc:.4f}"
            del Ztr
        print(f"{T}: train {n_tr:,} ({yrs_tr[0]}-{yrs_tr[-1]}) val {len(cva):,} test {len(cte):,} | {msg} | {time.time() - t0:.0f}s", flush=True)
        pd.concat(preds, ignore_index=True).to_parquet(TMP, index=False)      # checkpoint
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    P = pd.concat(preds, ignore_index=True)
    sample_counts(P, "scored")
    TMP.replace(OUT)
    if not a.logit:
        np.savez(FITS.with_suffix(".tmp.npz"), test_year=np.array([r["test_year"] for r in saved]), lam=np.array([r["lam"] for r in saved]),
                 ybar=np.array([r["ybar"] for r in saved]), n_train=np.array([r["n_train"] for r in saved]),
                 mu=np.stack([r["mu"] for r in saved]), beta=np.stack([r["beta"] for r in saved]))
        FITS.with_suffix(".tmp.npz").replace(FITS)
    write_manifest(MANIFEST, vars(a), spec=("ridge",),                    # the ridge group stamps the config's window; non-default windows need --suffix
                   output=str(OUT), **({} if a.logit else {"fits": str(FITS)}), rows=len(P),
                   years=f"{int(P.test_year.min())}-{int(P.test_year.max())}", seconds=round(time.time() - t_all))
    print("ALL DONE ->", OUT)


if __name__ == "__main__":
    main()
