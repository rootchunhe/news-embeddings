"""Didisheim–Kelly–Pourmohammadi–Tian, step 1: firm-month mean of the headline embeddings.
Run from Codes/2_replication/didisheim/:
    ../../../env/bin/python 01_firm_month_embeddings.py --model roberta-large
Sample: single-company, de-duplicated stories with an event category in the universe (common/universe.in_sample without the return rule; the paper's
single-stock and near-duplicate filters, returns come from the monthly JKP merge in step 2). Month = calendar month of the
story's arrival in New York time (all stories of a month are used at month end; a story arriving after the last close of a month
belongs to it while its first reaction falls in the next month's return). --month trade keys the firm-months by the trade day's
month instead (the paper's timing robustness; give --suffix).
Out: Data/clean/firm_month_emb_<model>.npz (permno, month YYYY-MM, n_stories, E float16) + run_firm_month_<model>.json"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Codes"))
from common.emb import load_emb
from common.runinfo import write_manifest
from common.universe import in_sample, sample_counts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--suffix", default="", help="appended to the output file name (test runs)")
    p.add_argument("--month", choices=["arrival", "trade"], default="arrival")
    a = p.parse_args()
    assert a.month == "arrival" or a.suffix, "--month trade needs --suffix: the default file is the replication's object"
    a.model = a.model.split("/")[-1]
    t0 = time.time()
    out = ROOT / f"Data/clean/firm_month_emb_{a.model}{a.suffix}.npz"
    MANIFEST = out.with_name(f"run_firm_month_{a.model}{a.suffix}.json")
    MANIFEST.unlink(missing_ok=True)                                # a run in progress has no valid manifest
    sums, counts, parts = {}, {}, []
    for fn in sorted((ROOT / "Data/clean").glob("clean_*.parquet")):     # every clean year: the sample period is chosen in step 2
        c = pd.read_parquet(fn, columns=["hid", "permno", "day0", "ts_ny", "dlycap", "single_comp", "category", "dup_ckx"])
        c = c[in_sample(c, need_return=False)].reset_index(drop=True)
        parts.append(c[["permno", "day0"]])
        c["permno"] = c.permno.astype(int)
        c["m"] = pd.to_datetime(c.ts_ny if a.month == "arrival" else c.day0).dt.strftime("%Y-%m")
        codes, uniq = pd.factorize(pd.Series(list(zip(c.permno.values, c.m.values))))
        S, n = None, np.zeros(len(uniq), np.int64)
        for s0 in range(0, len(c), 200000):
            blk = load_emb(a.model, c.hid.values[s0:s0 + 200000], np.float32)
            if S is None:
                S = np.zeros((len(uniq), blk.shape[1]), np.float64)
            np.add.at(S, codes[s0:s0 + 200000], blk)
            np.add.at(n, codes[s0:s0 + 200000], 1)
        for k, u in enumerate(uniq):
            sums[u] = sums.get(u, 0) + S[k]
            counts[u] = counts.get(u, 0) + n[k]
        print(fn.stem, f"{len(c):,} stories", f"[{time.time() - t0:.0f}s]", flush=True)
    sample_counts(pd.concat(parts), a.model)
    keys = sorted(sums)
    E = np.stack([sums[k] / counts[k] for k in keys]).astype(np.float16)
    np.savez(out, permno=np.array([k[0] for k in keys]), month=np.array([k[1] for k in keys]), n_stories=np.array([counts[k] for k in keys]), E=E)
    write_manifest(MANIFEST, vars(a), output=str(out), firm_months=len(keys), seconds=round(time.time() - t0))   # depends on the sample only
    print("saved", E.shape, flush=True)


if __name__ == "__main__":
    main()
