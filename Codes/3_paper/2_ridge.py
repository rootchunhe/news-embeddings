"""Exact split of the replication's ridge forecast into its between- and within-category input parts. Run from Codes/3_paper/:
    OMP_NUM_THREADS=8 ../../env/bin/python 2_ridge.py --model roberta-large      (Llama: on a compute node)
Output Results/paper/2_ridge/<model>/decomposition.csv + run.json"""
from lib.constants import PERIODS, REBUILD_TOL

from common.config import CHUNK
from lib.run import start, save, finish, years
from lib.ridge import WindowFit, category_index, load_year, replication_fit, replication_pred, rolling_windows
from lib.panel import firm_day
from lib.evaluate import component_table

import numpy as np
import pandas as pd

PARTS = ["C", "perp"]                                                     # the coefficients times z_C / z_perp
LABELS = {"C": "between input (z_C)", "perp": "within input (z_perp)", "yhat": "forecast"}


def main():
    a, log, out = start("2_ridge", years="")
    lam_T, sd_T, fits = replication_fit(a.model)
    pred = replication_pred(a.model).rename(columns={"yhat": "yhat_ckx"})
    cats = category_index()
    rows = []
    for T, _, st, _ in rolling_windows(a.model, cats, years(a.years)):
        f = WindowFit(st)
        lam = float(lam_T[T])
        beta = f.replication_beta(fits[T])
        b32 = beta.astype(np.float32)
        X, c = load_year(T, a.model, cats)
        S = c[["rp_story_id", "permno", "day0", "y", "cap", "small", "yr", "category", "group"]].copy()
        comp = {k: np.empty(len(c), np.float32) for k in PARTS}
        for i in range(0, len(c), CHUNK):
            z = f.z(X[i:i + CHUNK])
            z_c = f.zc[c.code.values[i:i + CHUNK]].astype(np.float32)
            comp["C"][i:i + CHUNK], comp["perp"][i:i + CHUNK] = z_c @ b32, (z - z_c) @ b32
        for k in PARTS:
            S[k] = comp[k]
        S["yhat"] = f.ybar + S[PARTS].sum(1)
        S = S.merge(pred[pred.test_year == T][["rp_story_id", "permno", "yhat_ckx"]], on=["rp_story_id", "permno"], how="left")
        assert S.yhat_ckx.notna().all(), f"{T}: stories without a replication forecast"
        err = float((S.yhat - S.yhat_ckx).abs().max() / sd_T[T])
        assert err < REBUILD_TOL, f"{T}: the components do not rebuild the replication forecast (max error {err:.1e} forecast sd)"
        rows.append(S)
        log(f"{T}: lambda {lam:g} | max |rebuilt - replication| {err:.2e}")
    S = pd.concat(rows, ignore_index=True)
    max_err = float(((S.yhat - S.yhat_ckx).abs() / S.yr.map(sd_T)).max())
    names = PARTS + ["yhat"]
    F = firm_day(S, names)
    save(out, "decomposition", component_table(F, names, LABELS, PERIODS), "exact decomposition of the forecast (firm-days)")
    finish(out, a, log, stories=len(S), max_rebuild_error=max_err)


if __name__ == "__main__":
    main()
