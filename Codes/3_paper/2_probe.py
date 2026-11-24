"""The replication's ridge refitted on the headline embedding, the category template, the category centroid and the differences,
and the category mean plus a ridge forecast of its residual: E minus its training category centroid is orthogonal to the category
dummies, so its ridge equals the ridge of the return minus the training category mean (Frisch-Waugh). For the combination the penalty
is chosen on the validation years for the combination as deployed: the eight-year category mean of each validation year (labels
realized before it, lib.category) plus the residual forecast; the test forecast adds the residual
forecast to the eight-year category mean. Run from Codes/3_paper/ after 2_embed.py:
    OMP_NUM_THREADS=8 ../../env/bin/python 2_probe.py --model roberta-large
Output Results/paper/2_probe/<model>/{summary,increments,residual,residual_<segment>}.csv + run.json"""
from lib import ROOT
from lib.constants import PERIODS, REBUILD_TOL, SEGMENTS, TEST_YEARS
from lib.category import WINDOWS, category_means, history_rows
from lib.run import start, save, finish, years
from lib.ridge import category_index, fit_ridge, load_year, predict, replication_fit, replication_pred, rolling_windows, stats_year, transformed, validation_score, windows
from lib.panel import firm_day, segment
from lib.evaluate import increments, mse_gain, perf

import numpy as np
import pandas as pd

from common.config import FIRST_YEAR
from common.runinfo import require_same_sample
from common.universe import sample_counts

SIGNALS = ["E", "T", "E_minus_T", "Ebar", "E_minus_Ebar"]
LABELS = {"E": "headline embedding E", "T": "category template t_c alone", "E_minus_T": "E - t_c", "Ebar": "category centroid alone",
          "E_minus_Ebar": "E - centroid (within-category)", "yhat": "CKX replication forecast"}
INCREMENTS = [(s, "E") for s in SIGNALS[1:]]                    # every refitted input against the headline embedding
RESIDUAL = {"m8": "category mean (8-yr)", "m8_resid": "category mean + residual ridge"}


def residual(F, lo=TEST_YEARS[0], hi=TEST_YEARS[-1]):
    """Table 2 Panel B on the signal years lo..hi: the category mean and the category mean + residual ridge, the differences (DM t,
    JK z), the out-of-sample R2 of each against a zero forecast (R2_bp, on the firm-days of both) and relative to the category mean"""
    F = segment(F, lo, hi)
    res = perf(F, list(RESIDUAL), RESIDUAL, {f"{lo}-{hi % 100:02d}": (lo, hi)}).set_index("signal")
    inc = increments(F, [("m8_resid", "m8")], RESIDUAL).iloc[0]
    gain = mse_gain(F, "m8_resid", "m8")
    res.loc[RESIDUAL["m8"], "R2_bp"], res.loc[RESIDUAL["m8_resid"], "R2_bp"] = gain["R2_b_bp"], gain["R2_a_bp"]
    for k in ["d_IC_%", "DM_t", "d_Sharpe", "JK_z", "d_Sharpe_VW", "JK_z_VW", "R2_vs_bp", "t_mse"]:
        res.loc[RESIDUAL["m8_resid"], k] = inc[k] if k in inc else gain[k]
    return res.reset_index(), inc, gain


def main():
    a, log, out = start("2_probe", years="")
    require_same_sample(ROOT / f"Data/clean/run_template_emb_{a.model}.json", f"templates of {a.model}")      # depends on the model and the category list
    tm = np.load(ROOT / f"Data/clean/template_emb_{a.model}.npz")
    cats, t = category_index(), tm["emb"].astype(np.float64)
    assert (tm["categories"] == cats.values).all(), "template file built on another category index"
    test = years(a.years)
    lam_rep = replication_fit(a.model)[0]
    cache, rows, lam_E = {}, [], {}
    H = history_rows(range(FIRST_YEAR, test[-1]))                          # the labels of the validation years' category means
    for T, yrs_tr, tr, _ in rolling_windows(a.model, cats, test):
        yrs_va = windows(T)[1]
        for y in yrs_va + [T]:
            if y not in cache:
                cache[y] = load_year(y, a.model, cats)
                stats_year(a.model, y, cats, loaded=cache[y])          # fills the statistics cache from the rows in memory
        for y in [y for y in cache if y < yrs_va[0]]:
            del cache[y]
        mu = tr["S"].sum(0) / tr["n"]
        cen = np.where(tr["nc"][:, None] > 0, tr["S"] / np.maximum(tr["nc"], 1)[:, None], mu)      # every category's own centroid
        shifts = {"E": None, "T": ("only", t), "E_minus_T": ("minus", t), "Ebar": ("only", cen), "E_minus_Ebar": ("minus", cen)}
        cva = pd.concat([cache[y][1] for y in yrs_va], ignore_index=True)
        Xte, cte = cache[T]
        S = cte[["rp_story_id", "permno", "day0", "day1_yr", "y", "y_fit", "cap", "small", "yr", "category", "group"]].copy()
        lams = {}
        for s, shift in shifts.items():
            fit = fit_ridge(*transformed(tr, shift))
            pva = np.concatenate([predict(cache[y][0], cache[y][1].code.values, shift, fit) for y in yrs_va])
            val = {lam: validation_score(cva, pva[:, j], T) for j, lam in enumerate(fit[3])}
            lams[s] = max(val, key=val.get)
            pte = predict(Xte, cte.code.values, shift, fit)
            S[s] = pte[:, fit[3].index(lams[s])]
            if s == "E":                                               # the refit at the replication's penalty, for the reproduction check
                S["E_rep"] = pte[:, fit[3].index(lam_rep[T])]
            if s == "E_minus_Ebar":                                    # the residual forecast, its penalty chosen for the combination
                m8 = category_means(cva.copy(), "m8", WINDOWS["m8"], yrs_va, history=H[~H.yr.isin(yrs_va)]).m8.values
                comb = {lam: validation_score(cva, m8 + pva[:, j] - fit[1], T) for j, lam in enumerate(fit[3])}
                lams["resid"] = max(comb, key=comb.get)
                S["resid"] = pte[:, fit[3].index(lams["resid"])] - fit[1]
        rows.append(S)
        lam_E[T] = lams["E"]
        log(f"{T}: train {tr['n']:,} ({yrs_tr[0]}-{yrs_tr[-1]}) val {len(cva):,} test {len(cte):,} | penalty " + " ".join(f"{s} {lams[s]:g}" for s in lams))
    S = pd.concat(rows, ignore_index=True)
    sample_counts(S, "P2")
    S = S.merge(replication_pred(a.model)[["rp_story_id", "permno", "yhat"]], on=["rp_story_id", "permno"], how="left")
    assert S.yhat.notna().all(), "stories without a replication forecast"
    err = ((S.E_rep - S.yhat).abs() / S.yr.map(S.groupby("yr").yhat.std())).max()
    assert err < REBUILD_TOL, f"the refitted E does not reproduce the replication forecast (max error {err:.1e} forecast sd)"
    corr_ckx = S[["E", "yhat"]].corr().iloc[0, 1]
    S = S.drop(columns="E_rep")
    other = {T: (lam_E[T], lam_rep[T]) for T in test if not np.isclose(lam_E[T], lam_rep[T], rtol=1e-6)}
    if other:                     # a near tie of two penalties in the validation score can fall the other way (float32 scoring here and there)
        log(f"WARNING: the refitted E picked another penalty than the replication in {other} (refit, replication)")

    category_means(S, "m8", WINDOWS["m8"], test, history=H[H.yr < test[0]])   # all earlier years: a --years run too
    S["m8_resid"] = S.m8 + S.resid
    F = firm_day(S, SIGNALS + ["yhat", "m8", "m8_resid"])
    summary = save(out, "summary", perf(F, SIGNALS + ["yhat"], LABELS, PERIODS))
    print("\n==== firm-days, EW decile long-short ====")
    print(summary.pivot(index="signal", columns="period", values=["IC_%", "EW_Sharpe"]).round(2).to_string())
    save(out, "increments", increments(F, INCREMENTS, LABELS), "differences from E (all years): rank correlation (DM t), Sharpe (JK z)")
    res, inc, gain = residual(F)
    save(out, "residual", res, "category mean and category mean + residual ridge; differences (DM t, JK z), the out-of-sample R2 "
         "against a zero forecast (R2_bp) and relative to the category mean (bp of the firm-day return y, Newey-West t of the daily "
         "squared-error difference)")
    for s, (lo, hi) in SEGMENTS[a.model].items():
        if lo <= test[-1] and hi >= test[0]:                            # a partial --years run may miss a segment
            save(out, f"residual_{s}", residual(F, lo, hi)[0], f"the same on the signal years {lo}-{hi} (Tables C1, B1)")
    log(f"residual ridge over the category mean: IC {inc['d_IC_%']:+.3f} pp (DM t {inc.DM_t:.2f}), R2 {gain['R2_vs_bp']:+.2f} bp (t {gain['t_mse']:.2f})")
    finish(out, a, log, stories=len(S), categories=len(cats), corr_with_ckx_forecast=float(corr_ckx),
           penalty_differs={str(k): v for k, v in other.items()})


if __name__ == "__main__":
    main()
