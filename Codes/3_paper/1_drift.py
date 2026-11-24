"""Event-time paths after a headline, in excess of the same-day mean over the sample stories with a complete path, by category and for
the top-minus-bottom firm-day decile of the signals: h = 0 is the jump to the first open after the story (from the last close before it;
from the day's open for news during the session, lib.timing), h = 1..H the cumulative open-to-open return from that open (the start of
the forecasts' return window).
Run from Codes/3_paper: ../../env/bin/python 1_drift.py
The same long-short within each Fama-French size group of the firm-day (lib.panel.size_group), over SIZE_H trading days: the days after
H from common.crsp.o2o, the definition of Data/clean's o2o_k (checked on the stored days).
Outputs Results/paper/1_drift/{drift_category,drift_ls,drift_ls_size}.csv"""
import numpy as np
import pandas as pd

from lib.constants import DRIFT_MIN_STORIES, FORECAST_LABEL, H, NW_LAGS_DAILY, PRED, SIZE_H, TEST_YEARS
from lib.run import start, save, finish
from lib.panel import SIZE_GROUPS, firm_day, forecasts, sample_rows, size_group
from lib.evaluate import decile
from lib.category import LABELS as CAT_LABELS, WINDOWS, category_means, direction, history_rows
from lib.timing import immediate_reaction

from common.crsp import daily_panel, o2o
from common.portfolio import nw_tstat

LABELS = {"m8": CAT_LABELS["m8"], **{e: FORECAST_LABEL[m] for e, m in PRED.items()}}
PATH = [f"c_{h}" for h in range(H + 1)]
SIZE_PATH = [f"c_{h}" for h in range(SIZE_H + 1)]


def t_path(d):
    """Newey-West t of each column's mean over days; the lag count grows with the horizon of the cumulative return"""
    return pd.Series({c: nw_tstat(d[c].dropna().values, max(NW_LAGS_DAILY, h)) for h, c in enumerate(PATH)})


def later_days(S):
    """o2o_k of every story for k = H+1..SIZE_H (open of day0+k-1 to open of day0+k), with common.crsp.o2o on the CRSP rows of day0 as
    in 1_data/04_clean.py; the stored o2o_1..o2o_H must be reproduced"""
    cr, _ = daily_panel(["permno", "dlycaldt", "dlyret", "dlyopen", "dlyclose"])
    cr = cr[cr.permno.isin(S.permno.unique()) & (cr.dlycaldt >= S.day0.min())].reset_index(drop=True)   # o2o looks forward only
    g = cr.groupby("permno")
    ks = list(range(1, SIZE_H + 1))
    for k in ks:
        cr[f"x_{k}"] = o2o(cr, k, g)
    M = S[["permno", "day0"]].merge(cr[["permno", "dlycaldt"] + [f"x_{k}" for k in ks]].rename(columns={"dlycaldt": "day0"}),
                                    on=["permno", "day0"], how="left", validate="m:1")
    for k in range(1, H + 1):
        a, b = pd.to_numeric(S[f"o2o_{k}"], errors="coerce").values, M[f"x_{k}"].values
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-6, atol=1e-9), \
            f"o2o_{k}: common.crsp.o2o does not reproduce Data/clean"
    return pd.DataFrame({f"o2o_{k}": M[f"x_{k}"].values for k in range(H + 1, SIZE_H + 1)}, index=S.index)


def size_paths(S, log):
    """the decile long-short of every signal within each size group, h = 0..SIZE_H, on the stories with the whole SIZE_H-day path"""
    T = pd.concat([S, later_days(S)], axis=1)
    post = pd.DataFrame({f"c_{h}": pd.to_numeric(T[f"o2o_{h}"], errors="coerce") for h in range(1, SIZE_H + 1)}).cumsum(axis=1, skipna=False)
    C = pd.concat([T.r_imm.rename("c_0"), post], axis=1)
    keep = C.notna().all(1).values
    T, C = T[keep].reset_index(drop=True), C[keep].reset_index(drop=True)
    R = 1e4 * (C - C.groupby(T.day0.values).transform("mean"))
    F = firm_day(pd.concat([T, R], axis=1), list(LABELS) + SIZE_PATH)
    F["size"] = size_group(F.day0, F.cap)
    rows = []
    for g in SIZE_GROUPS:                                                  # deciles within the group's names of the day
        Fg = F[F["size"] == g]
        for s, lab in LABELS.items():
            d = Fg.assign(decile=decile(Fg, s))
            daily = d.groupby(["day0", "decile"])[SIZE_PATH].mean()
            ls = (daily.xs(10, level="decile") - daily.xs(1, level="decile")).dropna()
            t = pd.Series({c: nw_tstat(ls[c].dropna().values, max(NW_LAGS_DAILY, h)) for h, c in enumerate(SIZE_PATH)})
            rows += [{"size": g, "signal": lab, "h": h, "ls_bp": ls[c].mean(), "t": t[c], "days": len(ls)} for h, c in enumerate(SIZE_PATH)]
        log(f"size {g}: {len(Fg):,} of {len(F):,} firm-days with a {SIZE_H}-day path")
    return pd.DataFrame(rows)


def main():
    a, log, out = start("1_drift", model=False)
    S0 = sample_rows(TEST_YEARS, columns=["gap0"] + [f"o2o_{h}" for h in range(1, H + 1)])
    S = immediate_reaction(forecasts(S0))
    assert len(S) == len(S0) and S[list(PRED)].notna().all().all(), "stories without exactly one forecast of each main encoder"
    category_means(S, "m8", WINDOWS["m8"], TEST_YEARS, history=history_rows())
    S = S.dropna(subset=["m8"]).reset_index(drop=True)                # the same firm-days for every signal
    post = pd.DataFrame({f"c_{h}": pd.to_numeric(S[f"o2o_{h}"], errors="coerce") for h in range(1, H + 1)}).cumsum(axis=1, skipna=False)
    C = pd.concat([S.r_imm.rename("c_0"), post], axis=1)
    keep = C.notna().all(1).values                                     # the same stories at every horizon, also in the benchmark
    S, C = S[keep].reset_index(drop=True), C[keep].reset_index(drop=True)
    R = 1e4 * (C - C.groupby(S.day0.values).transform("mean"))        # excess of the same-day mean over these stories, bp
    log(f"{len(S):,} stories with an 8-year category mean, a jump and a complete {H}-day path")

    rows = []
    for cat, idx in S.groupby("category").indices.items():
        if len(idx) < DRIFT_MIN_STORIES:
            continue
        d = R.iloc[idx].groupby(S.day0.values[idx]).mean()                # each day equally weighted
        t = t_path(d)
        rows += [{"category": cat, "group": S.group.iloc[idx[0]], "direction": direction(cat), "h": h, "car_bp": d[c].mean(), "t": t[c],
                  "stories": len(idx)} for h, c in enumerate(PATH)]
    F = firm_day(pd.concat([S, R], axis=1), list(LABELS) + PATH)
    ls_rows = []
    for s, lab in LABELS.items():
        d = F.assign(decile=decile(F, s))                                  # NaN outside the days with >= MIN_NAMES names
        daily = d.groupby(["day0", "decile"])[PATH].mean()
        ls = (daily.xs(10, level="decile") - daily.xs(1, level="decile")).dropna()
        t = t_path(ls)
        ls_rows += [{"signal": lab, "h": h, "ls_bp": ls[c].mean(), "t": t[c]} for h, c in enumerate(PATH)]
    cat = save(out, "drift_category", pd.DataFrame(rows))
    log(f"{cat.category.nunique()} categories with >= {DRIFT_MIN_STORIES} stories; {len(F):,} firm-days")
    save(out, "drift_ls", pd.DataFrame(ls_rows), "top minus bottom decile path (bp)", digits=1)
    save(out, "drift_ls_size", size_paths(S, log), f"top minus bottom decile path within each size group, {SIZE_H} days (bp)", digits=1)
    finish(out, a, log, stories=len(S))


if __name__ == "__main__":
    main()
