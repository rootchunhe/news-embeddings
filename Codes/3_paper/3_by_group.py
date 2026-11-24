"""Where each part of the forecast matters, by event group: the rank IC of the category mean and of the forecast, the
content and timing shares of the forecast's predictive covariance and the group's contribution to the overall EW decile long-short.
    OMP_NUM_THREADS=8 ../../env/bin/python 3_by_group.py --model roberta-large      (after 3_decompose.py)
Output Results/paper/3_by_group/<model>/by_group.csv + run.json"""
import pandas as pd

from lib.constants import COMP, NW_LAGS_DAILY, TEST_YEARS, TIMING
from lib.run import finish, save, start
from lib.category import WINDOWS, category_means, history_rows
from lib.evaluate import contribution_table, ic_series, ls_weights, predictive_cov
from lib.panel import decomposition, firm_day

from common.config import MIN_GROUP_STORIES

PARTS = COMP + list(TIMING)


def decile_weights(S, sig="yhat", q=10):
    """story-level weights of the overall EW decile long-short on firm-days (+-1/n_leg per firm-day, split over its stories)"""
    F = ls_weights(firm_day(S, [sig]), sig, q=q)
    F["w"] = F.w_long - F.w_short
    n_stories = S.groupby(["day0", "permno"]).rp_story_id.transform("size")
    w = S[["day0", "permno"]].merge(F[["day0", "permno", "w"]], on=["day0", "permno"], how="left").w.fillna(0).values
    return w / n_stories.values


def group_rows(S, n_days):
    rows = []
    for g, d in S.groupby("group"):
        if len(d) < MIN_GROUP_STORIES:
            continue
        F = firm_day(d, PARTS + ["yhat", "m8"])
        r = {"group": g, "stories": len(d)}
        for s, lab in [("m8", "category_mean"), ("yhat", "forecast")]:
            r[f"IC_{lab}_%"] = 100 * ic_series(F, s).mean()
        C = predictive_cov(F, PARTS + ["yhat"])
        T = contribution_table(C, PARTS, "yhat", NW_LAGS_DAILY).set_index("part")
        r.update({f"A_{c}": T.share[c] for c in PARTS})
        r["t_forecast"] = T.t["yhat"]
        r["contribution_bp_day"] = 1e4 * (d.w * d.y).sum() / n_days
        rows.append(r)
    return pd.DataFrame(rows).sort_values("stories", ascending=False)


def main():
    a, log, out = start("3_by_group")
    S = decomposition(a.model)
    category_means(S, "m8", WINDOWS["m8"], TEST_YEARS, history=history_rows())
    S["w"] = decile_weights(S)
    n_days = S.loc[S.w != 0, "day0"].nunique()                              # days with a position
    log(f"{len(S):,} stories; long-short {1e4 * (S.w * S.y).sum() / n_days:.2f} bp/day over {n_days:,} days")
    save(out, "by_group", group_rows(S, n_days), "by event group", digits=2)
    finish(out, a, log, stories=len(S), days_with_position=int(n_days))


if __name__ == "__main__":
    main()
