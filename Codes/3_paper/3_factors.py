"""Factor exposures of the decile long-short on the forecast and on the parts of its two decompositions (content: yC, yF, yU; timing:
lev, dev), on the Fama-French five factors, momentum and short- and long-term reversal (lib.constants.FACTORS): the portfolios of the
decomposition table (lib.evaluate.ls_weights, ties to the larger firm), valued at the close (common.portfolio.close_valued, the
replication's convention) so that the daily return lines up with the same day's factor returns.
    OMP_NUM_THREADS=8 ../../env/bin/python 3_factors.py --model roberta-large      (after 3_decompose.py)
Output Results/paper/3_factors/<model>/factors.csv + run.json"""
import pandas as pd

from lib.constants import COMP, FACTORS, LABELS, TEST_YEARS, TIMING
from lib.evaluate import factor_alpha, ls_weights
from lib.panel import decomposition, firm_day, sample_rows
from lib.run import finish, save, start

from common.factors import ff_daily
from common.portfolio import close_valued

SIGS = ["yhat"] + COMP + list(TIMING)


def main():
    a, log, out = start("3_factors")
    S = decomposition(a.model, ["rp_story_id", "day0", "permno", "y", "cap", "small", "yr"] + SIGS)
    legs = sample_rows(TEST_YEARS, columns=["intra0", "gap1"])[["rp_story_id", "permno", "intra0", "gap1"]]
    F = firm_day(S.merge(legs, on=["rp_story_id", "permno"], how="left"), SIGS, extra={"intra0": ("intra0", "first"), "gap1": ("gap1", "first")})
    ff = ff_daily()
    fac, cal = ff[FACTORS].dropna(), ff.index                          # close_valued runs on the trading calendar
    rows = []
    for c in SIGS:
        for weight in ["EW", "VW"]:
            fa = factor_alpha(close_valued(ls_weights(F, c, vw=weight == "VW"), cal), fac)
            rows.append({"part": c, "label": LABELS[c], "weight": weight, "days": fa["periods"], "alpha_%/yr": fa["alpha_%/yr"],
                         "t_alpha": fa["t_alpha"], "R2": fa["R2"], **{f"b_{k}": fa[k] for k in fac.columns}})
            log(f"{c} {weight}: alpha {fa['alpha_%/yr']:.1f} (t {fa['t_alpha']:.1f}), R2 {fa['R2']:.2f}")
    save(out, "factors", pd.DataFrame(rows), "EW and VW decile long-short of the forecast and its parts, valued at the close, on the factors")
    finish(out, a, log, firm_days=len(F))


if __name__ == "__main__":
    main()
