"""Firm-month news indicators: whether the firm has a sample story of each RavenPack event group that month (month of arrival);
groups with stories in fewer than GROUP_MIN_INCIDENCE of the firm-months are merged into one indicator "other". For the event-group
control of 4_didisheim.py. Run from Codes/3_paper/:  ../../env/bin/python 4_groups.py
Output Data/clean/firm_month_groups.npz (permno, month, groups, D) + run_firm_month_groups.json"""
import numpy as np

from lib import ROOT
from lib.constants import FIRST_YEAR, GROUP_MIN_INCIDENCE, PAPER, TEST_YEARS
from lib.panel import sample_rows
from lib.run import finish, start

from common.runinfo import write_manifest

def main():
    a, log, out = start("4_groups", model=False)
    S = sample_rows(range(FIRST_YEAR, TEST_YEARS[-1] + 2), need_return=False)    # Didisheim's sample: no return rule; the extra file holds December arrivals traded in January
    S = S[S.ts_ny.dt.year <= TEST_YEARS[-1]]
    D = S.groupby(["permno", S.ts_ny.dt.strftime("%Y-%m").rename("month"), "group"]).size().unstack(fill_value=0) > 0
    rare = D.columns[D.mean() < GROUP_MIN_INCIDENCE]
    D = D.drop(columns=rare).assign(other=D[rare].any(axis=1))
    fn = ROOT / "Data/clean/firm_month_groups.npz"
    np.savez(fn.with_suffix(".tmp.npz"), permno=D.index.get_level_values(0).values.astype(np.int64), month=D.index.get_level_values(1).values.astype(str),
             groups=np.array(D.columns, dtype=str), D=D.values.astype(np.float32))
    fn.with_suffix(".tmp.npz").replace(fn)
    write_manifest(fn.with_name("run_firm_month_groups.json"), vars(a), spec=PAPER, firm_months=len(D), groups=list(D.columns), merged=list(rare))
    log(f"{len(D):,} firm-months x {D.shape[1]} indicators ({len(rare)} rare groups merged into other) -> {fn}")
    finish(out, a, log, firm_months=len(D), indicators=D.shape[1])


if __name__ == "__main__":
    main()
