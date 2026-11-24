"""Didisheim–Kelly–Pourmohammadi–Tian, *The Inefficient Pricing of News*, step 2: the paper's monthly analysis on our firm-month
headline embeddings (the functions are in newsshock.py). Run from Codes/2_replication/didisheim/:
    ../../../env/bin/python 02_msrr.py --model roberta-large                                   # replication period 2000-01..2022-12 (signal months)
    ../../../env/bin/python 02_msrr.py --model roberta-large --end 2025-12 --results <...>/Results/replication/didisheim_extended
Figure 4   MSRR on the z-scored embedding: raw, and residualized month by month on a constant, on the characteristics behind CAPM,
           FF3 and FF6 (newsshock.FF_CHARS), on the 132 highest-coverage JKP characteristics, on GICS industry groups, or on both; with
           the predicted part of every residualization and the control MSRR on the characteristics. Figure 7: the same within the JKP
           size groups (mega + large, small + micro).
Table 7    quintile sorts, equal- and capped value-weighted, on the MSE forecast (expanding-window ridge, refit every year once 36
           months are available, with the GCV penalty, of the next-month return on the news shock; Panel A) and on the stocks' MSRR
           weights (Panel B), both on the months of Panel A.
Figure 10  trading delay tau = 1..36 months against the full JKP panel, weights fixed at the undelayed strategy's (the paper's Panel A:
           Sharpe, mean and CAPM alpha) and re-optimized (Sharpe).
The news shock of Table 7 and Figures 7 and 10 is the residual on the JKP characteristics (the paper's main specification). Sharpe
annualized; means and CAPM alphas at 10% annual volatility with Newey-West t; the return of signal month t is earned in t+1: JKP's
excess return, not winsorized (also Table 7's MSE target).
Output <results>/<model>/ (staged, then published): F04_msrr.csv, F04_returns.csv, F07_size.csv, F07_returns.csv, T07_sorts.csv,
F10_lags.csv, lag_returns.csv, lag_returns_fixed.csv, weights.npz (the undelayed MSRR weights by month), run.json (with
the characteristics). The paper's variants of this pipeline are 3_paper/4_didisheim.py."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.runinfo import logger, publish, require_same_sample, staging_dir, write_manifest
from newsshock import figure4, lag_profile, load_panel, mse_scores, quintile_sorts, recursive_zscore


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--start", default="2000-01")
    p.add_argument("--end", default="2022-12")                             # their sample ends in December 2022
    p.add_argument("--n_chars", type=int, default=132)
    p.add_argument("--lags", default=",".join(map(str, range(1, 37))))
    p.add_argument("--results", default=str(ROOT / "Results/replication/didisheim"))
    a = p.parse_args()
    a.model = a.model.split("/")[-1]
    changed = [k for k in ["start", "end", "n_chars", "lags"] if getattr(a, k) != p.get_default(k)]
    assert not changed or a.results != p.get_default("results"), f"non-default {changed}: give --results, the default folder holds the replication period"
    return a


def main():
    a = parse()
    T0, log = time.time(), logger()
    require_same_sample(ROOT / f"Data/clean/run_firm_month_{a.model}.json", "firm-month embeddings")
    out = staging_dir(Path(a.results) / a.model)
    P = load_panel(a.model, "", a.start, a.end, n_chars=a.n_chars)
    log(f"firm-months with news {P.keys:,}; merged {len(P.M):,} over {P.n_months} months; stocks per month {P.M.groupby('t').permno.nunique().mean():.0f}; "
        f"JKP characteristics {len(P.chars)}")
    Z = recursive_zscore(P.E, P.M.t.values, P.n_months)
    log("z-score done")

    f4 = figure4(Z, P, log)
    for name, df in [("F04_msrr", f4.table), ("F07_size", f4.size)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(df.round(2).to_string(index=False), flush=True)
    f4.returns.to_csv(out / "F04_returns.csv")
    f4.size_returns.to_csv(out / "F07_returns.csv")
    Xs = f4.shock

    lags, opt, fix, B0 = lag_profile(Xs, P, [0] + [int(x) for x in a.lags.split(",")])
    for _, r in lags.iterrows():
        log(f"tau={int(r.tau)}: Sharpe re-optimized {r.Sharpe:.2f}, weights fixed at tau 0 {r.Sharpe_fixed_b:.2f}")
    lags.to_csv(out / "F10_lags.csv", index=False)
    opt.to_csv(out / "lag_returns.csv")
    fix.to_csv(out / "lag_returns_fixed.csv")

    mse = mse_scores(Xs, P.M)
    msrr_w = np.full(len(P.M), np.nan)                               # each stock's weight in the MSRR portfolio of its month
    for t, idx in P.M.groupby("t").indices.items():
        msrr_w[idx] = Xs[idx].astype(np.float64) @ B0[t]
    msrr_w[np.isnan(mse)] = np.nan                                   # Panel B on the months of Panel A (the MSE forecast needs 36 months)
    T07 = pd.concat([quintile_sorts(mse, P.M, P.months, P.mkt).assign(panel="MSE"), quintile_sorts(msrr_w, P.M, P.months, P.mkt).assign(panel="MSRR")])
    T07.to_csv(out / "T07_sorts.csv", index=False)
    print(T07[T07.portfolio == "H-L"].round(3).to_string(index=False), flush=True)
    np.savez(out / "weights.npz", months=np.array(P.months), msrr=B0)   # the strategy's weights, for the paper's attribution (3_paper/4_didisheim.py)
    write_manifest(out, vars(a), spec=("monthly",), firm_months=len(P.M), months=P.n_months, characteristics=P.chars, seconds=round(time.time() - T0))
    publish(out)
    log("ALL DONE")


if __name__ == "__main__":
    main()
