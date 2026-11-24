"""Is the delayed news-shock return annual seasonality? Run from Codes/2_replication/didisheim/ after 02_msrr.py:
    ../../../env/bin/python 04_seasonality.py --models roberta-large,Llama-2-13b-hf
Seasonal factor (Heston-Sadka): each month, stocks sorted on their average excess return in the same calendar month of the
previous five years (at least three observations), EW top-minus-bottom decile, from the JKP monthly panel. The lag-tau MSRR
returns (lag_returns.csv: weights re-optimized per delay; lag_returns_fixed.csv: weights of the undelayed strategy; indexed by formation month,
earned in the month after it) are regressed on the market and the seasonal factor; alpha at 10% vol, Newey-West t.
Output Results/replication/didisheim/<model>/seasonality.csv"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Codes"))
from common.config import NW_LAGS_MONTHLY
from common.factors import nw_ols
from common.jkp import monthly_market, monthly_panel, monthly_sharpe
from common.runinfo import require_same_sample, write_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="roberta-large,Llama-2-13b-hf")
    p.add_argument("--results", default=str(ROOT / "Results/replication/didisheim"))
    a = p.parse_args()

    J = monthly_panel(columns=["permno", "eom", "ret_exc"])
    J = J[J.ret_exc.notna()].copy()
    J["month"] = pd.PeriodIndex(J.month, freq="M")
    W = J.pivot(index="month", columns="permno", values="ret_exc").astype(float).sort_index()
    W = W.reindex(pd.period_range(W.index.min(), W.index.max(), freq="M"))
    lagged = [W.shift(12 * k) for k in range(1, 6)]
    count = sum(x.notna().astype(float) for x in lagged)
    hist = sum(x.fillna(0.0) for x in lagged) / count.replace(0, np.nan)
    hist = hist.where(count >= 3)
    seasonal = {}
    for m in W.index:
        h, r = hist.loc[m], W.loc[m]
        ok = h.notna() & r.notna()
        if ok.sum() < 200:
            continue
        q = pd.qcut(h[ok].rank(method="first"), 10, labels=False)
        seasonal[m] = r[ok][q == 9].mean() - r[ok][q == 0].mean()
    seasonal = pd.Series(seasonal)
    print(f"seasonal factor: {len(seasonal)} months, Sharpe {monthly_sharpe(seasonal):.2f}", flush=True)
    mkt = monthly_market()
    mkt.index = pd.PeriodIndex(mkt.index, freq="M")

    for m in [x.split("/")[-1] for x in a.models.split(",")]:
        rows = []
        require_same_sample(Path(a.results) / m / "run.json", f"Didisheim replication of {m}", needs=("monthly",))
        for fname, weights in [("lag_returns.csv", "re-optimized"), ("lag_returns_fixed.csv", "fixed at tau 0")]:
            L = pd.read_csv(Path(a.results) / m / fname, index_col=0)
            L.index = pd.PeriodIndex(L.index, freq="M") + 1          # indexed by formation month: earned in the month after it
            for tau in L.columns:
                d = pd.DataFrame({"r": L[tau], "mkt": mkt.reindex(L.index), "seas": seasonal.reindex(L.index)}).dropna()   # one sample for every statistic
                d["r"] = d.r / (d.r.std() * np.sqrt(12)) * 0.10
                b1, t1, _ = nw_ols(d.r.values, d[["mkt"]].values, NW_LAGS_MONTHLY)
                b2, t2, _ = nw_ols(d.r.values, d[["mkt", "seas"]].values, NW_LAGS_MONTHLY)
                rows.append({"weights": weights, "tau": int(tau), "months": len(d), "Sharpe": monthly_sharpe(d.r), "corr_with_seasonal": d.r.corr(d.seas),
                             "CAPM_alpha_%/yr": 1200 * b1[0], "t_CAPM": t1[0], "alpha_CAPM+seasonal_%/yr": 1200 * b2[0], "t": t2[0],
                             "beta_seasonal": b2[2], "t_beta_seasonal": t2[2]})
        T = pd.DataFrame(rows)
        T.to_csv(Path(a.results) / m / "seasonality.csv", index=False)
        write_manifest(Path(a.results) / m / "run_seasonality.json", vars(a), spec=("monthly",), model=m, seasonal_months=len(seasonal))
        print("---", m)
        print(T.round(2).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
