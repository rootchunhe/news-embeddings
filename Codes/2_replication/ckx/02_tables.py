"""The CKX tables implemented here, from the out-of-sample forecasts, named by the paper's table / figure numbers (2026 version unless marked v2024).
Run from Codes/2_replication/ckx/ (CPU, ~40 min per model):
    ../../../env/bin/python 02_tables.py --models Llama-2-13b-hf,roberta-large
Periods (trade days): paper = 2004-01..2019-06 (the paper's sample ends June 2019), extended = 2004-2025 (the paper analyses' test
years).
Per model and period, Results/replication/ckx/<model>/<period>/ (written to a staging folder, published on success) receives
  counts.csv              stories, firms, days per year
  T03_correlation.csv     daily cross-sectional Pearson correlation: mean, NW t, pooled, above NYSE median, Spearman, by year
  T05_quintile.csv        quintile long / short / long-short, EW and VW: annualized return and std in percent, Sharpe
  T06_decile.csv          same for deciles
  T08_ewct.csv            transaction costs with EWCT turnover control (common/portfolio.ewct)
  T13_cutoff.csv          EW quintile before / after the model's knowledge cutoff (extended period only)
  T17_reversal.csv        quintile sorts on the past 5-day return: all CRSP common stocks vs the stocks with news, over the same period
  T18_delay.csv           the day-0 quintiles held with delay k = 1..10 (fixed composition), EW and VW
  F09_double_sort.csv     5x5 sort on past 5-day return and forecast, EW mean daily return in bp
  F10_size_delay.csv      EW long-short at delay 1..6 for small / large (NYSE median), bp per day with its Newey-West t
  T20_21_alphas.csv       close-valued long-short returns on DMRS hedged factors + q5 + momentum + reversals (EW, VW), OLS SE;
                          through the end of the DMRS series (2023Q1)
  F05_cumret.csv          cumulative log returns of the legs and the market
  v2024_T05_accuracy.csv, v2024_T06_sentiment.csv   sentiment (logistic) model, when it was estimated
  returns_daily.csv       daily leg and long-short returns (for the notebook's plots)
Cross-model (Table 4 Diebold-Mariano, row RoBERTa, column Llama-2, positive = the column model better; Figure 7 strategy correlation), one row per period:
Results/replication/ckx/cross_model.csv + run_cross_model.json."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Codes"))
from common.config import MIN_NAMES, MIN_NAMES_DOUBLE_SORT, MIN_NAMES_LARGE
from common.portfolio import ANN, KMAX, close_valued, ewct_table, legs, market_data, nw_tstat, sharpe, weights
from common.runinfo import logger, publish, require_same_sample, staging_dir, write_manifest
from common.universe import sample_counts

CUTOFF = {"Llama-2-13b-hf": "2022-09-01", "roberta-large": "2019-07-31"}
PERIODS_DEFAULT = "paper:2004-01-01:2019-06-30,extended:2004-01-01:2025-12-31"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="Llama-2-13b-hf,roberta-large")
    p.add_argument("--periods", default=PERIODS_DEFAULT, help="name:start:end,... (the paper's tables read paper and extended)")
    p.add_argument("--results", default=str(ROOT / "Results/replication/ckx"))
    p.add_argument("--pred_suffix", default="", help="read pred_<model><suffix>.parquet (test runs)")
    a = p.parse_args()
    assert not (a.pred_suffix or a.periods != PERIODS_DEFAULT) or a.results != p.get_default("results"), "test runs need their own --results"
    return a


# ---- tables -----------------------------------------------------------------------------------------------------------
def stats3(r):
    """annualized mean return and standard deviation in percent, annualized Sharpe ratio (the rows of CKX's Tables 5-6)"""
    r = pd.Series(r).dropna()
    return {"Ret_%/yr": 100 * r.mean() * ANN, "Std_%/yr": 100 * r.std() * np.sqrt(ANN), "SR": sharpe(r)}


def leg_table(sd, score, q):
    cols = {}
    for vw in [False, True]:
        r = legs(weights(sd, score, q, vw, MIN_NAMES))
        for leg, s in [("Long", r.L), ("Short", r.S), ("L-S", r.LS)]:
            cols[("VW" if vw else "EW", leg)] = stats3(s)
    return pd.DataFrame(cols).round(4)


def correlation_table(sd):
    d = sd.dropna(subset=["yhat", "o2o_1"])
    def daily(x, n, **kw):                                              # the day's correlation, days with at least n names
        return x.groupby("day0").apply(lambda g: g.yhat.corr(g.o2o_1, **kw) if len(g) >= n else np.nan).dropna()
    by_day, sp, big_day = daily(d, MIN_NAMES), daily(d, MIN_NAMES, method="spearman"), daily(d[d.cap_prev >= d.nyse50], MIN_NAMES_LARGE)
    rows = {"Mean (Pearson) %": 100 * by_day.mean(), "t_nw": nw_tstat(by_day), "Pooled %": 100 * d.yhat.corr(d.o2o_1),
            "Over NYSE median %": 100 * big_day.mean(), "Mean (Spearman) %": 100 * sp.mean()}
    for y, s in by_day.groupby(by_day.index.year):
        rows[str(y)] = 100 * s.mean()
    return pd.Series(rows).to_frame("value").round(3)


def accuracy_table(st):
    s = st.dropna(subset=["label_pos"])
    rows = []
    for y, x in s.groupby("year"):
        hit = (x.p_sent > 0.5) == (x.label_pos == 1)
        top, bot = x.p_sent >= x.p_sent.quantile(0.8), x.p_sent <= x.p_sent.quantile(0.2)
        rows.append({"year": y, "accuracy_%": 100 * hit.mean(), "top20_%": 100 * hit[top].mean(), "bottom20_%": 100 * hit[bot].mean(), "n": len(x)})
    t = pd.DataFrame(rows).set_index("year")
    hit = (s.p_sent > 0.5) == (s.label_pos == 1)
    t.loc["overall"] = [100 * hit.mean(), np.nan, np.nan, len(s)]
    return t.round(3)


def reversal_table(sd, mk):
    out = {}
    m = mk.MKT[(mk.MKT.day0 >= sd.day0.min()) & (mk.MKT.day0 <= sd.day0.max())]
    for name, d in [("Entire market", m), ("Stocks with news", sd)]:
        d = d.dropna(subset=["past5"])
        for vw in [False, True]:
            r = legs(weights(d, "past5", 0.2, vw, MIN_NAMES))
            for leg, s in [("Long", r.L), ("Short", r.S), ("L-S", r.LS)]:
                out[(name, "VW" if vw else "EW", leg)] = stats3(s)
    return pd.DataFrame(out).round(4)


def delay_table(sd):
    rows = []
    for vw in [False, True]:
        W = weights(sd, "yhat", 0.2, vw, MIN_NAMES)
        for k in range(1, KMAX + 1):
            r = legs(W, f"o2o_{k}")
            rows.append({"delay": k, "weight": "VW" if vw else "EW", "LS_bp": 1e4 * r.LS.mean(), "LS_t_nw": nw_tstat(r.LS), **{f"{leg}_SR": sharpe(r[c]) for leg, c in [("Long", "L"), ("Short", "S"), ("LS", "LS")]}})
    return pd.DataFrame(rows).set_index(["delay", "weight"]).round(3)


def double_sort(sd, n=5):
    d = sd.dropna(subset=["past5", "yhat", "o2o_1"]).copy()
    d = d[d.groupby("day0").permno.transform("size") >= MIN_NAMES_DOUBLE_SORT]
    for col, name in [("past5", "q_past"), ("yhat", "q_sig")]:
        d[name] = d.groupby("day0")[col].rank(pct=True, method="first").mul(n).apply(np.ceil).clip(1, n).astype(int)
    c = d.groupby(["day0", "q_past", "q_sig"]).o2o_1.mean().unstack()
    c["L-S (sig5 - sig1)"] = c[n] - c[1]                                    # the daily spread, then its mean
    m = c.groupby(level="q_past").mean() * 1e4
    m.index.name, m.columns.name = "past5 rank", "forecast rank"
    return m.round(1)


def size_delay(sd, kmax=6):
    rows = []
    for size, d in [("small", sd[sd.cap_prev < sd.nyse50]), ("large", sd[sd.cap_prev >= sd.nyse50])]:
        W = weights(d, "yhat", 0.2, False, MIN_NAMES)
        for k in range(1, kmax + 1):
            ls = legs(W, f"o2o_{k}").LS
            rows.append({"size": size, "delay": k, "LS_bp": 1e4 * ls.mean(), "t_nw": nw_tstat(ls)})
    return pd.DataFrame(rows).set_index(["size", "delay"]).round(2)


def alpha_table(sd, fac, cols, tdays):
    """Tables 20-21: the close-valued long-short on the factors, OLS standard errors as in the paper; the alpha and the average
    return annualized"""
    rows = []
    for vw in [False, True]:
        r = close_valued(weights(sd, "yhat", 0.2, vw, MIN_NAMES), tdays)
        d = pd.DataFrame({"r": r}).join(fac[cols], how="inner").dropna()
        X = np.column_stack([np.ones(len(d)), d[cols].values])
        b, *_ = np.linalg.lstsq(X, d.r.values, rcond=None)
        e = d.r.values - X @ b
        n, k = X.shape
        r2 = 1 - e.var() / d.r.var(ddof=0)
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (n - k))
        rows.append({"weight": "VW" if vw else "EW", "days": n, "avg_ret_ann_%": 100 * d.r.mean() * ANN, "alpha_ann_%": 100 * b[0] * ANN,
                     "se_alpha_ann_%": 100 * se[0] * ANN, **{f"b_{c}": b[i + 1] for i, c in enumerate(cols)},
                     **{f"se_{c}": se[i + 1] for i, c in enumerate(cols)}, "R2": r2, "adj_R2": 1 - (1 - r2) * (n - 1) / (n - k)})
    return pd.DataFrame(rows).set_index("weight").round(4)


def returns_daily(sd, mk):
    out = {}
    for vw in [False, True]:
        W = weights(sd, "yhat", 0.2, vw, MIN_NAMES)
        r = legs(W)
        tag = "VW" if vw else "EW"
        out[f"{tag}_L"], out[f"{tag}_S"], out[f"{tag}_LS"] = r.L, r.S, r.LS
        out[f"{tag}_LS_close_valued"] = close_valued(W, mk.TDAYS)
    mkt = mk.FF.mktrf + mk.FF.rf
    out["market"] = mkt[(mkt.index >= sd.day0.min()) & (mkt.index <= sd.day0.max())]
    return pd.DataFrame(out)


def dm_test(A, B, q=0.2):
    """Diebold-Mariano on the daily MSE of the two models' forecasts of the same firm-days: each day, those in the top or bottom
    quintile of either model's forecast (the paper's Table 4: "for top and bottom quintile predictions"); positive = the second
    (column) model better, as in the paper"""
    d = A[["day0", "permno", "yhat", "o2o_1"]].merge(B[["day0", "permno", "yhat"]], on=["day0", "permno"], suffixes=("_a", "_b")).dropna()
    g = d.groupby("day0")
    pa, pb = g.yhat_a.rank(pct=True), g.yhat_b.rank(pct=True)
    d = d[(pa > 1 - q) | (pa <= q) | (pb > 1 - q) | (pb <= q)]
    dd = ((d.yhat_a - d.o2o_1) ** 2 - (d.yhat_b - d.o2o_1) ** 2).groupby(d.day0).mean()
    return nw_tstat(dd), len(dd)


# ---- per model ------------------------------------------------------------------------------------------------------
def main():
    a = parse()
    R = Path(a.results)
    PERIODS = [(s.split(":")[0], pd.Timestamp(s.split(":")[1]), pd.Timestamp(s.split(":")[2])) for s in a.periods.split(",")]
    log = logger()
    mk = market_data(log)
    for m in a.models.split(","):
        require_same_sample(ROOT / f"Data/clean/run_pred_{m}{a.pred_suffix}.json", f"forecasts of {m}", needs=("ridge",))
    SD = {}
    cols = ["rp_story_id", "permno", "day0", "label_pos", "intra0", "gap1", "dlycap"] + [f"o2o_{k}" for k in range(1, KMAX + 1)]   # dlycap in Data/clean = cap at the close of day0-1
    clean = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((ROOT / "Data/clean").glob("clean_*.parquet"))], ignore_index=True)
    clean["day0"] = pd.to_datetime(clean.day0)
    for m in a.models.split(","):
        pred = pd.read_parquet(ROOT / f"Data/clean/pred_{m}{a.pred_suffix}.parquet", columns=["rp_story_id", "permno", "day0", "yhat"])
        sent = ROOT / f"Data/clean/sent_{m}{a.pred_suffix}.parquet"   # the logistic model (01_regress --logit), for the v2024 tables
        has_sent = sent.exists()
        if has_sent:
            require_same_sample(sent.with_name(f"run_sent_{m}{a.pred_suffix}.json"), f"sentiment forecasts of {m}", needs=("ridge",))
            pred = pred.merge(pd.read_parquet(sent, columns=["rp_story_id", "permno", "day0", "p_sent"]), on=["rp_story_id", "permno", "day0"], how="left", validate="1:1")
            assert pred.p_sent.notna().all(), "ridge forecasts without a sentiment forecast"
        pred["day0"] = pd.to_datetime(pred.day0)
        st = pred.merge(clean, on=["rp_story_id", "permno", "day0"], how="left", validate="1:m", indicator=True)   # clean holds other rows on one key (multi-company stories)
        assert len(st) == len(pred) and (st._merge == "both").all(), "a forecast without exactly one clean row"
        st = st.drop(columns="_merge")
        st = st.rename(columns={"dlycap": "cap_prev"})                # the forecasts are of the sample rows only (01_regress)
        st["year"] = st.day0.dt.year
        sample_counts(st, m)
        agg = {"yhat": ("yhat", "mean"), "year": ("year", "first")} | ({"p_sent": ("p_sent", "mean")} if has_sent else {})
        agg.update({c: (c, "first") for c in ["intra0", "gap1", "cap_prev"] + [f"o2o_{k}" for k in range(1, KMAX + 1)]})
        sd = st.groupby(["day0", "permno"]).agg(**agg).reset_index()
        sd = sd.merge(mk.MKT[["permno", "day0", "past5"]], on=["permno", "day0"], how="left").merge(mk.BP[["nyse50"]], left_on="day0", right_index=True, how="left")
        SD[m] = sd
        for name, lo, hi in PERIODS:
            D = sd[(sd.day0 >= lo) & (sd.day0 <= hi)]
            S = st[(st.day0 >= lo) & (st.day0 <= hi)]
            if len(D) == 0:
                continue
            out = staging_dir(R / m / name)
            tabs = {"counts": lambda: S.groupby("year").agg(stories=("rp_story_id", "size"), firms=("permno", "nunique"), days=("day0", "nunique"), months=("day0", lambda x: x.dt.to_period("M").nunique())).assign(firms_per_day=D.groupby("year").apply(lambda x: x.groupby("day0").permno.nunique().mean())),
                    "T03_correlation": lambda: correlation_table(D), "T05_quintile": lambda: leg_table(D, "yhat", 0.2), "T06_decile": lambda: leg_table(D, "yhat", 0.1),
                    "T08_ewct": lambda: ewct_table(weights(D, "yhat", 0.2, False, MIN_NAMES), mk), "T17_reversal": lambda: reversal_table(D, mk), "T18_delay": lambda: delay_table(D),
                    "F09_double_sort": lambda: double_sort(D), "F10_size_delay": lambda: size_delay(D),
                    "T20_21_alphas": lambda: alpha_table(D, mk.ALLF, ["MktRF_DMRS", "HML_DMRS", "RMW_DMRS", "CMA_DMRS", "SMB_DMRS", "R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG", "ltrev", "strev", "umd"], mk.TDAYS)}
            if has_sent:
                tabs["v2024_T05_accuracy"] = lambda: accuracy_table(S)
                tabs["v2024_T06_sentiment"] = lambda: leg_table(D, "p_sent", 0.2)
            if name == "extended" and m in CUTOFF:
                def cutoff():
                    c = pd.Timestamp(CUTOFF[m])
                    res = {}
                    for lab, dd in [("all", D), ("pre cutoff", D[D.day0 < c]), ("post cutoff", D[D.day0 >= c])]:
                        r = legs(weights(dd, "yhat", 0.2, False, MIN_NAMES))
                        for leg, s in [("Long", r.L), ("Short", r.S), ("L-S", r.LS)]:
                            res[(lab, leg)] = stats3(s)
                    return pd.DataFrame(res).round(4)
                tabs["T13_cutoff"] = cutoff
            for k, fn in tabs.items():
                fn().to_csv(out / f"{k}.csv")
            rd = returns_daily(D, mk)
            rd.to_csv(out / "returns_daily.csv")
            np.log1p(rd[["EW_L", "EW_S", "EW_LS", "VW_L", "VW_S", "VW_LS", "market"]].fillna(0)).cumsum().to_csv(out / "F05_cumret.csv")
            write_manifest(out, vars(a), spec=("ridge",), model=m, period=name)
            publish(out)
            log(f"{m}/{name}: {len(tabs)} tables written")

    # ---- cross-model: one table, a row per period -----------------------------------------------------------------------
    ms = ["roberta-large", "Llama-2-13b-hf"]                            # the paper's row and column order (RoBERTa row, LLaMA2 column)
    if set(ms) <= set(SD):
        rows = []
        for name, lo, hi in PERIODS:
            sub = {m: SD[m][(SD[m].day0 >= lo) & (SD[m].day0 <= hi)] for m in ms}
            if any(len(s) == 0 for s in sub.values()):
                continue
            dm, n = dm_test(sub[ms[0]], sub[ms[1]])
            row = {"period": name, "row_model": ms[0], "col_model": ms[1], "DM_stat": dm, "n_days": n}
            for vw in [False, True]:
                rets = [legs(weights(sub[m], "yhat", 0.2, vw, MIN_NAMES)).LS for m in ms]
                row[f"strategy_corr_{'VW' if vw else 'EW'}"] = rets[0].corr(rets[1])
            rows.append(row)
        fn = R / "cross_model.csv"
        pd.DataFrame(rows).to_csv(fn.with_suffix(".tmp"), index=False)
        fn.with_suffix(".tmp").replace(fn)
        write_manifest(R / "run_cross_model.json", vars(a), spec=("ridge",), periods=[r["period"] for r in rows])
    log("ALL DONE")


if __name__ == "__main__":
    main()
