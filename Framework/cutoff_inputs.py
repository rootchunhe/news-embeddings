"""Data inputs of the look-ahead section and the calibration (framework.tex), computed through the project's code (read
only) and written next to this file; calibration.py reads them. Run from Codes/3_paper/ (it imports the project's modules):
    ../../env/bin/python ../../Framework/cutoff_inputs.py
Outputs
    cutoff_daily.csv.gz  per encoder and trading day, on Table 11's firm-days (4_horizon: the whole return path): the next-day rank
                         IC (open-to-open return of day 1) and the cross-sectional standard deviation of F, I_L, E_L, E_D and C_D
                         of the CKX daily forecast
    cutoff_inputs.json
      availability       means of A, (1 - A) H, (1 - A)(1 - H) and H, row by row, over Table 11's firm-days and over DKPT's
                         arrival-month firm-months with a next-month return. A: the firm has a firm effect in its test year's
                         projection (lib.ridge.firm_years, identified); H: a historical level. A firm constant sits in F with
                         A = 1, otherwise in I_L (H = 1) or I_D
      firm_month_inv_n   mean of 1 / n over the same firm-months, n the firm-month's news items
      return_variance    cross-sectional variance of WINDOW-month sums of news firms' monthly log returns: months with at least 15
                         daily returns, daily returns floored at -99%, demeaned and winsorized at 1/99 within month; firms with a
                         firm-day in that calendar year and returns in every month; non-overlapping windows from January 2004
      lagged_breadth     DKPT's standalone strategy on the level lagged LEVEL_LAG months, signal months 2005-01..2025-11: the
                         firm-months with a lagged level over those with a current level input (A or H); history from all news
                         months
      projection_years   first and last year of the two-way projection of every test year
      n_train            news items of the ridge's training window, mean over the test years with TRAIN_YEARS training years
      firm_valuation     Table 4's December windows: inv_n, the mean of 1 / N over the table's firms with log P/B (N: news items in
                         the window; mean within December, then over Decembers), and window_months"""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
spec = importlib.util.spec_from_file_location("h4", os.path.join(os.getcwd(), "4_horizon.py"))
h4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h4)
from lib.constants import H, LEVEL_LAG, PROJECTION_YEARS, TEST_YEARS   # noqa: E402
from lib.firms import decembers, december_windows   # noqa: E402
from lib.evaluate import ic_series               # noqa: E402
from lib.panel import decomposition, firm_day   # noqa: E402
from lib.monthly import jkp_returns, trailing_level              # noqa: E402
import lib                                       # noqa: E402
from lib.ridge import _cached, _key, category_index, firm_years, identified, replication_fit, windows   # noqa: E402
from common.config import FIRST_YEAR, TRAIN_YEARS   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path(os.getcwd()).resolve().parents[1]           # the project root (run from Codes/3_paper)
MODELS = ("chrono-gpt-v1-19991231", "roberta-large", "Llama-2-13b-hf")
PARTS = ("yF", "uL", "lev", "dev", "cD")                # cD: no memory in the model, a check on the dispersion change
WINDOW = 36                                          # months of the return-variance window


def panel(model):
    """Table 11's firm-days of the encoder: headline signals averaged to firm-days, kept when the whole return path exists"""
    F = firm_day(h4.load(model), list(h4.SIGS) + ["yhat"], extra={f"r_{h}": (f"r_{h}", "mean") for h in range(0, H + 1)})
    return F.dropna(subset=[f"r_{h}" for h in range(0, H + 1)])


def daily(model, F):
    ic = {k: ic_series(F, k, ycol="r_1") for k in PARTS}
    sd = F.groupby("day0")[list(PARTS)].std()
    out = pd.DataFrame({**{f"ic_{k}": v for k, v in ic.items()}, **{f"sd_{k}": sd[k] for k in PARTS}})
    return out.rename_axis("day0").reset_index().assign(enc=model)


def projection_firms(model):
    """the firms with a firm effect in the two-way projection of every test year T: lib.ridge.identified on the firm statistics of
    firm_years(T) (news items and years with news)"""
    cats, d = category_index(), lib.WORK / "ridge_stats" / model
    firms = {}
    for y in sorted({y for T in TEST_YEARS for y in firm_years(T)}):
        st = _cached(d / f"{y}_firm.npz", _key(model, y, cats, "firm"), ["permno", "N"])
        assert st is not None, f"{model} {y}: no current firm statistics (run 3_decompose.py)"
        firms[y] = (st["permno"], st["N"].sum(1))
    out = {}
    for T in TEST_YEARS:
        yrs = firm_years(T)
        p = np.concatenate([firms[y][0] for y in yrs])
        n = np.concatenate([firms[y][1] for y in yrs])
        u, inv = np.unique(p, return_inverse=True)
        nF, nY = np.bincount(inv, n), np.bincount(inv)
        out[T] = set(u[identified(nF, nY, len(yrs))].tolist())
    return out


def n_train(model):
    """the news items of the ridge's training window (lib.ridge.replication_fit, after its sample check), mean over the test years with
    TRAIN_YEARS training years"""
    fits = replication_fit(model)[2]
    return float(np.mean([f["n_train"] for T, f in fits.items() if len(windows(T)[0]) == TRAIN_YEARS]))


def firm_valuation(model):
    """Table 4's December windows: the mean of 1 / N over the firms with a firm effect in the window (lib.ridge.identified; N their
    news items), the table's characteristics and log P/B at that December (mean within December, then over Decembers), and the months of
    news in each window"""
    D = decembers()
    D = D[D.log_pb.notna()]
    inv, months = [], []
    for _, dec, _, fst in december_windows(model, TEST_YEARS, firsts={FIRST_YEAR}):
        permno, n = fst[0], fst[2].sum(1)
        n = n[identified(n, fst[4], fst[5]) & np.isin(permno, D.permno[D.year == dec].values)]
        inv.append(float(np.mean(1 / n)))
        months.append(12 * (dec - max(FIRST_YEAR, dec - PROJECTION_YEARS + 1) + 1))
    return {"inv_n": float(np.mean(inv)), "window_months": months, "decembers": [TEST_YEARS[0], TEST_YEARS[-1]]}


def presence(firms, permno, year):
    """A of each row: its firm has a firm effect in the fit of its test year"""
    return np.array([p in firms.get(int(y), ()) for p, y in zip(permno, year)])


def availability_shares(model, F, firms):
    """the means of A, (1 - A) H, (1 - A)(1 - H) and H over Table 11's firm-days F (A and H of each headline, averaged to the firm-day) and
    over DKPT's arrival-month firm-months with a next-month return (A: 4_monthly's flag, checked against the fits' firms)"""
    D = decomposition(model, ["permno", "day0", "yr", "hasL"])
    D["A"], D["H"] = presence(firms, D.permno.values, D.yr.values).astype(float), D.hasL.astype(float)
    fd = D.groupby(["permno", "day0"])[["A", "H"]].mean()
    fd = fd.join(F.set_index(["permno", "day0"])[[]], how="inner")
    z = np.load(ROOT / f"Data/clean/firm_month_parts_{model}.npz")
    M = pd.DataFrame({"permno": z["permno"].astype("int64"), "month": pd.PeriodIndex(z["month"].astype(str), freq="M"),
                      "A": z["seen"].astype(float), "H": z["hasL"].astype(float)})
    assert (presence(firms, M.permno.values, M.month.dt.year.values) == z["seen"]).all(), "4_monthly's firm flag differs from the fits' firms"
    M = M.merge(jkp_returns()[["permno", "month"]], on=["permno", "month"])
    return {"firm_days": shares(fd.A, fd.H), "firm_months": shares(M.A, M.H)}


def firm_month_inv_n(model):
    """the mean of 1 / n over DKPT's arrival-month firm-months with a next-month return, n the firm-month's news items (4_monthly)"""
    z = np.load(ROOT / f"Data/clean/firm_month_parts_{model}.npz")
    M = pd.DataFrame({"permno": z["permno"].astype("int64"), "month": pd.PeriodIndex(z["month"].astype(str), freq="M"), "n": z["n_stories"]})
    M = M.merge(jkp_returns()[["permno", "month"]], on=["permno", "month"])
    return float((1 / M.n).mean())


def shares(A, H):
    """the means of A, (1 - A) H, (1 - A)(1 - H) and H, each product formed row by row"""
    A, H = np.asarray(A, float), np.asarray(H, float)
    return {"F": float(A.mean()), "IL": float(((1 - A) * H).mean()), "ID": float(((1 - A) * (1 - H)).mean()), "H": float(H.mean())}


def breadth(lagged, A, H):
    """the rows with a lagged level over the rows with a current level input (A or H), and the share of the latter"""
    cur = np.asarray(A, bool) | np.asarray(H, bool)
    return float(np.sum(lagged) / cur.sum()), float(cur.mean())


def return_variance(F):
    """mean cross-sectional variance of WINDOW-month sums of news firms' demeaned, winsorized monthly log returns"""
    news = set(map(tuple, F.assign(y=F.day0.dt.year)[["permno", "y"]].drop_duplicates().values))
    rows = []
    for y in range(2004, 2026):
        d = pd.read_parquet(ROOT / f"Data/raw/crsp/dsf_{y}.parquet", columns=["permno", "dlycaldt", "dlyret"]).dropna(subset=["dlyret"])
        d["m"] = pd.to_datetime(d.dlycaldt).dt.to_period("M")
        d["lr"] = np.log1p(d.dlyret.clip(lower=-0.99))
        g = d.groupby(["permno", "m"]).agg(lr=("lr", "sum"), n=("lr", "size")).reset_index()
        rows.append(g[g.n >= 15])
    M = pd.concat(rows)
    M = M[[(p, m.year) in news for p, m in zip(M.permno, M.m)]]
    M["e"] = M.groupby("m").lr.transform(lambda x: x - x.mean())
    M["e"] = M.groupby("m").e.transform(lambda x: x.clip(*x.quantile([0.01, 0.99])))
    M["e"] = M.groupby("m").e.transform(lambda x: x - x.mean())
    W = M.pivot(index="m", columns="permno", values="e").sort_index().loc[pd.Period("2004-01", "M"):pd.Period("2025-12", "M")]
    vk = []
    for s in range(0, len(W) - WINDOW + 1, WINDOW):
        b = W.iloc[s:s + WINDOW]
        b = b.loc[:, b.notna().all()]
        vk.append(float(b.sub(b.mean(axis=1), axis=0).sum().var()))
    return {"months": WINDOW, "var_window": float(np.mean(vk)), "windows": len(vk)}


def lagged_breadth(model, firms, first="2005-01", last="2025-11"):
    """on DKPT's arrival-month firm-months with a next-month return in first..last: the rows with a level lagged LEVEL_LAG months (news
    in the LEVEL_WINDOW months ending LEVEL_LAG + 1 months earlier) over the rows with a current level input (A or H), with the
    history of all news months"""
    z = np.load(ROOT / f"Data/clean/firm_month_emb_{model}.npz")
    M = pd.DataFrame({"permno": z["permno"].astype("int64"), "month": pd.PeriodIndex(z["month"].astype(str), freq="M")})
    none = np.zeros((len(M), 0), np.float32)                                           # only whether a level exists
    M["hasL"], M["hasLlag"] = trailing_level(none, M)[1], trailing_level(none, M, lag=LEVEL_LAG + 1)[1]   # as 4_didisheim.inputs
    M = M[(M.month >= pd.Period(first, "M")) & (M.month <= pd.Period(last, "M"))]
    M = M.merge(jkp_returns()[["permno", "month"]], on=["permno", "month"])
    ratio, share = breadth(M.hasLlag.values, presence(firms, M.permno.values, M.month.dt.year.values), M.hasL.values)
    return {"lagged_over_current": ratio, "current_share": share, "months": [first, last]}


def main():
    out, rows = {}, []
    for m in MODELS:
        F = panel(m)
        rows.append(daily(m, F))
        if m == "Llama-2-13b-hf":                     # shares, counts and the news firms are the same for every encoder (same headlines)
            firms = projection_firms(m)
            out["projection_years"] = {str(T): [firm_years(T)[0], firm_years(T)[-1]] for T in TEST_YEARS}
            out["n_train"], out["firm_valuation"] = n_train(m), firm_valuation(m)
            out["availability"], out["firm_month_inv_n"] = availability_shares(m, F, firms), firm_month_inv_n(m)
            out["return_variance"] = return_variance(F)
            out["lagged_breadth"] = lagged_breadth(m, firms)
        print(m, len(F), "firm-days", flush=True)
    pd.concat(rows, ignore_index=True).to_csv(HERE / "cutoff_daily.csv.gz", index=False)
    json.dump(out, open(HERE / "cutoff_inputs.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
