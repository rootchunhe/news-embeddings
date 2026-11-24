"""Daily factor returns for the alpha tables, all as decimal returns indexed by date.
  ff_daily():   Fama-French 5 factors + momentum from WRDS (ff.fivefactors_daily, ff.factors_daily) + Ken French's short- and
                long-term reversal factors (Data/raw/factors/french/*.csv): mktrf smb hml rmw cma umd strev ltrev rf
  dmrs_daily(): Daniel-Mota-Rottke-Santos hedged factors (kentdaniel.net): MktRF_DMRS SMB_DMRS HML_DMRS RMW_DMRS CMA_DMRS
  q5_daily():   Hou-Mo-Xue-Zhang q5 (global-q.org): R_MKT R_ME R_IA R_ROE R_EG (the file is in percent)"""
from pathlib import Path
import numpy as np
import pandas as pd

from common.config import NW_LAGS_DAILY
ROOT = Path(__file__).resolve().parents[2]; F = ROOT / "Data/raw/factors"

def _french_csv(fn, cols):
    """Ken French daily csv: header lines, then yyyymmdd rows in percent, then a blank line / copyright footer"""
    rows = []
    for line in Path(fn).read_text(errors="ignore").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8: rows.append(parts[:len(cols) + 1])
    d = pd.DataFrame(rows, columns=["date"] + cols); d["date"] = pd.to_datetime(d.date, format="%Y%m%d")
    for c in cols: d[c] = pd.to_numeric(d[c], errors="coerce") / 100
    return d.set_index("date")

def ff_daily():
    ff5 = pd.read_parquet(F / "ff5_daily_wrds.parquet"); ff5["date"] = pd.to_datetime(ff5.date); ff5 = ff5.set_index("date").astype(float)
    umd = pd.read_parquet(F / "umd_daily_wrds.parquet"); umd["date"] = pd.to_datetime(umd.date); umd = umd.set_index("date").astype(float)
    st = _french_csv(F / "french/F-F_ST_Reversal_Factor_daily.csv", ["strev"]); lt = _french_csv(F / "french/F-F_LT_Reversal_Factor_Daily.csv", ["ltrev"])
    return ff5.join(umd, how="left").join(st, how="left").join(lt, how="left")

def dmrs_daily():
    d = pd.read_csv(F / "dmrs_factor_portfolios_daily.txt", sep="\t"); d["date"] = pd.to_datetime(d.date.astype(str), format="%Y%m%d")
    return d.set_index("date").rename(columns=lambda c: c.replace("_star", "_DMRS")).astype(float)

def q5_daily():
    q = pd.read_csv(F / "q5_factors_daily_2025.csv"); q["date"] = pd.to_datetime(q.date.astype(str), format="%Y-%m-%d")
    return (q.set_index("date")[["R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG"]].astype(float) / 100)

def nw_ols(y, X, lags=NW_LAGS_DAILY):
    """OLS with Newey-West (Bartlett, `lags`) standard errors; X without constant. Returns (beta, t, R2) with the constant first."""
    Xm = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(Xm, y, rcond=None); e = y - Xm @ b
    XtXi = np.linalg.inv(Xm.T @ Xm); S = (Xm * e[:, None]).T @ (Xm * e[:, None])
    for l in range(1, lags + 1): G = (Xm[l:] * e[l:, None]).T @ (Xm[:-l] * e[:-l, None]); S += (1 - l / (lags + 1)) * (G + G.T)
    return b, b / np.sqrt(np.diag(XtXi @ S @ XtXi)), 1 - e.var() / y.var()
