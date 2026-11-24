"""JKP monthly panel (Data/raw/jkp, WRDS contrib.global_factor, US) and monthly helpers."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def characteristics():
    """the 153 JKP characteristics (Kelly lab 'Cluster Labels')"""
    return pd.read_csv(ROOT / "Codes/common/jkp_characteristics.csv").characteristic.tolist()


def monthly_panel(columns=None, start=None, end=None):
    """one row per (permno, month YYYY-MM); asserts no permno twice in a month"""
    files = sorted((ROOT / "Data/raw/jkp").glob("jkp_us_*.parquet"))
    J = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    J = J[J.permno.notna()].copy()
    J["permno"] = J.permno.astype(int)
    J["month"] = pd.to_datetime(J.eom).dt.strftime("%Y-%m")
    if start:
        J = J[J.month >= start]
    if end:
        J = J[J.month <= end]
    assert not J.duplicated(["permno", "month"]).any(), "JKP: a permno twice in one month"
    return J.reset_index(drop=True)


def monthly_market():
    """monthly market excess return (from the daily Fama-French file): compounded total return minus compounded risk-free rate, indexed YYYY-MM"""
    ff = pd.read_parquet(ROOT / "Data/raw/factors/ff5_daily_wrds.parquet")
    ff["date"] = pd.to_datetime(ff.date)
    g = ff.assign(mkt=1 + ff.mktrf.astype(float) + ff.rf.astype(float), rf1=1 + ff.rf.astype(float)).groupby(ff.date.dt.strftime("%Y-%m"))
    return g.mkt.prod() - g.rf1.prod()


def monthly_sharpe(r):
    r = pd.Series(r).dropna()
    return r.mean() / r.std() * np.sqrt(12) if len(r) > 1 else np.nan


def next_month(m):
    return (pd.Period(m, "M") + 1).strftime("%Y-%m")


def month_ranks(J, cols):
    """cols ranked within each month of J across all of its stocks, centered at 0 within [-0.5, 0.5] ((rank - 0.5) / n - 0.5), missing
    to 0, the mean rank (JKP / Gu-Kelly-Xiu convention); aligned with J"""
    g = J.groupby("month")[cols]
    return ((g.rank() - 0.5) / g.transform("count") - 0.5).fillna(0.0).astype("float64")   # numpy floats (the raw columns are nullable Float64)

