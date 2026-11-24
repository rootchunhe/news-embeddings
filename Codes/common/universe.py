"""Sample definition shared by the replications and the paper analyses.

A story-firm row (one RavenPack story, one tagged US company, from Data/clean) is in the sample when
  1. the story tags exactly one company with relevance 100 (CKX: "articles tagged to a single stock"; RavenPack's relevance score
     decides which tagged company a story is about, and companies mentioned at a lower relevance are allowed) and RavenPack
     records an event category for it (a handful of stories have none);
  2. the firm has a non-missing next open-to-open return (CKX: "non-missing open-to-open returns");
  3. no headline of the same firm in the prior 5 trading days has bag-of-words cosine >= 0.8 with it (CKX's de-duplication; the
     words exclude a short stop-word list, 1_data/04_clean.py);
  4. the firm's previous-day market cap is at or above the NYSE 5th percentile.
Rules 1-3 are the paper's (the event-category requirement of rule 1 is ours); CKX's length rule (100-100,000 characters) applies to articles, not to headlines. Rule 4 is an
additional filter of ours: the Dow Jones edition covers far more micro-caps than Reuters, and the cap floor brings the number
of stocks with news per month close to the paper's (the counts are reported in the replication notebook). No content-type
filter is applied because the papers specify none; the Dow Jones edition carries automatically generated items (insider
filings, chart commentary, filing notices) that Reuters does not, which is documented in the notebook rather than filtered.
Didisheim et al. aggregate stories to firm-months; rules 1, 3 and 4 apply there (rule 2 is replaced by the monthly return merge)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PCT = 5
DESCRIPTION = "single-company stories with a RavenPack event category, CKX 5-day text de-duplication, next o2o return available (daily samples; firm-months need none), cap >= NYSE 5th percentile"


def nyse_breakpoints():
    """per trading day: NYSE market-cap percentiles (5, 20, 50) of the PREVIOUS trading day, i.e. what is known at the open"""
    fn = ROOT / "Data/clean/nyse_breakpoints.parquet"
    if fn.exists():   # cached: delete the file after a CRSP refresh
        return pd.read_parquet(fn)
    files = sorted((ROOT / "Data/raw/crsp").glob("dsf_*.parquet"))
    cr = pd.concat([pd.read_parquet(f, columns=["dlycaldt", "dlycap", "primaryexch"]) for f in files])
    cr = cr[cr.primaryexch == "N"]
    cr["dlycaldt"] = pd.to_datetime(cr.dlycaldt)
    cr["dlycap"] = pd.to_numeric(cr.dlycap, errors="coerce").astype(float)
    bp = cr.groupby("dlycaldt").dlycap.quantile([0.05, 0.20, 0.50]).unstack()
    bp.columns = ["nyse5", "nyse20", "nyse50"]
    bp = bp.shift(1).iloc[1:]
    bp.index.name = "day0"
    bp = bp.reset_index()
    bp.to_parquet(fn, index=False)
    return bp


def in_universe(df, day_col="day0", cap_col="dlycap"):
    """rule 4: previous-day cap at or above the previous-day NYSE 5th percentile (missing cap -> False)"""
    bp = nyse_breakpoints().set_index("day0")[f"nyse{UNIVERSE_PCT}"]
    thr = bp.reindex(pd.to_datetime(df[day_col])).values
    cap = pd.to_numeric(df[cap_col], errors="coerce").astype(float).values
    return (cap >= thr) & np.isfinite(cap) & np.isfinite(thr)


def in_sample(df, need_return=True):
    """rules 1-4 (rule 2 optional) for a frame with single_comp, category, dup_ckx, o2o_1, day0, dlycap"""
    m = df.single_comp.values.astype(bool) & df.category.notna().values & ~df.dup_ckx.values.astype(bool) & in_universe(df)
    if need_return:
        m &= pd.to_numeric(df.o2o_1, errors="coerce").notna().values
    return m


def sample_counts(df, label=""):
    """one line with stories per month and stocks per month / per day, for the log of every script"""
    d = pd.DataFrame({"day0": pd.to_datetime(df.day0), "permno": df.permno})
    months = d.day0.dt.to_period("M")
    print(f"sample{(' ' + label) if label else ''}: {len(d):,} stories | {len(d) / months.nunique():,.0f} per month | "
          f"stocks per month {d.groupby(months).permno.nunique().mean():,.0f}, per day {d.groupby('day0').permno.nunique().mean():,.0f} | "
          f"{d.day0.min().date()}..{d.day0.max().date()}", flush=True)
