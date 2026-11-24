"""Story-firm rows of the sample (common/universe.in_sample) read from Data/clean/clean_<year>.parquet, with the paper's
standard columns, the out-of-sample forecasts of the replication, and the firm-day aggregation."""
import hashlib

import numpy as np
import pandas as pd

from lib import ROOT
from lib.constants import PAPER, PRED

from common.config import Y_CLIP

from common.runinfo import require_same_sample
from common.jkp import monthly_panel
from common.universe import in_sample, nyse_breakpoints

SIZE_GROUPS = ("micro", "small", "big")                                  # Fama and French (2008); micro is the `small` of sample_rows
BASE = ["rp_story_id", "permno", "hid", "day0", "day1", "ts_ny", "pub_day", "category", "group", "o2o_1", "dlycap", "single_comp", "dup_ckx", "event_sentiment_score"]


def sample_rows(years, columns=(), need_return=True):
    """the sample stories of the given clean years: BASE columns plus `columns`, cast; adds yr, day1_yr, y (next open-to-open
    return), y_fit (y winsorized at +-Y_CLIP), cap, small, ess (0 when missing, flagged by ess_missing)"""
    cols = list(dict.fromkeys(BASE + list(columns)))
    rows = []
    for y in years:
        c = pd.read_parquet(ROOT / f"Data/clean/clean_{y}.parquet", columns=cols)
        rows.append(c[in_sample(c, need_return)].reset_index(drop=True))
    S = pd.concat(rows, ignore_index=True)
    S["permno"] = S.permno.astype("int64")
    S["hid"] = S.hid.astype("int64")
    for c in ["day0", "day1", "ts_ny", "pub_day"]:
        S[c] = pd.to_datetime(S[c])
    S["yr"] = S.day0.dt.year
    S["day1_yr"] = S.day1.dt.year
    S["y"] = pd.to_numeric(S.o2o_1, errors="coerce").astype(float)                 # the realized return of every evaluation
    S["y_fit"] = S.y.clip(-Y_CLIP, Y_CLIP)                                            # the training target (ridge, category means)
    S["cap"] = pd.to_numeric(S.dlycap, errors="coerce").astype(float)
    S["small"] = S.cap < S.day0.map(nyse_breakpoints().set_index("day0").nyse20).values
    ess = pd.to_numeric(S.event_sentiment_score, errors="coerce").astype(float)
    S["ess_missing"] = ess.isna()
    S["ess"] = ess.fillna(0)
    for c in ["category", "group"]:
        S[c] = S[c].astype(str)
    drop = ["o2o_1", "dlycap", "single_comp", "dup_ckx", "event_sentiment_score"]
    return S.drop(columns=[c for c in drop if c not in columns])          # a raw column asked for explicitly is kept


def size_group(day0, cap):
    """Fama and French (2008) size groups on the NYSE breakpoints of the previous trading day (common.universe.nyse_breakpoints, as
    `small` of sample_rows): micro below the 20th percentile (= small), small from the 20th percentile to the median, big above it"""
    bp = nyse_breakpoints().set_index("day0")
    b20, b50 = day0.map(bp.nyse20).values, day0.map(bp.nyse50).values
    cap = np.asarray(cap, dtype=float)
    assert np.isfinite(b20).all() and np.isfinite(b50).all() and np.isfinite(cap).all(), "a row without its cap or NYSE breakpoints"
    return np.select([cap < b20, cap < b50], ["micro", "small"], "big")


def forecasts(S, models=PRED):
    """merge the replication's out-of-sample forecasts (Data/clean/pred_<model>.parquet) as the columns of `models`"""
    for col, model in models.items():
        require_same_sample(ROOT / f"Data/clean/run_pred_{model}.json", f"forecasts of {model}", needs=("ridge",))
        pr = pd.read_parquet(ROOT / f"Data/clean/pred_{model}.parquet", columns=["rp_story_id", "permno", "yhat"]).rename(columns={"yhat": col})
        pr["permno"] = pr.permno.astype("int64")
        S = S.merge(pr, on=["rp_story_id", "permno"], how="left")
    return S


def rewrites(model):
    """the rewrite texts of 2_perturb.py --build and their embeddings (2_embed.py --texts), checked against each other; without the
    stories whose headline Dow Jones continues in a second one ("... Non-GAAP EPS of -2-"), where the magnitude rewrites scale the marker"""
    P = pd.read_parquet(ROOT / "Data/clean/perturb_texts.parquet")
    m = require_same_sample(ROOT / f"Data/clean/run_perturb_texts_emb_{model}.json", f"rewrite embeddings of {model}")
    require_same_sample(ROOT / "Data/clean/run_perturb_texts.json", "rewrite texts")
    sha = hashlib.sha1("\n".join(P.text.fillna("")).encode()).hexdigest()
    assert m.get("texts_sha1") == sha, "the rewrite embeddings were made from other texts; rerun 2_embed.py --texts"
    Z = np.load(ROOT / f"Data/clean/perturb_texts_emb_{model}.npz")["emb"]                  # float16
    assert len(Z) == len(P)
    cont = P.rp_story_id.isin(P.rp_story_id[(P.variant == "original") & P.text.str.contains(r"-\d+-\s*(?:>\S+\s*)?$")]).values
    return P[~cont].reset_index(drop=True), Z[~cont]


def rewrite_pairs(P, Z, T):
    """the rewrites of test year T (rows of P other than the originals) with their embeddings and those of their originals"""
    d = P[P.yr == T]
    orig = d[d.variant == "original"]
    pos = pd.Series(orig.index.values, index=orig.rp_story_id.values)
    v = d[d.variant != "original"]
    return v, Z[v.index.values].astype(np.float32), Z[pos[v.rp_story_id.values].values].astype(np.float32)


def decomposition(model, columns=None):
    """the story-level decomposition of 3_decompose (Data/clean/decomposition_<model>.parquet) after the manifest check"""
    require_same_sample(ROOT / f"Data/clean/run_decomposition_{model}.json", f"decomposition of {model}", needs=PAPER)
    D = pd.read_parquet(ROOT / f"Data/clean/decomposition_{model}.parquet", columns=columns)
    for c in ["day0", "ts_ny"]:
        if c in D:
            D[c] = pd.to_datetime(D[c])
    return D


def segment(F, lo, hi):
    """the rows of the trade-day years lo..hi"""
    return F[F.day0.dt.year.between(lo, hi)]


def firm_day(S, cols, extra=None):
    """story signals averaged to firm-days; keeps y, cap, small and the year"""
    agg = {c: (c, "mean") for c in cols}
    agg.update({"y": ("y", "first"), "cap": ("cap", "first"), "small": ("small", "first"), "yr": ("yr", "first")})
    if extra:
        agg.update(extra)
    return S.groupby(["day0", "permno"]).agg(**agg).reset_index()


def industry(S):
    """6-digit GICS industry of the firm at the last JKP month end before the story's day0 (point in time: the last known code is
    carried forward, never backward), "NA" if unknown; returns S sorted by permno, day0"""
    J = monthly_panel(columns=["permno", "eom", "gics"])
    J = J[J.gics.notna()].copy()
    J["eom"], J["permno"] = pd.to_datetime(J.eom), J.permno.astype("int64")
    J["ind"] = J.gics.astype(str).str.slice(0, 6)
    S = S.sort_values("day0").reset_index(drop=True)
    m = pd.merge_asof(S[["permno", "day0"]], J[["permno", "eom", "ind"]].sort_values("eom"), left_on="day0", right_on="eom", by="permno",
                      direction="backward", allow_exact_matches=False)
    S["ind"] = m.ind.fillna("NA").values
    return S.sort_values(["permno", "day0"]).reset_index(drop=True)
