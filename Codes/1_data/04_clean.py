"""Step 4: the story-firm panel with trading dates, CRSP labels and sample flags. Run from Codes/1_data/ (CPU, ~1 h, ~20 GB RAM)
over all years at once (pass 2 takes the duplicate context of a year's first days from the previous year's December, so a run
with a later --start lacks it):
    mkdir -p ../../Logs && nohup ../../env/bin/python 04_clean.py > ../../Logs/clean.log 2>&1 &
In : Data/raw/ravenpack/dj_<y>.parquet, company_mappings_hist.parquet, Data/raw/crsp/*.parquet, Data/headlines/headlines.parquet
Out: Data/clean/clean_<Y>.parquet, one row per (story, US company tagged with relevance 100 and mapped to a CRSP permno),
     filed by the year of day0 (the trading day), so a file holds exactly the trades of year Y.
Columns
  identity   rp_story_id, hid (row in Data/headlines; embeddings are keyed by hid), headline, rp_entity_id, entity_name, permno, via
  timing     timestamp_utc, ts_ny, pub_day (NY date -> trading day on/after), before_open (NY time < 09:00), day0 (trade day: the trading
             day on/after the NY date, advanced one calendar day when the NY time is 09:00 or later), day1 (trading day after day0, when o2o_1 is realized)
  story      news_type, rp_source_id, source_name, n_events, n_comp100 (companies tagged with relevance 100, any country),
             n_comp_any (companies at any relevance); single_comp = n_comp100 == 1
  primary    the firm's event with the highest event_relevance (ties: first in the story): category, group, type, sub_type, property,
  event      fact_level, event_relevance, event_similarity_key, event_similarity_days, event_sentiment_score (ESS), event_text,
             earnings_type, reporting_period; categories_all = '|'-joined set of all categories of this firm in the story
  dedup      dup_exact5: same headline for the same firm earlier in the prior 5 trading days (incl. the same day)
             dup_ckx: CKX's filter, binary bag-of-words cosine >= 0.8 with any of the firm's headlines in that window
  labels     o2o_k (open day0+k-1 -> open day0+k, k=1..10; o2o_1 is the CKX target), ret1_c2c (close day0 -> close day1),
             ret3_c2c (the KKX / CKX-2024 sentiment label: the three-day return from close t-2 to close t+1, with t the trading day
             whose 4pm-to-4pm window contains the publication, column kkx_day), label_pos = 1[ret3_c2c > 0],
             gap0 (close day0-1 -> open day0), intra0 (open day0 -> close day0), gap1 (close day0 -> open day1): (1+o2o_1) = (1+intra0)(1+gap1)
             dlycap (market cap at close day0-1)
All returns are split/dividend adjusted via dlyret (CRSP open/close are not); a k-step return is NaN unless the row k steps ahead is
exactly k trading days ahead, and an open print off by more than 2x from the close is treated as missing.
Open-to-open returns come from Codes/common/crsp.py (shared with the CKX tables). The sample used by the replications
and the paper is defined in Codes/common/universe.py from these flags. Nothing is dropped
here except firms without a permno and news after the CRSP calendar. A RavenPack entity whose CUSIPs match several permnos on the
publication day (the share classes of a dual-class firm) is mapped to the class with the largest market cap at the close before day0."""
import argparse
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data/raw"
sys.path.insert(0, str(ROOT / "Codes"))
from common.crsp import daily_panel, o2o, open_to_close

p = argparse.ArgumentParser()
p.add_argument("--start", type=int, default=2000)
p.add_argument("--end", type=int, default=2026)                  # the last year of the raw data (a December story trades in end + 1 only after the CRSP calendar)
p.add_argument("--kmax", type=int, default=10)
p.add_argument("--out", default=str(ROOT / "Data/clean"), help="output folder (a scratch folder for tests)")
a = p.parse_args()
K = a.kmax
OUT = Path(a.out)
TMP = OUT / "_tmp"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir()                                                     # a leftover _tmp of an interrupted run is refused
t0 = time.time()
log = lambda *x: print(*x, f"[{time.time() - t0:.0f}s]", flush=True)
num = lambda s: pd.to_numeric(s, errors="coerce").astype(float)

# ---- CRSP labels on the trading calendar ---------------------------------------------------------------------------
crsp, tdays = daily_panel()
log(f"crsp {len(crsp):,} rows, {crsp.permno.nunique():,} permnos, {len(tdays):,} trading days {tdays[0].astype('datetime64[D]')}..{tdays[-1].astype('datetime64[D]')}")
g = crsp.groupby("permno")
sh = lambda col, k: g[col].shift(-k) if k else crsp[col]
ok = {k: (sh("pos", k) == crsp.pos + k) for k in [-1, 1]}
for k in range(1, K + 1):
    crsp[f"o2o_{k}"] = o2o(crsp, k, g)
crsp["ret3_c2c"] = ((1 + sh("dlyret", -1)) * (1 + crsp.dlyret) * (1 + sh("dlyret", 1)) - 1).where(ok[-1] & ok[1])
crsp["ret1_c2c"] = sh("dlyret", 1).where(ok[1])
crsp["intra0"] = open_to_close(crsp)
crsp["gap1"] = (1 + crsp.o2o_1) / (1 + crsp.intra0) - 1
crsp["gap0"] = ((1 + crsp.dlyret) * crsp.dlyopen / crsp.dlyclose - 1).where(ok[-1] & (crsp.dlyopen / crsp.dlyclose).between(0.5, 2.0))
crsp["cap_prev"] = g.dlycap.shift(1).where(ok[-1])
crsp["day1"] = pd.Series(tdays[np.minimum(crsp.pos + 1, len(tdays) - 1)]).where(ok[1])
lab = crsp[["permno", "dlycaldt", "ret3_c2c"]].rename(columns={"dlycaldt": "kkx_day"})
trd_cols = ["permno", "dlycaldt", "day1", "cap_prev", "ret1_c2c", "gap0", "intra0", "gap1"] + [f"o2o_{k}" for k in range(1, K + 1)]
trd = crsp[trd_cols].rename(columns={"dlycaldt": "day0", "cap_prev": "dlycap"})    # dlycap in the clean files = cap at the close of day0-1
universe = set(crsp.permno.unique())
link_c = crsp[["permno", "cusip"]].drop_duplicates()
del crsp, g
log("returns computed")

# ---- RavenPack entity -> permno: 8-digit CUSIP (RavenPack cusip/ISIN vs CRSP current + historical cusips), then ticker --------
h = pd.read_parquet(RAW / "ravenpack/company_mappings_hist.parquet", columns=["rp_entity_id", "range_start", "range_end", "ticker", "cusip", "isin"])
h["c8"] = h.cusip.astype("string").str[:8]; us = h["isin"].astype("string").str.startswith("US").fillna(False)
h.loc[h.c8.isna() & us, "c8"] = h.loc[h.c8.isna() & us, "isin"].astype("string").str[2:10]
h["tk"] = h.ticker.astype("string").str.split(".").str[0].str.upper()
h["range_start"] = pd.to_datetime(h.range_start); h["range_end"] = pd.to_datetime(h.range_end).fillna(pd.Timestamp("2099-12-31"))
sn = pd.read_parquet(RAW / "crsp/stocknames.parquet"); sn["namedt"] = pd.to_datetime(sn.namedt); sn["nameenddt"] = pd.to_datetime(sn.nameenddt); sn = sn[sn.permno.isin(universe)]
link = pd.concat([link_c.rename(columns={"cusip": "c8"}), sn[["permno", "ncusip"]].rename(columns={"ncusip": "c8"})]).dropna()
link = link.groupby(["c8", "permno"]).size().reset_index(name="n").sort_values("n").drop_duplicates("c8", keep="last")[["c8", "permno"]]
map_c = h.dropna(subset=["c8"]).merge(link, on="c8")[["rp_entity_id", "range_start", "range_end", "permno"]].drop_duplicates()
map_t = h.dropna(subset=["tk"])[["rp_entity_id", "range_start", "range_end", "tk"]].drop_duplicates()
tick = sn.dropna(subset=["ticker"])[["permno", "ticker", "namedt", "nameenddt"]].rename(columns={"ticker": "tk"})
log(f"mapping: {map_c.rp_entity_id.nunique():,} entities via CUSIP, {map_t.rp_entity_id.nunique():,} with a ticker")
def map_permno(n):
    """one permno per (story, entity): CUSIP match valid on the publication date (several share classes: the largest cap at the close
    before day0), else a unique point-in-time ticker match"""
    key = ["rp_story_id", "rp_entity_id"]
    x = n[key + ["pub_date", "day0"]].merge(map_c, on="rp_entity_id", how="inner"); x = x[(x.pub_date >= x.range_start) & (x.pub_date <= x.range_end)]
    x = x[key + ["day0", "permno"]].drop_duplicates().merge(trd[["permno", "day0", "dlycap"]], on=["permno", "day0"], how="left")
    n_multi = int((x.groupby(key).size() > 1).sum())
    x = x.sort_values(key + ["dlycap", "permno"], ascending=[True, True, False, True], na_position="last").drop_duplicates(key)[key + ["permno"]]
    rest = n.merge(x[key], on=key, how="left", indicator=True); rest = rest[rest._merge == "left_only"][key + ["pub_date"]]
    y = rest.merge(map_t, on="rp_entity_id", how="inner"); y = y[(y.pub_date >= y.range_start) & (y.pub_date <= y.range_end)].merge(tick, on="tk", how="inner")
    y = y[(y.pub_date >= y.namedt) & (y.pub_date <= y.nameenddt)][key + ["permno"]].drop_duplicates(); y = y[~y.duplicated(key, keep=False)]
    out = pd.concat([x.assign(via="cusip"), y.assign(via="ticker")], ignore_index=True); return n.merge(out, on=key, how="inner"), n_multi
def next_tday(dates):
    """trading day on/after each date; NaT when the date is after the last day of the CRSP calendar (such news cannot be traded in our data)"""
    i = np.searchsorted(tdays, dates.values.astype("datetime64[ns]")); out = tdays[np.minimum(i, len(tdays) - 1)].copy(); out[i >= len(tdays)] = np.datetime64("NaT"); return out
hid = pd.read_parquet(ROOT / "Data/headlines/headlines.parquet", columns=["hid", "headline"]).set_index("headline").hid

# ---- pass 1: per publication year, story-firm rows with the primary event, timing, permno and labels -------------------------
EV = ["event_relevance", "event_similarity_key", "event_similarity_days", "event_sentiment_score", "topic", "group", "type", "sub_type", "property", "fact_level", "category", "event_text", "earnings_type", "reporting_period"]
for y in range(a.start, a.end + 1):
    fn = RAW / f"ravenpack/dj_{y}.parquet"
    r = pd.read_parquet(fn); n_rows = len(r); r["relevance"] = num(r.relevance); r["event_relevance"] = num(r.event_relevance); r["rp_story_event_index"] = num(r.rp_story_event_index)
    comp = r[r.entity_type == "COMP"]; nc = comp.groupby("rp_story_id").rp_entity_id.nunique().rename("n_comp_any"); nc100 = comp[comp.relevance == 100].groupby("rp_story_id").rp_entity_id.nunique().rename("n_comp100")
    f = comp[(comp.relevance == 100) & (comp.country_code == "US")].sort_values(["rp_story_id", "rp_entity_id", "event_relevance", "rp_story_event_index"], ascending=[True, True, False, True])
    cats = f.groupby(["rp_story_id", "rp_entity_id"]).category.agg(lambda s: "|".join(sorted(set(s.dropna())))).rename("categories_all")
    f = f.drop_duplicates(["rp_story_id", "rp_entity_id"])[["timestamp_utc", "rp_story_id", "rp_entity_id", "entity_name", "headline", "news_type", "rp_source_id", "source_name", "rp_story_event_count"] + EV].rename(columns={"rp_story_event_count": "n_events"})
    f = f.merge(cats, left_on=["rp_story_id", "rp_entity_id"], right_index=True).merge(nc100, left_on="rp_story_id", right_index=True, how="left").merge(nc, left_on="rp_story_id", right_index=True, how="left")
    f["headline"] = f.headline.fillna("").astype(str); f["hid"] = hid.reindex(f.headline).values; assert f.hid.notna().all()
    ts = pd.to_datetime(f.timestamp_utc, utc=True).dt.tz_convert("America/New_York"); f["ts_ny"] = ts.dt.tz_localize(None); pub = ts.dt.normalize().dt.tz_localize(None)
    f["before_open"] = (ts.dt.hour * 60 + ts.dt.minute) < 9 * 60; f["pub_day"] = next_tday(pub); f["day0"] = next_tday(pub.where(f.before_open, pub + pd.Timedelta(days=1)))
    f["kkx_day"] = next_tday(pub.where(ts.dt.hour < 16, pub + pd.Timedelta(days=1)))        # KKX day t: news between 4pm t-1 and 4pm t
    n_late = int(f.day0.isna().sum()); f = f[f.day0.notna()]                      # news after the last CRSP day: dropped (not back-filled)
    f["pub_date"] = pub; n0 = len(f)    # pub_date: the calendar date of publication (pub_day is the trading day on or after it)
    f, n_multi = map_permno(f); f = f.drop(columns="pub_date"); f = f.merge(lab, on=["permno", "kkx_day"], how="left").merge(trd, on=["permno", "day0"], how="left")
    f["label_pos"] = (f.ret3_c2c > 0).astype("float").where(f.ret3_c2c.notna()); f["single_comp"] = f.n_comp100 == 1
    f = f.sort_values(["ts_ny", "rp_story_id", "permno"]).reset_index(drop=True)
    f.to_parquet(TMP / f"pub_{y}.parquet", index=False)
    log(f"pub {y}: {n_rows:,} raw rows, {r.rp_story_id.nunique():,} stories ({n_late:,} rows after the CRSP calendar dropped) -> {n0:,} US relevance-100 firm rows -> {len(f):,} with permno ({(f.via == 'ticker').mean():.2f} via ticker, {n_multi:,} with another share class) | "
        f"single-company {f.single_comp.mean():.2f} | with o2o_1 {f.o2o_1.notna().mean():.2f} | firms {f.permno.nunique():,}")
    del r, comp, f

# ---- pass 2: regroup by day0 year and flag duplicates within firm over the prior 5 trading days -------------------------------
STOP = set("a an the and or of to in on for by with at from as is are be was were has have had its it this that will said says inc corp co ltd plc".split())
toks = lambda s: frozenset(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP)
def dup_flags(d):
    """d sorted by (permno, ts_ny); returns exact / cosine>=0.8 duplicate flags against the same firm's headlines in the prior 5 trading days"""
    pos = np.searchsorted(tdays, d.pub_day.values.astype("datetime64[ns]")); hl = d.headline.values; tk = [toks(s) for s in hl]
    ex = np.zeros(len(d), bool); cs = np.zeros(len(d), bool)
    for _, idx in d.groupby("permno", sort=False).indices.items():
        dq = deque()
        for i in idx:
            while dq and dq[0][0] < pos[i] - 5: dq.popleft()
            t, hh = tk[i], hl[i]
            for pp, tt, h2 in dq:
                if h2 == hh: ex[i] = True; cs[i] = True; break
                if t and tt and len(t & tt) / np.sqrt(len(t) * len(tt)) >= 0.8: cs[i] = True
            dq.append((pos[i], t, hh))
    return ex, cs
prev = None
for Y in range(a.start, a.end + 1):
    cur = pd.read_parquet(TMP / f"pub_{Y}.parquet")
    allr = pd.concat([x for x in [prev, cur] if x is not None], ignore_index=True); ctx = allr[(allr.pub_day >= pd.Timestamp(Y - 1, 12, 1)) | (allr.day0.dt.year == Y)].sort_values(["permno", "ts_ny"]).reset_index(drop=True)
    ex, cs = dup_flags(ctx); ctx["dup_exact5"] = ex; ctx["dup_ckx"] = cs
    out = ctx[ctx.day0.dt.year == Y].sort_values(["day0", "ts_ny", "rp_story_id", "permno"]).reset_index(drop=True)
    tmp = OUT / f"clean_{Y}.parquet.tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, OUT / f"clean_{Y}.parquet")
    rep = out[out.single_comp & out.category.notna() & out.o2o_1.notna()]
    log(f"clean {Y}: {len(out):,} story-firm rows | dup_exact5 {out.dup_exact5.mean():.3f} dup_ckx {out.dup_ckx.mean():.3f} | "
        f"CKX sample (single, with a category, non-dup, with return) {int((~rep.dup_ckx).sum()):,} stories, {rep.permno.nunique():,} firms | same-day trades {(out.day0 == out.pub_day).mean():.2f}")
    prev = cur
for f in TMP.glob("*.parquet"):
    f.unlink()
TMP.rmdir()
log("ALL DONE")
