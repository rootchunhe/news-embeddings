"""Step 1 of the data layer: pull raw inputs from WRDS into Data/raw/ (no sample filters beyond the provider and the country).
Run from Codes/1_data/ on a node with internet access (WRDS):
    mkdir -p ../../Logs && nohup ../../env/bin/python 01_pull.py --wrds_user <user> > ../../Logs/pull.log 2>&1 &
Outputs (relative to the project root), all written atomically (temp file + rename) and recorded in Data/raw/MANIFEST.json
with the SQL hash, row count, time range and pull time; a file is re-pulled when its SQL changed, when it is the current
(still growing) year, or with --force; the manifest is updated under a file lock and written atomically:
  Data/raw/ravenpack/dj_<year>.parquet     ALL rows (every company x event) of every RavenPack RPA 1.0 story from the Dow Jones
                                           edition (provider_id='DJ') that tags at least one US company with relevance=100,
                                           all 52 columns (headline, event_text, category, event_relevance, event_similarity_*,
                                           ESS, source_name, news_type, ...). No novelty, length or single-company filter: those
                                           are sample definitions and belong to 04_clean.
  Data/raw/ravenpack/provider_counts.parquet   per year and provider: US single-company relevance-100 stories and how many
                                           pass event_similarity_days >= 5 / >= 90 (documents what the DJ choice excludes)
  Data/raw/ravenpack/company_mappings_hist.parquet, company_mappings.parquet, taxonomy.parquet
  Data/raw/crsp/dsf_<year>.parquet         CRSP daily (dsf v2): US common stocks on NYSE/AMEX/NASDAQ, 1999-
  Data/raw/crsp/stocknames.parquet, stkdelists.parquet (CRSP CIZ delisting events: date, action type, delisting return)
  Data/raw/factors/ff5_daily_wrds.parquet, umd_daily_wrds.parquet   Fama-French 5 factors and momentum from WRDS (ff.fivefactors_daily,
                                           ff.factors_daily); the other factor files are copied by hand (README)
  Data/raw/jkp/jkp_us_<year>.parquet       JKP Global Factor Data (contrib.global_factor): US common stocks, primary security, main
                                           observation and exchange, all columns, one file per year of eom (through end - 1)
  Data/raw/ibes/{actuals,surprise,link,ltg}.parquet   IBES quarterly EPS actuals and surprises (Table 3), the IBES-CRSP link and
                                           the long-term growth consensus (Table 4)
"""
import argparse, fcntl, hashlib, json, os, time
from pathlib import Path
import pandas as pd, wrds

ROOT = Path(__file__).resolve().parents[2]; RAW = ROOT / "Data/raw"
p = argparse.ArgumentParser()
p.add_argument("--start", type=int, default=2000); p.add_argument("--end", type=int, default=2026)
p.add_argument("--wrds_user", required=True); p.add_argument("--force", action="store_true"); p.add_argument("--no_refresh", action="store_true", help="do not re-pull the current year")
p.add_argument("--only", default="", help="comma list of blocks to run: rp,crsp,factors,jkp,ibes,counts (default all)")
a = p.parse_args(); blocks = set(a.only.split(",")) if a.only else {"rp", "crsp", "factors", "jkp", "ibes", "counts"}; THIS_YEAR = time.localtime().tm_year
for d in ["ravenpack", "crsp", "factors", "jkp", "ibes"]: (RAW / d).mkdir(parents=True, exist_ok=True)
MF = RAW / "MANIFEST.json"; manifest = json.loads(MF.read_text()) if MF.exists() else {}
db = wrds.Connection(wrds_username=a.wrds_user); T0 = time.time()

def pull(key, sql, fn, ts_col=None):
    """run sql -> parquet (atomic), unless manifest says the same sql already produced fn"""
    h = hashlib.sha1(sql.encode()).hexdigest()[:12]; current = key.endswith(f"_{THIS_YEAR}")        # the current year keeps growing on WRDS
    if not a.force and fn.exists() and manifest.get(key, {}).get("sql_sha1") == h and not (current and not a.no_refresh): return
    t0 = time.time(); df = db.raw_sql(sql); tmp = fn.with_name(f"{fn.name}.{os.uname().nodename}.{os.getpid()}.tmp")   # per process: parallel pulls may write one table
    df.to_parquet(tmp, index=False); os.replace(tmp, fn)
    entry = {"file": str(fn.relative_to(ROOT)), "rows": int(len(df)), "sql_sha1": h, "pulled_at": time.strftime("%Y-%m-%d %H:%M"),
             "secs": round(time.time() - t0), **({"ts_min": str(df[ts_col].min()), "ts_max": str(df[ts_col].max())} if ts_col and len(df) else {})}
    with open(MF.with_suffix(".lock"), "w") as lk:                                   # read-modify-write under a lock, written atomically
        fcntl.flock(lk, fcntl.LOCK_EX); cur = json.loads(MF.read_text()) if MF.exists() else {}; cur[key] = entry
        tmp = MF.with_suffix(".tmp"); tmp.write_text(json.dumps(cur, indent=1)); os.replace(tmp, MF)
    print(f"{key}: {len(df):,} rows {time.time() - t0:.0f}s  [{time.time() - T0:.0f}s]", flush=True)

if "rp" in blocks:
    pull("rp_mappings_hist", "SELECT * FROM rpna.wrds_rpa_company_mappings_hist WHERE entity_type='COMP'", RAW / "ravenpack/company_mappings_hist.parquet")
    pull("rp_mappings", "SELECT * FROM rpna.wrds_rpa_company_mappings WHERE entity_type='COMP'", RAW / "ravenpack/company_mappings.parquet")
    pull("rp_taxonomy", "SELECT * FROM rpna.rpa_taxonomy", RAW / "ravenpack/taxonomy.parquet")
    for y in range(a.start, a.end + 1):
        sql = f"""WITH q AS (SELECT DISTINCT rp_story_id FROM rpna.rpa_full_equities_{y}
                    WHERE provider_id='DJ' AND entity_type='COMP' AND country_code='US' AND relevance=100)
                  SELECT b.* FROM rpna.rpa_full_equities_{y} b JOIN q USING (rp_story_id)
                  ORDER BY b.timestamp_utc, b.rp_story_id, b.rp_entity_id, b.rp_story_event_index"""
        pull(f"rp_dj_{y}", sql, RAW / f"ravenpack/dj_{y}.parquet", ts_col="timestamp_utc")
if "crsp" in blocks:
    pull("crsp_stocknames", "SELECT permno, ncusip, ticker, comnam, namedt, nameenddt, shrcd, exchcd FROM crsp.stocknames", RAW / "crsp/stocknames.parquet")
    pull("crsp_stkdelists", "SELECT permno, delistingdt, delactiontype, delret, delretmisstype FROM crsp.stkdelists WHERE delistingdt >= '1999-01-01'",
         RAW / "crsp/stkdelists.parquet", ts_col="delistingdt")
    for y in range(a.start - 1, a.end + 1):
        sql = f"""SELECT permno, cusip, dlycaldt, dlyret, dlyopen, dlyclose, dlyhigh, dlylow, dlyvol, dlycap, dlyprc, dlyprcflg, dlydelflg, primaryexch, securitytype, sharetype, securitysubtype
                  FROM crspm.wrds_dsfv2_query
                  WHERE dlycaldt BETWEEN '{y}-01-01' AND '{y}-12-31' AND securitytype='EQTY' AND sharetype='NS' AND securitysubtype='COM'
                    AND usincflg='Y' AND issuertype IN ('ACOR','CORP') AND primaryexch IN ('N','A','Q')"""
        pull(f"crsp_{y}", sql, RAW / f"crsp/dsf_{y}.parquet", ts_col="dlycaldt")
if "factors" in blocks:
    pull("ff5_daily", "SELECT date, mktrf, smb, hml, rmw, cma, rf FROM ff.fivefactors_daily WHERE date >= '1999-01-01'", RAW / "factors/ff5_daily_wrds.parquet", ts_col="date")
    pull("umd_daily", "SELECT date, umd FROM ff.factors_daily WHERE date >= '1999-01-01'", RAW / "factors/umd_daily_wrds.parquet", ts_col="date")
if "jkp" in blocks:
    for y in range(a.start, a.end):                                      # JKP publishes a year with a lag: the last raw year is not there yet
        sql = f"""SELECT * FROM contrib.global_factor WHERE excntry='USA' AND common=1 AND primary_sec=1 AND obs_main=1 AND exch_main=1
                  AND eom BETWEEN '{y}-01-01' AND '{y}-12-31' ORDER BY permno, eom, id"""   # a fixed row order
        pull(f"jkp_us_{y}", sql, RAW / f"jkp/jkp_us_{y}.parquet", ts_col="eom")
if "ibes" in blocks:
    pull("ibes_actuals", "SELECT ticker, pends, anndats, anntims, value AS actual, curr_act FROM ibes.actu_epsus WHERE measure='EPS' AND pdicity='QTR' AND usfirm=1 AND anndats >= '2003-01-01'",
         RAW / "ibes/actuals.parquet", ts_col="anndats")
    pull("ibes_surprise", "SELECT ticker, pyear, pmon, anndats, actual, surpmean, surpstdev, suescore FROM ibes.surpsumu WHERE measure='EPS' AND fiscalp='QTR' AND usfirm=1 AND anndats >= '2003-01-01'",
         RAW / "ibes/surprise.parquet", ts_col="anndats")
    pull("ibes_link", "SELECT ticker, permno, sdate, edate, score FROM wrdsapps.ibcrsphist WHERE permno IS NOT NULL AND score <= 3", RAW / "ibes/link.parquet")
    pull("ibes_ltg", "SELECT ticker, statpers, medest, meanest, numest FROM ibes.statsumu_epsus WHERE measure='EPS' AND fpi='0' AND usfirm=1 AND statpers >= '2000-01-01'",
         RAW / "ibes/ltg.parquet", ts_col="statpers")
if "counts" in blocks:
    rows = []
    for y in range(a.start, a.end + 1):
        q = f"""WITH cnt AS (SELECT rp_story_id FROM rpna.rpa_full_equities_{y} WHERE entity_type='COMP' AND relevance=100 GROUP BY rp_story_id HAVING COUNT(DISTINCT rp_entity_id)=1)
                SELECT b.provider_id, COUNT(DISTINCT b.rp_story_id) AS stories,
                       COUNT(DISTINCT CASE WHEN b.event_similarity_days >= 5 THEN b.rp_story_id END) AS stories_novel5,
                       COUNT(DISTINCT CASE WHEN b.event_similarity_days >= 90 THEN b.rp_story_id END) AS stories_novel90
                FROM rpna.rpa_full_equities_{y} b JOIN cnt USING (rp_story_id)
                WHERE b.country_code='US' AND b.entity_type='COMP' AND b.relevance=100 GROUP BY b.provider_id"""
        d = db.raw_sql(q); d.insert(0, "year", y); rows.append(d); print(f"counts {y} done [{time.time() - T0:.0f}s]", flush=True)
        tmp = RAW / "ravenpack/provider_counts.parquet.tmp"; pd.concat(rows).to_parquet(tmp, index=False); os.replace(tmp, RAW / "ravenpack/provider_counts.parquet")
db.close(); print("ALL DONE", f"[{time.time() - T0:.0f}s]", flush=True)
