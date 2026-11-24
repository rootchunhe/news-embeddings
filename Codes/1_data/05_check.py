"""Sanity checks of the data layer (run from Codes/1_data/ after 04_clean.py): per-year counts, flag shares, label coverage, embedding
coverage of the headlines, and a few spot checks (a sample of primary events vs all events, duplicate pairs)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "Codes")); pd.set_option("display.width", 250)
H = pd.read_parquet(ROOT / "Data/headlines/headlines.parquet"); print(f"headlines: {len(H):,} unique strings for {H.n_stories.sum():,} stories")
for m in sorted(x.name for x in (ROOT / "Data/embedding").iterdir() if x.is_dir()):
    mfs = sorted((ROOT / "Data/embedding" / m).glob("emb_*.json"))
    if mfs: js = [json.loads(f.read_text()) for f in mfs]; print(f"  {m}: {len(js)} chunks, {sum(j['n'] for j in js):,} rows, truncated {sum(j['truncated'] for j in js)}, p99 len {max(j['len_p99'] for j in js)}, {sum(j['secs'] for j in js) / 60:.0f} GPU-min")
rows = []
for f in sorted((ROOT / "Data/clean").glob("clean_*.parquet")):
    d = pd.read_parquet(f, columns=["rp_story_id", "permno", "day0", "pub_day", "single_comp", "dup_exact5", "dup_ckx", "o2o_1", "ret3_c2c", "category", "event_relevance", "n_comp100", "via"])
    ck = d[d.single_comp & d.category.notna() & ~d.dup_ckx & d.o2o_1.notna()]
    rows.append({"year": int(f.stem.split("_")[1]), "story_firm_rows": len(d), "stories": d.rp_story_id.nunique(), "single_comp_%": 100 * d.single_comp.mean(), "dup_exact5_%": 100 * d.dup_exact5.mean(), "dup_ckx_%": 100 * d.dup_ckx.mean(),
                 "o2o_1_ok_%": 100 * d.o2o_1.notna().mean(), "ret3_ok_%": 100 * d.ret3_c2c.notna().mean(), "via_ticker_%": 100 * (d.via == "ticker").mean(), "same_day_%": 100 * (d.day0 == d.pub_day).mean(),
                 "CKX_sample": len(ck), "CKX_firms": ck.permno.nunique(), "CKX_per_day": len(ck) / ck.day0.nunique() if len(ck) else 0, "evrel_lt100_%": 100 * (d.event_relevance < 100).mean()})
t = pd.DataFrame(rows).set_index("year"); print(t.round(1).to_string()); print("CKX_*: single-company, with an event category, non-duplicate, with a return, before the cap floor of common/universe.py; total 2004-2019:", int(t.loc[2004:2019].CKX_sample.sum()))
d = pd.read_parquet(ROOT / "Data/clean/clean_2015.parquet"); print("\ncolumns:", d.columns.tolist())
print("\ntop categories (primary event) 2015:"); print(d.category.value_counts().head(15).to_string())
print("\nexample duplicates (dup_ckx) 2015, each with the firm's last earlier headline in the window (not necessarily the matching one):"); x = d[d.dup_ckx].sample(5, random_state=0)
for _, r in x.iterrows():
    prev = d[(d.permno == r.permno) & (d.ts_ny < r.ts_ny) & (d.pub_day >= r.pub_day - pd.Timedelta(days=9))].tail(1)
    print(f"  {r.ts_ny} [{r.category}] {r.headline[:90]}\n      earlier: {prev.ts_ny.iloc[0] if len(prev) else '?'} {prev.headline.iloc[0][:90] if len(prev) else '?'}")
print("\nexample multi-event stories (categories_all has several) 2015:"); x = d[d.categories_all.str.contains(r"\|")].sample(5, random_state=1)
for _, r in x.iterrows(): print(f"  primary={r.category} ({r.event_relevance:.0f}) all={r.categories_all} | {r.headline[:80]}")
