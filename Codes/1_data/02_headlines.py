"""Step 2: the table of unique headline strings to embed. Run from Codes/1_data/ after 01_pull.py:
    ../../env/bin/python 02_headlines.py
In : Data/raw/ravenpack/dj_<year>.parquet (all rows)
Out: Data/headlines/headlines.parquet  columns hid (0..H-1, sorted by the first year, then by the headline), headline, n_stories, first_year
     Identical headline strings get one hid, so each string is embedded once; 04_clean attaches hid to every story."""
import time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; RAW = ROOT / "Data/raw/ravenpack"; OUT = ROOT / "Data/headlines"; OUT.mkdir(exist_ok=True)
t0 = time.time(); parts = []
for f in sorted(RAW.glob("dj_*.parquet")):
    y = int(f.stem.split("_")[1]); d = pd.read_parquet(f, columns=["rp_story_id", "headline"]).drop_duplicates("rp_story_id")
    d["headline"] = d.headline.fillna("").astype(str); g = d.groupby("headline").size().rename("n").reset_index(); g["first_year"] = y; parts.append(g)
    print(y, f"{len(d):,} stories, {len(g):,} distinct headlines [{time.time() - t0:.0f}s]", flush=True)
h = pd.concat(parts).groupby("headline").agg(n_stories=("n", "sum"), first_year=("first_year", "min")).reset_index().sort_values(["first_year", "headline"]).reset_index(drop=True)
h.insert(0, "hid", np.arange(len(h), dtype=np.int64)); h.to_parquet(OUT / "headlines.parquet", index=False)
# hid = rank by (first_year, headline): adding headlines can shift ids, so a rebuilt table needs rebuilt clean_*.parquet
# (emb.verify checks the vectors against this table, not the clean panels)
print(f"{len(h):,} unique headlines for {h.n_stories.sum():,} stories; empty headlines: {(h.headline == '').sum()} [{time.time() - t0:.0f}s]")
