# Data layer (block 1)

Run from this folder with `../../env/bin/python`; the raw pull needs a WRDS login, the embedding step a GPU (a batch job per chunk on your cluster).
A fresh download of everything needs `01_pull.py --wrds_user <user> --force`: without it
only the current year's files (unless `--no_refresh`) and files whose query changed are refreshed, and the stock names, delisting events, factors, RavenPack mappings and JKP files (whose
latest year keeps growing) already on disk are kept.
After new CRSP data, delete `Data/clean/nyse_breakpoints.parquet` too (`common/universe.py` builds it on first use and never refreshes it).

1. `01_pull.py --wrds_user <user>` (`--only` for some blocks): RavenPack stories from Dow Jones (`rpa_full_equities_<year>` with `provider_id = 'DJ'`; all company × event rows of every story tagging a US company with
   relevance 100), RavenPack company mappings and taxonomy, CRSP daily, CRSP CIZ delisting events (`stkdelists`), Fama–French factors, JKP monthly data, IBES (EPS actuals and
   surprises, the CRSP link, long-term growth), provider counts → `Data/raw/`, with `MANIFEST.json`.
2. `02_headlines.py`: the table of unique headline strings → `Data/headlines/headlines.parquet` (`hid`).
3. `03_embed.py --chunk N --model <hf name>`: mean-pooled last-layer embeddings of the unique headlines, one chunk of 500k per job →
   `Data/embedding/<model>/emb_NNN.npy` with a manifest (text hash, model revision) checked by `Codes/common/emb.py`.
4. `04_clean.py`: one row per (story, US company), filed by the year of the trade day: timing, primary event and all
   categories, CRSP labels (open-to-open returns, gap / intraday split), duplicate flags → `Data/clean/clean_<Y>.parquet`. The open-to-open
   return is (o1 / o0)(1 + r1)(c0 / c1) - 1 with CRSP's total return r1 of day 1: exact for splits, approximate on ex-dividend days
   (a dividend D of day 1 enters as D o1 / c1 instead of D; on the 2015 CRSP stock-days: mean error 0.07 bp,
   sd 2.5 bp, on the 0.6% of stock-days that are ex-days).
5. `05_check.py`: counts and spot checks.

Other inputs under `Data/raw/`, placed by hand unless noted (the hand-placed files downloaded on 2026-10-18):
- `jkp/jkp_us_<Y>.parquet`, 2000-2025 (pulled by `01_pull.py`): the Jensen-Kelly-Pedersen Global Factor Data on WRDS (`contrib.global_factor`), all 444
  columns, rows with `excntry = 'USA'`, `common = 1`, `primary_sec = 1`, `obs_main = 1`, `exch_main = 1`, one file per year of
  `eom`; read through `Codes/common/jkp.py` (identifier `permno`, month `eom`): by the DKPT replication (`ret_exc_lead1m`, `ret_exc`, `me`,
  `size_grp`, `crsp_exchcd`, `gics` and the characteristics of the residualizations) and by `Codes/3_paper` (also `permco`, `ff49`,
  `market_equity`, `me_company`, `be_me`, `ni_me`, `beta_60m`, `at_gr1`, `gp_at`, `div_at`).
- `factors/french/`: Ken French data library, daily: `F-F_ST_Reversal_Factor_daily.csv`, `F-F_LT_Reversal_Factor_Daily.csv` (read),
  `F-F_Research_Data_5_Factors_2x3_daily.csv`, `F-F_Momentum_Factor_daily.csv` (a cross-check of the WRDS series).
- `factors/dmrs_factor_portfolios_daily.txt` (and `_monthly`, `dmrs_hedge_portfolios_daily.txt`): the Daniel-Mota-Rottke-Santos
  hedged factors from Kent Daniel's website (kentdaniel.net), tab-separated, `date` as yyyymmdd, decimal returns; the series end in 2023Q1.
- `factors/q5_factors_daily_2025.csv` (and `_monthly`): the Hou-Mo-Xue-Zhang q5 factors from global-q.org, 2025 vintage, in percent.
- `Data/Loughran-McDonald_MasterDictionary_1993-2023.csv`: the Loughran-McDonald master dictionary (sraf.nd.edu), 1993-2023 version.

Entities without a CUSIP match are linked by a unique point-in-time ticker match (column `via`). The CUSIP link of `04_clean.py` matches RavenPack's CUSIPs to CRSP without CRSP's name-date ranges; in our CRSP files no 8-digit
CUSIP maps to more than one PERMNO, so the link picks no wrong firm, but it does not check that a CUSIP was the firm's at the story's
date. A RavenPack entity can carry several CUSIPs valid at once (the share classes of a dual-class firm); its stories are then mapped
to the class with the largest market cap at the close before the trade day. The CKX de-duplication (`dup_ckx`) compares a headline,
without a short stop-word list, with all of the firm's headlines of the prior five trading days, before the sample filters. The sample definition used downstream is `Codes/common/universe.py`.
