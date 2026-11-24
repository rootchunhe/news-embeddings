# Paper analyses (`Codes/3_paper`)

Flat scripts, one per analysis, run from this folder with `../../env/bin/python <script> --model roberta-large|Llama-2-13b-hf`.
The environment is `env` at the project root with the packages of `Codes/requirements.txt` (top-level README).
`1_category.py` and `1_drift.py` take no `--model` (they use every encoder's forecasts), `4_groups.py` uses none, `2_networks.py` takes
`--encoders` (one or several, one network run for all of them) and `--first` (the windows' first year). Llama runs need a
compute node (8 threads, up to 200 GB) and `2_embed.py` a GPU: submit them through your cluster's scheduler. A script writes `Results/paper/<script>/<model>/*.csv` (model-free scripts:
`Results/paper/<script>/`; `4_didisheim` writes `4_didisheim/{attribution,timing,industry}/<model>/`, `4_monthly --template` `4_monthly_T/<model>/`, `2_networks` `2_networks/<encoders joined by +>/`) plus a `run.json` manifest, first into a `.staging` folder that replaces the final one only when the run
succeeded; a partial `--years` run (`2_probe`, `2_ridge`, `2_spectrum`, `2_representation`, `2_networks`: the smoke test) writes its results and caches
under `Results/paper/_tests/<years>/` and never touches `Data/clean`. Intermediates go to `Data/clean/` with a `run_<name>.json`
manifest. Every manifest carries the sample definition (`common/universe.py`) and the specification groups the product depends on
(`common/runinfo.SPECS`: `ridge`, `monthly`, and `paper` = `lib.constants.PAPER_SPEC`); every reader, `report.py` included, refuses a
product of another sample or other parameters.
`report.py` loads the tables and draws the figures (`Results/paper/figures/*.pdf`); `notebook.py` builds and executes
`paper_results.ipynb`; `latex.py` writes every table of the paper (except the framework's calibration tables, Tables 12-13 and Appendix D, which `Framework/` writes), the appendix and the key replication tables with their notes to
`Results/paper/tex/` plus `tables.tex`, compiles `tex/exhibits.pdf` (every table and figure with its title and note) and stops on a table it cannot build or exits with status 1 on a compile problem.
Every table file sets the label `tab:<table key>`; the paper's own document defines `fig:<figure key>` for its figures.

## The framework

Every story's embedding is split with the two-way projection of its test year T, estimated on the six years before T
(`lib.ridge.firm_years`; the ridge coefficients use the training years T-8..T-3) (`3_decompose.py`): content `e = C + F + U` (the
two-way least-squares category and firm effects, the other text; a firm effect needs `MIN_STORIES` headlines and news in
`MIN_FIRM_YEARS` calendar years of the window, each year of a shorter window, and is 0 otherwise: `lib.ridge.identified`) and timing: the level of C and of U is the mean of the firm's monthly
averages over its news months among the previous `LEVEL_WINDOW` months (0 without such a month), F belongs entirely to the level,
the deviation is the rest; the
projection and the coefficients of T are applied to the history T needs (T - 1) as well. Cells `cL cD yF uL uD`; level `cL + yF + uL`,
deviation `cD + uD`. The contribution of a part to a linear predictor is its within-period covariance with the return
(`lib.evaluate.predictive_cov`, `contribution_table`); the parts add up to the predictor's.

- Daily CKX ridge: `1_category`, `1_drift`, `2_probe`, `2_ridge`, `3_decompose`, `3_surprise`, `2_perturb`, `2_spectrum`, `2_representation`, `2_networks`, `3_by_group`, `3_factors`.
- Horizons: `4_horizon` (daily ridge by day), `4_monthly_ridge` (the daily coefficients on the firm-month average, and a
  monthly ridge with the CKX estimator; by horizon h, the h-th day after the trade or month after the signal month).
- DKPT strategy: `4_monthly` (the parts C, F and the level of C on the Didisheim firm-months), `4_didisheim` (the replication's
  strategy decomposed exactly into the cells; each input as its own strategy, also with GICS-4 industry indicators in the residualization (`--industry`); appendix: trade-month timing).
- Robustness (Tables C1-C7): Tables 2, 6, 7, 8 and A16 from the January after each model's pretraining cutoff, saved next to the
  full results as `*_post` by `1_category`, `2_probe`, `4_horizon`, `4_monthly_ridge` and `4_didisheim` (`3_decompose`: one more period per segment); Tables 4 and 5 on the windows that
  start in that January (`2_representation`'s windows from each such year, `2_networks.py --first <that year>`). The signal-year segments are `lib.constants.SEGMENTS`.
- Robustness encoder (Table 10, Tables B1-B7): ChronoGPT pretrained on text to 1999 (`lib.constants.ROBUST_MODELS`, `common/chronogpt.py`),
  run through the same steps as the main encoders (`--model chrono-gpt-v1-19991231`, the headlines with `03_embed.py --model
  manelalab/chrono-gpt-v1-19991231`); Table 10 sets every encoder's main results side by side (`report.encoder_summary`); Tables 2, 6, 7, 8, A16 by segment (2004-2025, 2023-2025: the main tables' years and from the January after
  Llama-2's cutoff, `*_full`, `*_post`; RoBERTa's 2023-2025 as `*_split`), 4 (`2_representation.py`) and 5 (`2_networks.py` from 2000 and from 2023); Figure 5 plots the cumulative returns of its MSRR strategy and of the strategy's level and deviation next to the main encoders' (`figA_cutoff_returns_daily`: the daily forecast portfolio), Figure 6 its rolling daily ICs of the level and the deviation, and `figA_cutoff_monthly` their rolling monthly ICs from the MSRR signal. `1_category` runs every encoder's Fama-MacBeth on the same firm-days, by segment (`fm_<segment>`).
  `4_didisheim` also saves the strategy's monthly returns by part (`monthly.csv`) and the monthly rank ICs of the level and the
  deviation (`ic_monthly.csv`), `4_horizon` their daily rank ICs (`ic_daily.csv.gz`; next-day ICs within size groups and news-intensity terciles, `ic_daily_groups.csv.gz`, Table 11) and the daily covariances of the forecast and of every part with the return (`cov_daily.csv.gz`).

Two samples: each replication on its paper's period (CKX 2004-2019H1, Didisheim signal months 2000-2022) and the extended sample of the test
years `lib.constants.TEST_YEARS` (2004-2025), which every paper analysis uses.

## Reruns

The inputs are never touched: `Data/raw`, `Data/headlines`, `Data/embedding`, `Data/Loughran-McDonald_MasterDictionary_1993-2023.csv`
and, in `Data/clean`, `clean_<Y>.parquet` and `nyse_breakpoints.parquet` (the data layer, `Codes/1_data`; the breakpoints are built on first use by `common/universe.py`; delete them after new CRSP data).

- Full rerun (after a change to the replication, the sample or the specification): delete everything else in `Data/clean` (after new
  CRSP data also `nyse_breakpoints.parquet`)
  and all of `Results/replication` and `Results/paper`, then run every step below.
- Routine rerun (a change to the paper analyses only): keep the GPU and replication intermediates in `Data/clean`
  (`pred_<m>.parquet`, `ridge_fit_<m>.npz`, `sent_<m>.parquet`, `firm_month_emb_<m>[_trade].npz`, `template_emb_<m>.npz`, `perturb_texts.parquet`,
  `perturb_texts_emb_<m>.npz`, each with its `run_*.json`, `categories.txt` and the `ridge_stats/` cache, keyed inside its files), delete the other derived files in `Data/clean` (the
  3_surprise story cache, for one, is rebuilt only when an input file changes, not when its code does) and `Results/paper`, and run steps 2-12
  (step 5: `2_perturb.py` scoring only; after a change of the paper's specification also `2_perturb.py --build`, whose texts must keep
  the sha1 the rewrite embeddings record, so no GPU step is needed).
  Every reader checks the manifests, so a kept file that no longer matches the sample or the specification is refused.

Steps run per model (`roberta-large`, `Llama-2-13b-hf`) unless noted; a step needs the earlier ones. Exhibits are named by their `latex.py` keys
(the paper's own numbering may differ).

Hardware: disk about 160 GB for `Data/embedding` (three encoders) and 120 GB for `Data/clean`. A GPU only for the embeddings (`1_data/03_embed.py`,
`2_embed.py`) and, recommended, the logistic model (`01_regress.py --logit`); every other step runs on CPU (memory and threads: above). Results are not bit-identical across hardware: `03_embed.py` and `2_embed.py` use float16 on a GPU and float32 on a CPU, and BLAS
thread counts can change the last digits; bootstraps and other random draws have fixed seeds.

| step | where | command | produces |
|---|---|---|---|
| 0 | `2_replication/ckx` | `01_regress.py --model <hf name>` for both models and the robustness encoder (`--logit`, GPU recommended, for the replication's sentiment tables only), then `02_tables.py` (both models at once), `03_notebook.py` | `Data/clean/pred_<m>.parquet`, `ridge_fit_<m>.npz` (`sent_<m>.parquet`); `tabR1_ckx_corr` to `tabR5_ckx_alphas` |
| 0 | `2_replication/didisheim` | `01_firm_month_embeddings.py` (and `--month trade --suffix _trade` for Table 3's level $E_L$ and the timing appendix), `02_msrr.py` (the replication period, signal months 2000-2022), `02_msrr.py --end 2025-12 --results ../../../Results/replication/didisheim_extended` (the extended sample that `4_didisheim` reads), `04_seasonality.py`, `03_notebook.py` | `Data/clean/firm_month_emb_<m>[_trade].npz`; `tabR6_did_msrr`, `tabR7_did_sorts`, the strategy's weights |
| 1 | here | `2_embed.py --model <m>` (GPU) | `Data/clean/template_emb_<m>.npz` |
| 2 | here | `2_probe.py` (builds the ridge statistics cache) | `tab2` Panel B, `tabA_probe`, `tabC_post_residual` Panel B |
| 3 | here | `1_category.py`, `1_drift.py` (model-free, every encoder's forecasts) | `tab1`, `tab2` Panel A, `fig1_drift`, `figA_stability`, `figA_stability_coarse`, `tabA_drift` to `tabA_spanning`, `tabC_post_residual` Panel A, `figA_drift_size` |
| 4 | here | `2_ridge.py` | `tabA_centroid` |
| 5 | here | `2_perturb.py --build` (model-free) -> `2_embed.py --model <m> --texts ../../Data/clean/perturb_texts.parquet` (GPU) -> `2_perturb.py` | `tabA_rewrites` |
| 6 | here | `2_spectrum.py --perturb`, `2_representation.py`; then `2_networks.py` (no `--model`): `--encoders roberta-large,Llama-2-13b-hf --first 2000`, `--encoders roberta-large --first 2020`, `--encoders roberta-large --first 2023`, `--encoders Llama-2-13b-hf --first 2023`, and for Table 10 and the B tables `--encoders chrono-gpt-v1-19991231 --first 2000` and `--first 2023` | `tabA_spectrum`, `tab4`, `tab5`, `tabC_post_representation`, `tabC_post_networks`, the network rows of `tab10`, `figA_spectrum`, the subspace columns of `tabA_rewrites` |
| 7 | here | `3_decompose.py` | `Data/clean/decomposition_<m>.parquet`; `tab6`, `tabA_decomposition`, `tabC_post_split` |
| 8 | here | `3_surprise.py` (build the shared cache with the first model before the second starts) | `tab3` |
| 9 | here | `3_by_group.py`, `3_factors.py`, `4_horizon.py`, `4_monthly_ridge.py` (after the extended Didisheim run) | `tabA_factors`, `tabA_groups`, `tabA_horizons`, `tab7`, `fig2`, `figA_persistence`, `fig6_cutoff`, `figA_cutoff_returns_daily`, `tabC_post_horizons`, `figA_bridge_size`, `tab11` |
| 10 | here | `4_monthly.py`, `4_monthly.py --template`, `4_groups.py` (model-free) | `Data/clean/firm_month_parts_<m>.npz`, `firm_month_emb_<m>_T.npz`, `firm_month_groups.npz` |
| 11 | here | `4_didisheim.py` (after `4_groups.py`); `4_didisheim.py --timing`; `4_didisheim.py --industry` (RoBERTa, Llama-2) | `tab8`, `fig3_attribution`, `fig4_inputs`, `fig4_inputs_noci`, `figA_persistence_strategy`, `fig5_cutoff_returns`, `figA_cutoff_monthly`, `tabA_reoptimized`, `tabA_constructions`, `tab9`, `figA_old_news`, `tabC_post_attribution`, `tabC_post_constructions`, `figA_attribution_size`; from `--industry`: the industry rows of `tabA_constructions` and the lower panels of `fig4_inputs` |
| 12 | here | `notebook.py`, `latex.py` (needs `pdflatex` and `bibtex` on the path: TeX Live 2026 in our runs, with the packages geometry, booktabs, amsmath, amssymb, adjustbox, graphicx, rotating and natbib) | `paper_results.ipynb`, figures, `tex/` |

Table 10 and Tables B1-B7 need, with `--model chrono-gpt-v1-19991231`: `01_regress.py`, `01_firm_month_embeddings.py`, the extended
`02_msrr.py` (step 0), `2_embed.py`, `2_probe.py`, `2_representation.py`, `2_networks.py` (above), `3_decompose.py`, `4_horizon.py`, `4_monthly_ridge.py`,
`4_monthly.py` (also `--template`) and `4_didisheim.py`; step 3 runs once, after every encoder's forecasts exist.
