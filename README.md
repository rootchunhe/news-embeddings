# The Information Content of News Embeddings

Code for the empirical analysis in `The Information Content of News Embeddings`. The repository holds the complete pipeline, from the
data download to every result table and figure of the paper.

Paper link: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7751285)

LLM embeddings of news predict stock returns at horizons from one day to more than a year. The paper asks which information in the
embeddings carries this predictability. In a simple framework, predictability arises either from price adjustment after news events
or from persistent differences in expected returns. Two decompositions of the embedding separate the information behind each source:
a content decomposition into event-category, firm-identity and idiosyncratic components, and a timing decomposition into the firm's
historical embedding level and the deviation from it. The code builds both decompositions for Dow Jones headlines encoded with
RoBERTa-large and Llama-2 (13B) and attributes the return predictability of embedding forecasts, at daily and monthly horizons, to
the components.

In the code:

- `C`, `F`, `U` denote the content components of a headline embedding, $`E = C + F + U`$: the event-category and firm effects of a
  two-way least-squares projection of the embedding on event-category and firm indicators, and the remaining idiosyncratic text
  (`U` is $`I`$ in the paper). The projection of test year $`T`$ is estimated on the six years before $`T`$
  (`lib.ridge.firm_years`); a firm without enough news in that window has $`F = 0`$ (`lib.ridge.identified`).
- The level of `C` and of `U` is the mean of the firm's monthly averages over its news months among the previous 12 months
  (`LEVEL_WINDOW`); `F` belongs to the level; the deviation is the rest. The five cells are `cL`, `cD`, `yF`, `uL`, `uD`
  ($`C_L`$, $`C_D`$, $`F`$, $`I_L`$, $`I_D`$ in the paper); the level `lev` is `cL + yF + uL` ($`E_L`$), the deviation `dev` is
  `cD + uD` ($`E_D`$).
- A component's contribution to a linear forecast is its covariance with the return within each period (`lib.evaluate.predictive_cov`,
  `contribution_table`); the contributions add up to the forecast's.
- The encoders are `roberta-large`, `Llama-2-13b-hf` and the robustness encoder `chrono-gpt-v1-19991231` (ChronoGPT, pretrained on
  text published before 2000).

The main empirical applications in the code are:

- event drift and event-category benchmarks for short-horizon predictability
- news surprises and firm fundamentals measured from the embeddings
- the decomposition of next-day predictability and of predictability across trading-day horizons
- the decomposition of monthly predictability: monthly aggregation, the residualized-embedding strategy, and lagged inputs
- potential look-ahead bias: ChronoGPT, the samples after the pretraining cutoffs, rank ICs across Llama-2's cutoff
- the calibration of the framework (`Framework/`)

The daily forecasts follow the ridge of Chen, Kelly and Xiu (2026, CKX) and the monthly strategy the residualized-embedding
strategy of Didisheim, Kelly, Pourmohammadi and Tian (2026, DKPT); `Codes/2_replication/` builds both on our data, and the
decompositions start from them.

## Repository Layout

The analyses are flat scripts, one per analysis. `Codes/3_paper/notebook.py` builds and executes a notebook whose cells draw every
figure through `report.py`; `Codes/3_paper/latex.py` writes every table except Tables 12–13 and D.2–D.5 (`Framework/`) and Table D.1
(typed in the paper). The notebooks are generated and not in the repository. `Codes/1_data`, `Codes/2_replication/ckx`,
`Codes/2_replication/didisheim`, `Codes/3_paper` and `Codes/4_llm_qa` each have a README with the details.

- `Codes/1_data/`: WRDS pull, unique headlines, embeddings, the cleaned story panel
- `Codes/2_replication/ckx/`, `Codes/2_replication/didisheim/`: the CKX and DKPT replications (`didisheim/newsshock.py` holds the
  DKPT pipeline that `Codes/3_paper` imports)
- `Codes/3_paper/`: the paper's analyses; `lib/` holds their shared helpers, `report.py` loads the results and draws the figures
- `Codes/4_llm_qa/`: the encoders' answers to factual prompts (a footnote on ChronoGPT)
- `Codes/common/`: modules shared by all blocks:
  - `universe.py`: the sample definition
  - `crsp.py`: CRSP daily panel and open-to-open returns
  - `jkp.py`: JKP monthly panel
  - `emb.py`: access to the headline embeddings
  - `ridge.py`: the ridge kernel of the CKX replication and the paper
  - `portfolio.py`: portfolio construction and performance statistics
  - `factors.py`: daily factor returns
  - `runinfo.py`: output folders and run manifests
  - `chronogpt.py`: the ChronoGPT network
  - `config.py`: shared thresholds
  - `paper_numbers.py`: CKX and DKPT benchmark numbers quoted in the replication notebooks
- `Results/`: the outputs of our last full run; the analysis folders carry `run.json` manifests, figures are in
  `Results/paper/figures/`, table files in `Results/paper/tex/`, and `Results/paper/tex/exhibits.pdf` collects every table and
  figure with its title and note. The DKPT strategy weights (`weights.npz`) are not included.
- `Framework/`: the calibration (Section 7.3 and Appendix D), its inputs and outputs, and `framework.tex`, a longer standalone
  version of the framework; the `cal_*` files not listed below are the figures of `framework.tex`, `calibration.json` and
  `calibration_fit.csv` the fitted values

Scripts and paper mapping (run from `Codes/3_paper/` unless noted; table keys are files `Results/paper/tex/<key>.tex`, figures are
`Results/paper/figures/<name>.pdf`):

| Paper section | Script(s) | Exhibits |
| --- | --- | --- |
| Section 2, `Event Drift and Short-Horizon Predictability` | `1_category.py`, `1_drift.py`, `2_probe.py` | Figure 1 (`fig1_drift_analyst-ratings`), Tables 1–2 (`tab1`, `tab2`; Panel B from `2_probe.py`) |
| Section 4.2, `Measuring News Surprises` | `3_surprise.py` | Table 3 |
| Section 4.3, `Measuring Firm Fundamentals` | `2_representation.py`, `2_networks.py` | Tables 4–5 |
| Section 5.1, `Decomposing Next-Day Predictability` | `3_decompose.py` | Table 6 |
| Section 5.2, `Predictability Across Trading-Day Horizons` | `4_horizon.py`, `4_monthly_ridge.py` | Figure 2 (`fig2a`) |
| Section 6.1, `Monthly Aggregation of News Embeddings` | `4_horizon.py`, `4_monthly_ridge.py` | Table 7 |
| Section 6.2, `Persistent Firm Information after Characteristic Residualization` | `4_monthly.py`, `4_groups.py`, `4_didisheim.py` | Table 8, Figure 3 (`fig3_attribution`) |
| Section 6.3, `Long-Horizon Predictability from Distant Information` | `4_didisheim.py` (with `--industry` for the lower panels of Figure 4) | Figure 4 (`fig4_inputs`), Table 9 |
| Section 7.1, `Robustness to Pretraining Cutoffs` | the ChronoGPT steps listed at the end of `Codes/3_paper/README.md` | Table 10 |
| Section 7.2, `Predictability across the Pretraining Cutoffs` | `4_didisheim.py`, `4_horizon.py` | Figures 5–6 (`fig5_cutoff_returns`, `fig6_cutoff`), Table 11 |
| Section 7.3 and Appendix D, calibration | `Framework/cutoff_inputs.py`, `Framework/calibration.py` | Tables 12–13 (`cal_cutoff.tex`, `cal_checks.tex`), Tables D.2–D.5 (`cal_params.tex`, `cal_shares.tex`, `cal_targets.tex`, `cal_variants.tex`), Figure D.1 (`cal_cutoff_placebo.pdf`); Table D.1 is typed in the paper |
| Appendix A, `Additional Tables and Figures` | the scripts of the sections above | Tables A.1–A.16 (`tabA_*`), Figures A.1–A.14 (`figA_*`, `fig2b`, `fig2c`) |
| Appendix B, `Results with ChronoGPT` | the steps of Section 7.1 | Tables B.1–B.7 (`tabB_*`) |
| Appendix C, `Results after the Pretraining Cutoffs` | `1_category.py`, `2_probe.py`, `3_decompose.py`, `4_horizon.py`, `4_monthly_ridge.py`, `4_didisheim.py`, `2_representation.py`, `2_networks.py` | Tables C.1–C.7 (`tabC_post_*`) |

Appendix A in detail:

| Exhibits | Key or file | Script |
| --- | --- | --- |
| Tables A.1–A.6 | `tabA_drift`, `tabA_subsamples`, `tabA_costs`, `tabA_windows`, `tabA_mz`, `tabA_spanning` | `1_drift.py`, `1_category.py` |
| Table A.7 | `tabA_rewrites` | `2_perturb.py`, `2_spectrum.py` |
| Table A.8 | `tabA_centroid` | `2_ridge.py` |
| Table A.9 | `tabA_probe` | `2_probe.py` |
| Table A.10 | `tabA_decomposition` | `3_decompose.py` |
| Table A.11 | `tabA_factors` | `3_factors.py` |
| Table A.12 | `tabA_spectrum` | `2_spectrum.py` |
| Table A.13 | `tabA_groups` | `3_by_group.py` |
| Table A.14 | `tabA_horizons` | `4_horizon.py`, `4_monthly_ridge.py` |
| Tables A.15–A.16 | `tabA_reoptimized`, `tabA_constructions` | `4_didisheim.py` (`--industry` for the industry rows of A.16) |
| Figure A.1 | `figA_stability`, `figA_stability_coarse` | `1_category.py` |
| Figure A.2 | `figA_drift_size` | `1_drift.py` |
| Figure A.3 | `figA_spectrum` | `2_spectrum.py` |
| Figures A.4–A.9 | `fig2b`, `fig2c`, `figA_persistence_a`, `figA_persistence_b`, `figA_persistence_c`, `figA_bridge_size` | `4_horizon.py`, `4_monthly_ridge.py` |
| Figures A.10–A.13 | `figA_attribution_size`, `figA_persistence_strategy`, `figA_old_news`, `figA_cutoff_monthly` | `4_didisheim.py` |
| Figure A.14 | `figA_cutoff_returns_daily` | `4_horizon.py` |

Not in the paper: the replication tables `tabR1_ckx_corr` … `tabR7_did_sorts`, the other `fig1_drift_*` event groups and
`fig4_inputs_noci`. The paper's tables are pasted from the table files.

The code expects the following directories under the repository root:

```text
<repo root>/
├── Codes/
├── Data/        # not in the repository; built by Codes/1_data
├── Results/
├── Framework/
└── env/         # the Python environment
```

Paths in the code are built relative to the repository root, so this layout matters.

## Conceptual Mapping

The empirical workflow follows the paper closely:

1. Pull RavenPack stories, CRSP, JKP, IBES and factor data; keep stories tagging a US company with relevance 100.
2. Embed every unique headline with each encoder (mean-pooled last layer).
3. Build the story panel: trade day, open-to-open returns, event categories, duplicate flags; the sample rules are in
   `common/universe.py`.
4. Daily forecasts (CKX): a rolling ridge of the next open-to-open return on the embedding, trained on years $`T-8`$ to $`T-3`$ and
   validated on $`T-2`$ to $`T-1`$.
5. Monthly strategy (DKPT): firm-month embeddings, residualized each month on JKP characteristics, managed factors, and the MSRR
   strategy.
6. Split every embedding into `C`, `F`, `U` and into level and deviation, and attribute the daily forecast's covariance with returns
   and the monthly strategy's returns to the components, across horizons.
7. Repeat the main steps with ChronoGPT and on the samples after each encoder's pretraining cutoff; calibrate the framework to the
   results.

## Data Requirements

The pipeline relies on WRDS pulls and a few files placed by hand. None of the data are in the repository.

WRDS access is required for:

- RavenPack Analytics, Dow Jones stories (`rpna.rpa_full_equities_<year>` with `provider_id = 'DJ'`, `rpna.wrds_rpa_company_mappings`,
  `rpna.wrds_rpa_company_mappings_hist`, `rpna.rpa_taxonomy`); the headlines are licensed and cannot be redistributed
- CRSP daily stock file, CIZ format (`crspm.wrds_dsfv2_query`), stock names and delisting events (`crsp.stocknames`, `crsp.stkdelists`)
- Fama–French factors (`ff.fivefactors_daily`, `ff.factors_daily`)
- Jensen–Kelly–Pedersen Global Factor Data (`contrib.global_factor`)
- IBES actuals, surprises and long-term growth forecasts (`ibes.actu_epsus`, `ibes.surpsumu`, `ibes.statsumu_epsus`) and the
  IBES–CRSP link (`wrdsapps.ibcrsphist`)

Manual download step (sources and file names in `Codes/1_data/README.md`), into `Data/raw/factors/` and `Data/`:

- Ken French data library: daily short- and long-term reversal factors
- Daniel–Mota–Rottke–Santos hedged factors (Kent Daniel's website)
- Hou–Mo–Xue–Zhang q5 factors (global-q.org)
- Loughran–McDonald master dictionary (sraf.nd.edu)

Encoders are downloaded from Hugging Face at pinned revisions (`Codes/4_llm_qa/README.md`): `roberta-large`,
`NousResearch/Llama-2-13b-hf` (under Meta's Llama 2 Community License) and `manelalab/chrono-gpt-v1-19991231`.

## Environment

The code is written in Python 3.11. `Codes/requirements.txt` lists the package versions of our runs; install them into an environment
at `env/` in the repository root, for example:

```bash
python3.11 -m venv env && env/bin/pip install -r Codes/requirements.txt
```

The commands in the READMEs and docstrings call `env/bin/python` by its path relative to the folder they run in. `latex.py` also
needs `pdflatex` and `bibtex` (packages in `Codes/3_paper/README.md`).

Hardware:

- Disk: about 7 GB for `Data/raw`, 160 GB for `Data/embedding` (three encoders), 120 GB for `Data/clean`.
- GPU: the embeddings (`1_data/03_embed.py`, `3_paper/2_embed.py`) and, recommended, the CKX logistic model (`01_regress.py --logit`).
  On one NVIDIA H100 (80 GB), embedding the 11 million unique headlines took about 5 hours with Llama-2 (13B), 1 hour with
  ChronoGPT and 20 minutes with RoBERTa-large.
- CPU: every other step. Llama-2 steps need up to 200 GB of memory; the longest (the CKX ridge for Llama-2 on CPU, `3_surprise.py`
  for Llama-2) take 3–7 hours.

Results are not bit-identical across hardware: the embeddings are float16 on a GPU and float32 on a CPU, and BLAS thread counts can
change the last digits. Random draws use fixed seeds.

## Build Pipeline

To build the data from scratch, from `Codes/1_data/`:

```bash
../../env/bin/python 01_pull.py --wrds_user <user>
../../env/bin/python 02_headlines.py
../../env/bin/python 03_embed.py --chunk <N> --model <hf name>
../../env/bin/python 04_clean.py
../../env/bin/python 05_check.py
```

Notes:

- WRDS credentials are required
- `03_embed.py` embeds one chunk of 500,000 headlines per call, on a GPU: 23 chunks for each of the three encoders
- the hand-placed files above still need to be downloaded manually

## Recommended Execution Order

1. Build the data (above).
2. Run the replications, `Codes/2_replication/ckx/` and `Codes/2_replication/didisheim/` (step 0 of `Codes/3_paper/README.md`).
3. Run steps 1–12 of `Codes/3_paper/README.md` and the ChronoGPT steps listed at its end.
4. Run `Codes/4_llm_qa/completions.py`.
5. Run the calibration:
   - from `Codes/3_paper/`: `../../env/bin/python ../../Framework/cutoff_inputs.py`
   - from `Framework/`: `../env/bin/python calibration.py`, then `../env/bin/python calibration.py variants`

Every script of `Codes/2_replication` and `Codes/3_paper` writes into a staging folder that replaces its output only when the run
succeeds, with a manifest of the sample and specification; every reader refuses a product of another sample or specification.

## Important Functions

- `common.universe.in_sample(...)`: the sample definition
- `common.crsp.o2o(...)`: open-to-open returns (exact for splits, approximate on ex-dividend days)
- `lib.ridge.firm_years(T)`, `lib.ridge.identified(...)`: the projection window of test year $`T`$ and which firms get a firm effect
- `lib.evaluate.predictive_cov(...)`, `lib.evaluate.contribution_table(...)`: component contributions to a forecast's covariance
  with returns
- `newsshock.msrr(...)`: the MSRR strategy (in `Codes/2_replication/didisheim/`)
- `report.encoder_summary(...)`: Table 10
- `common.runinfo`: manifests and the sample and specification checks

## Note on ChronoGPT

The ChronoGPT network in `Codes/common/chronogpt.py` is copied from the model's `ChronoGPT_inference.py` on Hugging Face
(`manelalab/chrono-gpt-v1-19991231`), whose architecture is derived from modded-nanogpt; both are under the MIT license, and the
notices are in the file:

He, Songrun, Linying Lv, Asaf Manela, and Jimmy Wu (2026). "Chronologically Consistent Large Language Models." *Journal of
Financial Economics* 185, 104374. [doi:10.1016/j.jfineco.2026.104374](https://doi.org/10.1016/j.jfineco.2026.104374).

## Output Locations

- `Data/`: downloaded data, embeddings and intermediates (`Data/clean/`)
- `Results/replication/`: the CKX and DKPT replications
- `Results/paper/<script>/<model>/`: result tables (csv) of the paper's scripts
- `Results/paper/figures/`, `Results/paper/tex/`: figures and table files
- `Results/llm_qa/`: the encoders' answers
- `Framework/`: calibration inputs, tables and figures
