# Chen–Kelly–Xiu, *Expected Returns and Large Language Models*: replication

Sample: `Codes/common/universe.py` (the paper's rules plus a RavenPack event category and a NYSE 5th-percentile cap floor; see its docstring).
Inputs: `Data/clean/clean_<Y>.parquet` (block 1), embeddings via `Codes/common/emb.py`.

- `01_regress.py`: rolling ridge (next open-to-open return), output `Data/clean/pred_<model>.parquet` and the fitted ridge of
  every window `Data/clean/ridge_fit_<model>.npz` ($\mu$, $\bar y$, $\lambda$, $\beta$; read by `Codes/3_paper`):
  `../../../env/bin/python 01_regress.py --model <hf name>`. With `--logit`, a separate run, the logistic model (sign of the 3-day
  return) for the v2024 sentiment tables of `02_tables.py`, output `Data/clean/sent_<model>.parquet`. CPU or GPU (`--device`).
- `02_tables.py`: the implemented tables of the paper from the forecasts, named by the paper's table / figure number, one folder per
  model and period (default: paper 2004-2019H1, extended 2004-2025), written to a staging folder and published on success, with a `run.json`
  manifest; also `Results/replication/ckx/cross_model.csv` (with `run_cross_model.json`). Leg returns are annualized, in percent. Portfolio and cost conventions in `Codes/common/portfolio.py`.
- `03_notebook.py` + `report.py`: build `CKX_replication.ipynb` (summary table, then the paper's tables one by one with plots).

Implementation choices not specified by the paper: data start 2000 (2–5 training years for 2004–2007), the paper's penalty grid
1e-1..1e8 read as a per-observation penalty of c/P on the centred embedding (no standardization; the logistic solver alone
standardizes internally), validation criterion = firm-day Spearman, no refit on train+validation, firm-day signal = mean of
story forecasts, 09:00 ET cutoff for same-day trading, VW by previous-day cap, the ridge target winsorized at ±`Y_CLIP` = 0.5, and
at least `MIN_NAMES` = 20 firm-days on a day for the validation Spearman, a portfolio or a daily correlation (`MIN_NAMES_LARGE` = 10
for the correlation above the NYSE median, `MIN_NAMES_DOUBLE_SORT` = 25 for the 5x5 sort).
