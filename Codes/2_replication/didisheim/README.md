# Didisheim–Kelly–Pourmohammadi–Tian, *The Inefficient Pricing of News*: replication

Sample: `Codes/common/universe.py` (single-company, de-duplicated stories with a RavenPack event category, cap floor at the NYSE 5th percentile).

- `01_firm_month_embeddings.py`: firm-month mean of the headline embeddings (month of arrival, NY time, as in the paper: a story
  arriving after a month's last close belongs to that month while its first reaction falls in the next month's return) →
  `Data/clean/firm_month_emb_<model>.npz`; `--month trade --suffix _trade` keys by the trade day's month (the paper's timing appendix).
- `newsshock.py`: the pipeline's functions (panel, expanding z-score, monthly residualization on the 132 highest-coverage JKP
  characteristics, ranked each month across all JKP stocks, and GICS industry groups, managed factors, MSRR with leave-one-out ridge, CAPM alpha at 10% volatility with a
  Newey-West t, the MSE approach, delayed strategies); imported by `02_msrr.py` and by `Codes/3_paper` (`4_didisheim.py`,
  `3_surprise.py`, `4_monthly_ridge.py`).
- `02_msrr.py`: the paper's Figure 4 (MSRR on the embedding: raw, and residualized month by month on a constant, on the
  characteristics behind CAPM, FF3 and FF6, on the JKP characteristics, on GICS industry groups, or on both; the predicted parts and
  the control MSRR on the characteristics), Figure 7 (the residualizations without industry groups, within the JKP size groups), Table 7 (quintile sorts on the MSE forecast
  and on the MSRR weights, equal- and capped value-weighted), Figure 10 (trading delays of 1 to 36 months against the full JKP panel,
  weights fixed at the undelayed strategy's and re-optimized per delay; the paper names both for its Panel B and does not say which Panel A uses). Output
  `Results/replication/didisheim/<model>/` (staged, then published, with `run.json`, which lists the characteristics, and `weights.npz`: the undelayed
  MSRR weights by month). `--end 2025-12 --results .../Results/replication/didisheim_extended`
  is the extended run that `Codes/3_paper` reads. The paper's decomposition of this strategy and its other variants are
  `Codes/3_paper/4_didisheim.py`.
- `04_seasonality.py`: delayed returns (both weightings) against a same-calendar-month seasonal factor (Newey-West t).
- `03_notebook.py` + `report.py`: build `Didisheim_replication.ipynb` (summary table, then the paper's Figures 4, 7, 10 and Table 7 in
  their layout; figures to `Results/replication/didisheim/figures/`).

Differences from the paper: headlines instead of full articles, RoBERTa-large / Llama-2-13B mean-pooled instead of
E5-Mistral-7B, signal months 2000-01 to 2022-12 (the paper's sample ends in December 2022), the cap floor, the Table 7 ridge refit every year (features standardized on the training rows,
demeaned target) with GCV instead of leave-one-out, and our choice of the JKP variables behind CAPM, FF3 and FF6
(`newsshock.FF_CHARS`; the paper names the characteristics, not the variables). The JKP excess return is used as reported (not
winsorized) in the managed factors, the strategy returns, the sorts and the target of Table 7's MSE forecast. Settings the paper does
not state: at least `MIN_MONTH_STOCKS` = 50 firm-months for a managed factor or a monthly sort, and Table 7's Avg and Std annualized
(the paper reports them monthly).
