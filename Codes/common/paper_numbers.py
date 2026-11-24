"""Numbers from the two papers used as benchmarks in the notebooks (one place, cited by table).

CKX = Chen, Kelly, Xiu, Expected Returns and Large Language Models, 2026 version unless marked v2024.
DKPT = Didisheim, Kelly, Pourmohammadi, Tian, The Inefficient Pricing of News."""

# CKX Table 3 (mean daily cross-sectional Pearson correlation, %, and the subset above the NYSE median), Table 5 (quintile long-short,
# EW / VW Sharpe), Tables 20-21 (annualized average return, alpha, t, R_ROE loading); v2024 Table 5 (accuracy), Table 6 (sentiment EW Sharpe)
CKX = {
    "Llama-2-13b-hf": {"Pearson corr %": 2.46, "Pearson corr, over NYSE median %": 1.24, "EW quintile L-S Sharpe": 4.36, "EW quintile L-S return %/yr": 49.6,
                       "VW quintile L-S Sharpe": 1.02, "VW quintile L-S return %/yr": 11.4, "EW alpha, all factors %/yr": 48.3, "t(EW alpha)": 16.7,
                       "VW alpha, all factors %/yr": 11.4, "t(VW alpha)": 4.1, "EW loading on R_ROE": 0.43, "sentiment accuracy %": 53.8, "EW sentiment L-S Sharpe": 4.16},
    "roberta-large": {"Pearson corr %": 1.88, "Pearson corr, over NYSE median %": 0.92, "EW quintile L-S Sharpe": 3.69, "EW quintile L-S return %/yr": 38.6,
                      "VW quintile L-S Sharpe": 0.92, "VW quintile L-S return %/yr": 10.2, "EW alpha, all factors %/yr": 37.5, "t(EW alpha)": 14.4,
                      "VW alpha, all factors %/yr": 10.2, "t(VW alpha)": 3.9, "EW loading on R_ROE": 0.31, "sentiment accuracy %": 53.7, "EW sentiment L-S Sharpe": 3.75},
}
# CKX results reported for LLaMA3 only: Table 6 (decile), Table 8 (EWCT, articles and alerts), Table 18 (delays, EW quintile Sharpe)
CKX_LLAMA3 = {"EW decile L-S Sharpe": 4.28, "VW decile L-S Sharpe": 1.25, "EWCT gamma=0.1 gross Sharpe": 1.93, "EWCT gamma=0.1 net Sharpe": 0.77,
              "EWCT gamma=0.3 net Sharpe": 0.53, "delay-2 EW Sharpe": 1.35, "delay-3 EW Sharpe": 0.98}
CKX_EWCT = {"articles gross": {0.1: 1.93, 0.2: 2.85, 0.3: 3.46, 0.5: 4.16}, "articles net": {0.1: 0.77, 0.2: 0.72, 0.3: 0.53, 0.5: 0.04},
            "alerts gross": {0.1: 2.13, 0.3: 3.70, 0.5: 4.42}, "alerts net": {0.1: 1.26, 0.3: 1.65, 0.5: 1.66}}
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# CKX Table 9: net Sharpe ratio of the EW quintile long-short under EWCT by gamma, Panel A articles and Panel B alerts
CKX_NET = {
    "Llama-2-13b-hf": {"articles": dict(zip(GAMMAS, [0.68, 0.64, 0.47, 0.25, 0.02, -0.22, -0.44, -0.66, -0.87])), "alerts": dict(zip(GAMMAS, [1.19, 1.51, 1.65, 1.69, 1.67, 1.6, 1.51, 1.41, 1.29]))},
    "roberta-large": {"articles": dict(zip(GAMMAS, [0.35, 0.11, -0.22, -0.57, -0.9, -1.21, -1.51, -1.78, -2.02])), "alerts": dict(zip(GAMMAS, [1.02, 1.31, 1.36, 1.3, 1.19, 1.06, 0.93, 0.8, 0.66]))},
}

# DKPT Figure 4 and text (MSRR Sharpe: raw embeddings, residual on a constant, on the JKP characteristics, + industry, Section 6.5),
# Table 7 Panel A (EW H-L Sharpe of the MSE sorts), Figure 7 (size groups, JKP residual)
DKPT = {"Raw": 1.1, "Constant": 1.7, "JKP": 3.1, "JKP + IND": 3.3, "Table 7 MSE, EW H-L": 2.68, "size: Mega + Large": 1.4, "size: Small + Micro": 2.7}
