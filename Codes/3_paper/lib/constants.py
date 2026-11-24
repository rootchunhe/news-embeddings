"""Names and constants shared by the paper scripts and the report (sample thresholds, the ridge convention and the window live in
common/config.py). Importing it registers the specification group "paper" (PAPER_SPEC) that the paper's manifests carry."""
from common import runinfo
from common.config import (CATEGORY_MEAN_SHRINKAGE, FIRST_YEAR, LEVEL_WINDOW, MIN_GROUP_STORIES, MIN_MONTH_STOCKS, MIN_NAMES, NW_LAGS_DAILY,
                           NW_LAGS_MONTHLY)

MODELS = {"roberta-large": "RoBERTa", "Llama-2-13b-hf": "Llama-2"}                # display names in cells and legends; notes use FULL at first mention
FULL = {"roberta-large": "RoBERTa-large", "Llama-2-13b-hf": "Llama-2 (13B)"}         # the main encoders' full names, as in the paper's model description
ROBUST_MODELS = {"chrono-gpt-v1-19991231": "ChronoGPT$_{1999}$"}           # Table 10, Tables B1-B7: an encoder pretrained on text to 1999 only
ENCODERS = MODELS | ROBUST_MODELS
SHORT = {"roberta-large": "RoBERTa", "Llama-2-13b-hf": "Llama-2", "chrono-gpt-v1-19991231": "ChronoGPT"}
FORECAST_LABEL = {m: f"{s} forecast" for m, s in SHORT.items()}            # the encoders' forecasts as signals and regressors
FORECAST = {"roberta-large": "yhat_rob", "Llama-2-13b-hf": "yhat_lla", "chrono-gpt-v1-19991231": "yhat_chr"}      # forecast column per encoder in the story panel
PRED = {FORECAST[m]: m for m in MODELS}
ROBUST_PRED = {FORECAST[m]: m for m in ROBUST_MODELS}
TEST_YEARS = range(2004, 2026)                                              # the extended sample: signal years 2004-2025
HISTORY_YEARS = range(FIRST_YEAR, TEST_YEARS[0])                            # clean years that only extend training histories
TAUS = [0, 1, 2, 3, 5, 11, 17, 23, 35]                                       # monthly delays tau = h - 1: the return of the h-th month after the signal month, h = 1, 2, 3, 4, 6, 12, 18, 24, 36
TAUS_FIG = list(range(36))                                                  # every delay h = 1..36: the fixed-rule decomposition of the MSRR strategy and the monthly forecasts' contributions (the figures)
PROJECTION_YEARS = 6                                                        # years before T in the two-way category/firm projection of test year T (lib.ridge.firm_years)
MIN_STORIES = 10                                                            # headlines a firm needs in a window for a firm effect and a firm-level input (Sarkar 2026: 10 articles)
MIN_FIRM_YEARS = 2                                                          # calendar years with news a firm needs in a window for a firm effect (fewer when the window is shorter)
LEVEL_LAG = 36                                                              # months back of the lagged level E_{L,tau-36} and the lagged firm identity F_{tau-36}
FACTORS = ["mktrf", "smb", "hml", "rmw", "cma", "umd", "strev", "ltrev"]   # the alphas: FF5, momentum, short- and long-term reversal
PERIOD = f"{TEST_YEARS[0]}-{TEST_YEARS[-1] % 100:02d}"
PERIODS = {PERIOD: (TEST_YEARS[0], TEST_YEARS[-1])}
REBUILD_TOL = 1e-3                                                          # rebuilt forecast vs the replication, in test-year forecast sd (float32 scoring)
H = 10                                                                      # last daily horizon of the drift / term-structure figures
SIZE_H = 63                                                                 # last daily horizon of the drift paths by size group (three months of trading days)
COVERAGE_GROUPS = ("low", "mid", "high")                                    # news-intensity terciles of the firm-days (Table 11)
BANDS = [(1, 8), (9, 32), (33, 128), (129, 512), (513, 100000)]             # principal-direction bands of the spectrum tables
SUBSPACE = 32                                                               # singular directions kept for the category / firm subspaces
LARGE = 2000                                                                # smaller matrix side above which an SVD is randomized

COMP = ["yC", "yF", "yU"]                                                   # content: E = C + F + U (event category, firm identity, idiosyncratic)
CELLS = ["cL", "cD", "yF", "uL", "uD"]                                      # content x timing: C = cL + cD, U = uL + uD (L: the firm's level over its news months among the previous LEVEL_WINDOW months)
TIMING = {"lev": ["cL", "yF", "uL"], "dev": ["cD", "uD"]}                   # the historical level and the deviation from it
PARTS = list(dict.fromkeys(COMP + CELLS + list(TIMING)))                      # every part once (yF is both a content part and a cell)
TS_DAILY = ["yH", "yUH"]                                                    # appendix: U split by the firm's own 20-day mean instead of the 12-month level
HISTORY_DAYS = 20                                                           # own news history: trailing window of the remainder, calendar days
LABELS = {"yC": "event category", "yF": "firm identity", "yU": "idiosyncratic", "cL": "event category: level", "cD": "event category: deviation",
          "uL": "idiosyncratic: level", "uD": "idiosyncratic: deviation", "lev": "historical level", "dev": "deviation from the level",
          "yH": f"idiosyncratic: own {HISTORY_DAYS}-day mean", "yUH": f"idiosyncratic: minus own {HISTORY_DAYS}-day mean",
          "yhat": "forecast"}
# 4_didisheim: the standalone strategies' inputs beyond the parts of LABELS, as stored in the results (displayed via report.STANDALONE_NAMES)
STANDALONE = {"embedding": "embedding", "lagged": f"historical level {LEVEL_LAG} months earlier", "lagged_F": f"firm identity {LEVEL_LAG} months earlier",
              "template": "category templates", "groups": "embedding with event-group controls"}
ARRIVAL_MONTH, TRADE_MONTH = "firm-months by arrival month", "firm-months by trade month"   # the replication's keying; 4_didisheim --timing

DRIFT_MIN_STORIES, DRIFT_TABLE_MIN_STORIES = 5000, 20000                   # stories a category needs for its drift path / its row of the drift table
SURPRISE = {"eval": [2008, TEST_YEARS[-1]], "min_train": 5000}   # 3_surprise: test years, stories per fit
SURPRISE_DUMMY_STORIES, SURPRISE_RANK_PRIOR = 100, 500                     # 3_surprise: training stories of an earnings-category dummy; prior announcements of a rank
GROUP_MIN_INCIDENCE = 0.01                                                  # 4_groups: event groups below this share of firm-months are pooled

PRETRAIN_CUTOFF = {"roberta-large": "2019-07", "Llama-2-13b-hf": "2022-09"}   # knowledge cutoffs: RoBERTa as in CKX (2026), Llama-2 by Touvron et al. (2023)
POST_YEAR = {m: int(c[:4]) + 1 for m, c in PRETRAIN_CUTOFF.items()}           # Tables C1-C7: the calendar years after the cutoff
SPLIT_YEAR = POST_YEAR["Llama-2-13b-hf"]                                    # the B tables' second segment starts after Llama-2's cutoff
SEGMENTS = {**{m: {"post": (y, TEST_YEARS[-1])} for m, y in POST_YEAR.items()},
            **{m: {"full": (TEST_YEARS[0], TEST_YEARS[-1]), "post": (SPLIT_YEAR, TEST_YEARS[-1])} for m in ROBUST_MODELS}}   # signal years of the B and C tables
SEGMENTS["roberta-large"]["split"] = (SPLIT_YEAR, TEST_YEARS[-1])          # Table 10, Panel B: RoBERTa on the B tables' second segment
SEGMENT_PERIOD = {m: {s: f"{lo}-{hi % 100:02d}" for s, (lo, hi) in v.items()} for m, v in SEGMENTS.items()}
PAPER_SPEC = {"test_years": [TEST_YEARS[0], TEST_YEARS[-1]], "category_mean_shrinkage": CATEGORY_MEAN_SHRINKAGE,
              "history_days": HISTORY_DAYS, "projection_years": PROJECTION_YEARS, "firm_effect_min": [MIN_STORIES, MIN_FIRM_YEARS], "taus": TAUS, "taus_fig": [TAUS_FIG[0], TAUS_FIG[-1]], "s_lag": LEVEL_LAG, "nw_lags": [NW_LAGS_DAILY, NW_LAGS_MONTHLY], "bands": BANDS,
              "subspace": SUBSPACE, "min_names": MIN_NAMES, "min_month_stocks": MIN_MONTH_STOCKS, "min_group_stories": MIN_GROUP_STORIES,
              "level_window": LEVEL_WINDOW, "horizon": H,
              "drift_min_stories": [DRIFT_MIN_STORIES, DRIFT_TABLE_MIN_STORIES], "surprise": SURPRISE, "pretrain_cutoff": PRETRAIN_CUTOFF,
              "surprise_dummy_stories": SURPRISE_DUMMY_STORIES, "surprise_rank_prior": SURPRISE_RANK_PRIOR, "group_min_incidence": GROUP_MIN_INCIDENCE}
runinfo.SPECS["paper"] = PAPER_SPEC
PAPER = ("ridge", "monthly", "paper")                                        # the specification groups of every paper product
