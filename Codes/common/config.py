"""Named thresholds shared by the scripts."""
MIN_NAMES = 20              # stocks with a signal needed on a day to form portfolios or a daily correlation
MIN_NAMES_LARGE = 10        # same, for the subset above the NYSE median
MIN_NAMES_DOUBLE_SORT = 25  # for the 5x5 sort on past return and forecast
CATEGORY_MEAN_SHRINKAGE = 200   # category-mean return forecasts: weight n / (n + 200) on a category's mean return, the rest on its group's (or the pooled) mean
FIRST_YEAR = 2000           # first clean year
TRAIN_YEARS, VAL_YEARS = 6, 2   # the CKX window: train T-8..T-3, validate T-2..T-1
LEVEL_WINDOW = 12           # historical level: the firm's news months among the preceding 12 calendar months
MIN_GROUP_STORIES = 50000   # stories an event group needs to be reported on its own
MIN_MONTH_STOCKS = 50       # stocks a month needs for a monthly sort or a lagged managed factor
RIDGE_GRID = [10.0 ** k for k in range(-1, 9)]   # CKX: penalty from a log-uniform grid 1e-1..1e8 scaled by sample size and dimension (alpha = c n / P)
Y_CLIP = 0.5                # winsorization of the next-day return used as the ridge target
RET_CLIP = (-0.9, 5)        # winsorization of the JKP monthly excess return as the monthly ridge's training target
CHUNK = 65536               # rows per block when embeddings are streamed
NW_LAGS_DAILY, NW_LAGS_MONTHLY = 5, 6   # Newey-West lags of daily and monthly series
MSRR_GRID = [10.0 ** (k / 2) for k in range(-4, 9)]   # MSRR ridge penalties, times trace(FF') / D (log grid 1e-2..1e4)
MSRR_INIT = 12              # MSRR: months of managed factors needed before the first out-of-sample month


def lambdas(d):
    """the per-observation ridge penalties for a d-dimensional embedding: lambda = c / d for every c of RIDGE_GRID"""
    return [c / d for c in RIDGE_GRID]
