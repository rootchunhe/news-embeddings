"""Out-of-sample event-category mean returns (RavenPack categories) against the embedding forecasts and ESS: performance by sample,
coarser RavenPack levels, windows, by year, Fama-MacBeth (standardized and in return units with t of b = 1), spanning (alone and with
the factors), overlap, transaction costs.
Run from Codes/3_paper: ../../env/bin/python 1_category.py
Outputs Results/paper/1_category/{benchmark,levels,windows,by_year,fm,fm_<segment>,fm_raw,spanning,signals,costs,legs}.csv (fm: every
encoder; fm_<segment>: each encoder on its segments of lib.constants.SEGMENTS)"""

import numpy as np
import pandas as pd

from lib.constants import FACTORS, FORECAST, FORECAST_LABEL, HISTORY_YEARS, NW_LAGS_DAILY, PERIODS, PRED, ROBUST_PRED, SEGMENTS, TEST_YEARS
from lib.run import start, save, finish
from lib.panel import SIZE_GROUPS, firm_day, forecasts, sample_rows, segment, size_group
from lib.evaluate import perf, increments, fama_macbeth, ic_series, ls_series, ls_weights
from lib.category import COLS, LABELS as CAT_LABELS, WINDOWS, bucket, category_means, headline_direction, lm_tone

from common.factors import ff_daily, nw_ols
from common.portfolio import ANN, close_valued, ewct_table, market_data, sharpe

SIGS = ["m8", "m1", *PRED, "ess"]                                       # Table 1: the main encoders' forecasts
REF = "m8"                                                              # every difference column is against the 8-year category mean
LEVELS = {"g8": lambda d: d.group, "h8": lambda d: headline_direction(d.headline), "b8": lambda d: bucket(d.group, d.headline),
          "t8": lambda d: d.group + "/" + d["type"].astype(str)}
NO_GROUP = {"h8"}                                                       # direction alone: shrunk towards the overall mean, not a group's
LABELS = {**CAT_LABELS, "g8": "Event-group mean", "h8": "Direction mean", "lm": "LMMD sentiment", "b8": "Group x direction mean", "t8": "RavenPack-type mean",
          **{FORECAST[m]: FORECAST_LABEL[m] for m in FORECAST}, "ess": "RavenPack ESS"}
PRICE_DERIVED = ["stock-prices", "technical-analysis", "order-imbalances", "indexes", "stock-picks"]   # labels that describe a price move
HEADLINE_ONLY = ["NEWS-FLASH", "HOT-NEWS-FLASH"]
FM_SPECS = {e: [c + (e,) for c in [(), ("m8",), ("b8",)]] for e in FORECAST.values()}   # each forecast alone, with the category mean,
                                                                                         # with the group x direction mean
RAW_SPECS = [("m8",), *[(e,) for e in PRED], ("b8",)]                 # Mincer-Zarnowitz: each return forecast alone (ESS is a score)
COST_SIGS = SIGS + ["g8", "h8", "t8", "b8", "lm"]                     # Table 1's signals
SPAN_PAIRS = [(e, "m8") for e in PRED] + [("m8", e) for e in PRED]


def level_means(S, history):
    """8-year means keyed by the coarser levels (LEVELS: event group, direction of the headline, group x direction, group /
    RavenPack type)"""
    for name, key in LEVELS.items():
        A, B = S[COLS].copy(), history[COLS].copy()
        A["category"], B["category"] = key(S).values, key(history).values
        if name in NO_GROUP:
            A["group"] = B["group"] = "all"
        category_means(A, name, WINDOWS["m8"], TEST_YEARS, history=B)
        S[name] = A[name].values
    return S


def benchmark(F, sigs, periods):
    """perf rows plus the differences against REF (DM t on the IC, JK z on the EW and VW Sharpe)"""
    t = perf(F, sigs, LABELS, periods)
    extra = []
    for name, (lo, hi) in periods.items():
        d = segment(F, lo, hi)
        inc = increments(d, [(s, REF) for s in sigs if s != REF], LABELS).set_index("A")
        for s in sigs:
            if s != REF:
                r = inc.loc[LABELS[s]]
                extra.append({"period": name, "signal": LABELS[s], "d_IC_%": r["d_IC_%"], "DM_t": r.DM_t, "d_Sharpe": r.d_Sharpe, "JK_z": r.JK_z,
                              "d_Sharpe_VW": r.d_Sharpe_VW, "JK_z_VW": r.JK_z_VW})
    return t.merge(pd.DataFrame(extra), on=["period", "signal"], how="left")


def by_year(F, sigs):
    rows = []
    for s in sigs:
        ic, ew, vw = ic_series(F, s), ls_series(F, s), ls_series(F, s, vw=True)
        for T in TEST_YEARS:
            i, e, v = ic[ic.index.year == T], ew[ew.index.year == T], vw[vw.index.year == T]
            rows.append({"year": T, "signal": LABELS[s], "IC_%": 100 * i.mean(), "EW_Sharpe": sharpe(e), "EW_LS_bp": 1e4 * e.mean(), "VW_Sharpe": sharpe(v), "VW_LS_bp": 1e4 * v.mean()})
    return pd.DataFrame(rows)


def fm_table(F, specs, standardize=True):
    """Fama-MacBeth of y on each spec's signals: within-day standardized (coefficient in bp per sd) or in return units (the forecast's
    pass-through, b = 1 calibrated, with the t of b = 1)"""
    rows = []
    for spec in specs:
        t, _ = fama_macbeth(F, "y", list(spec), scale=1e4 if standardize else 1.0, standardize=standardize)
        t["x"] = t.x.map(LABELS)
        t.insert(0, "spec", " + ".join(LABELS[x] for x in spec))
        if not standardize:
            t["t_vs_1"] = (t.coef - 1) / (t.coef / t.t)
        rows.append(t.rename(columns={"x": "regressor", "t": "t_NW", "periods": "days"}))
    return pd.concat(rows, ignore_index=True)


def spanning(F, pairs):
    """decile long-short of one signal regressed on the other's (same weighting): open-to-open returns alone, and close-valued
    returns together with the factors of the same day"""
    rows = []
    sigs = {x for p in pairs for x in p}
    ff = ff_daily()
    fac, cal = ff[FACTORS].dropna(), ff.index                          # close_valued runs on the trading calendar
    for weight in ["EW", "VW"]:
        oo = {s: ls_series(F, s, vw=weight == "VW") for s in sigs}
        cv = {s: close_valued(ls_weights(F, s, vw=weight == "VW"), cal) for s in sigs}
        for a, b in pairs:
            for model, ra, X in [("none", *oo[a].align(oo[b], join="inner")), ("factors", *cv[a].align(cv[b], join="inner"))]:
                X = pd.concat([X.rename("b"), fac], axis=1, join="inner") if model == "factors" else X.to_frame("b")
                ra = ra.reindex(X.index)
                ok = ra.notna().values & X.notna().all(1).values
                coef, t, r2 = nw_ols(ra.values[ok], X.values[ok], NW_LAGS_DAILY)
                resid = ra.values[ok] - np.column_stack([np.ones(ok.sum()), X.values[ok]]) @ coef
                rows.append({"weight": weight, "factors": model, "dependent": LABELS[a], "on": LABELS[b], "beta": coef[1], "alpha_bp_day": 1e4 * coef[0],
                             "t_alpha": t[0], "R2": r2, "appraisal_ratio": coef[0] / resid.std() * np.sqrt(ANN), "days": int(ok.sum())})
    return pd.DataFrame(rows)


def signal_stats(S, F):
    """per signal: the share of its story-level variance between event categories and its firm-day correlations with the 8-year
    category and group x direction means"""
    rows = []
    for s in ["m8", *PRED, "ess", "b8"]:
        within = S[s] - S.groupby("category")[s].transform("mean")
        rows.append({"signal": LABELS[s], "between_category_share_%": np.nan if s == "m8" else 100 * (1 - within.var() / S[s].var()),
                     "corr_with_m8": np.nan if s == "m8" else F[[s, "m8"]].corr().iloc[0, 1],
                     "corr_with_b8": np.nan if s == "b8" else F[[s, "b8"]].corr().iloc[0, 1]})
    return pd.DataFrame(rows)


def costs(F, log):
    """CKX's transaction-cost test (their Section 3.5) on the paper's EW and VW decile long-short (CKX: EW quintiles) of each
    signal under exponentially weighted calendar-time rebalancing, 10 / 20 bp per 100% turnover for stocks above / below the
    NYSE 20th percentile"""
    mk = market_data(log)
    rows = []
    for s in COST_SIGS:
        for weight in ["EW", "VW"]:
            W = ls_weights(F, s, q=10, vw=weight == "VW")
            t = ewct_table(W.assign(w_ls=W.w_long - W.w_short), mk).reset_index()
            rows.append(t.assign(signal=LABELS[s], weight=weight))
            log(f"costs {s} {weight}: best net SR {t.net_SR.max():.2f} at gamma {t.gamma[t.net_SR.idxmax()]:.1f}")
    return pd.concat(rows, ignore_index=True)


def leg_sizes(F):
    """names on an average day of each signal's decile long-short: all, in the long and in the short leg"""
    rows = []
    for s in COST_SIGS:
        g = ls_weights(F, s).groupby("day0")
        rows.append({"signal": LABELS[s], "names": g.size().mean(), "long": g.w_long.apply(lambda x: (x > 0).sum()).mean(),
                     "short": g.w_short.apply(lambda x: (x > 0).sum()).mean()})
    return pd.DataFrame(rows)


def main():
    a, log, out = start("1_category", model=False)
    S0 = sample_rows(TEST_YEARS, columns=["news_type", "type", "intra0", "gap1", "headline"])
    S = forecasts(S0, PRED | ROBUST_PRED)                                      # every encoder's forecast; Table 1 uses the main ones
    assert len(S) == len(S0), "a story matched more than one forecast"
    history = sample_rows(HISTORY_YEARS, columns=["type", "headline"])
    for name, window in WINDOWS.items():
        category_means(S, name, window, TEST_YEARS, history=history[COLS])
    level_means(S, history)
    S["lm"] = lm_tone(S.headline).values
    assert S[list(PRED)].notna().all().all(), "stories without a main encoder's forecast"
    S = S.dropna(subset=SIGS + list(LEVELS)).reset_index(drop=True)      # every signal on the same stories and firm-days
    assert S[list(ROBUST_PRED)].notna().all().all(), "the robustness encoder's forecasts must cover the common sample"
    missing = sorted(set(TEST_YEARS) - set(S.yr.unique()))
    assert not missing, f"the common sample has no stories in {missing}: a signal is undefined there (history years, forecasts?)"
    log(f"common sample: {len(S):,} stories, {S.yr.min()}-{S.yr.max()}; ESS missing (imputed 0) for {100 * S.ess_missing.mean():.1f}%; news types: "
        + ", ".join(f"{k} {100 * v:.1f}%" for k, v in S.news_type.value_counts(normalize=True).head(6).items()))

    size = size_group(S.day0, S.cap)
    assert ((size == "micro") == S.small.values).all(), "micro must be the stories below the NYSE 20th percentile (`small`)"
    subsets = {"all": np.ones(len(S), bool), **{f"size: {g}": size == g for g in SIZE_GROUPS},
               "ex insider-trading": (S.group != "insider-trading").values, "ex price-derived groups": ~S.group.isin(PRICE_DERIVED).values,
               "headline-only news types": S.news_type.isin(HEADLINE_ONLY).values, "FULL-ARTICLE news type": (S.news_type == "FULL-ARTICLE").values}
    bench = []
    for name, mask in subsets.items():
        F = firm_day(S[mask], SIGS)
        t = benchmark(F, SIGS, PERIODS)
        bench.append(t.assign(sample=name, stories_pct=100 * mask.mean()))
        log(f"benchmark {name}: {int(mask.sum()):,} stories, {len(F):,} firm-days")
    save(out, "benchmark", pd.concat(bench, ignore_index=True), "benchmark", digits=2)
    lev = ["g8", "h8", "lm", "b8", "t8", "m8", *PRED]
    levels = perf(firm_day(S, lev), lev, LABELS, PERIODS)
    keys = {k: int(key(S).nunique()) for k, key in LEVELS.items()} | {"m8": int(S.category.nunique())}
    levels["cells"] = levels.signal.map({LABELS[k]: n for k, n in keys.items()})
    save(out, "levels", levels, "8-year means at coarser levels (cells: distinct keys)", digits=2)
    log("keys per level: " + ", ".join(f"{k} {n}" for k, n in keys.items()))

    F = firm_day(S, list(dict.fromkeys(SIGS + list(WINDOWS) + COST_SIGS + list(ROBUST_PRED))), extra={"intra0": ("intra0", "first"), "gap1": ("gap1", "first")})   # close_valued
    Fw = F.dropna(subset=list(WINDOWS))
    save(out, "windows", perf(Fw, list(WINDOWS), CAT_LABELS, PERIODS), "category-mean windows, every window on the same firm-days", digits=2)
    save(out, "by_year", by_year(F, COST_SIGS))
    encoders = {m: e for m, e in FORECAST.items() if e in PRED or e in ROBUST_PRED}
    fm = pd.concat([fm_table(F, FM_SPECS[e]) for e in encoders.values()], ignore_index=True)
    save(out, "fm", fm, "daily Fama-MacBeth, y in bp on within-day standardized signals", digits=2)
    for s in dict.fromkeys(k for m in encoders for k in SEGMENTS[m]):   # every encoder on its segments (Tables C1, B1)
        seg = [fm_table(segment(F, *SEGMENTS[m][s]), FM_SPECS[e]) for m, e in encoders.items() if s in SEGMENTS[m]]
        save(out, f"fm_{s}", pd.concat(seg, ignore_index=True), f"the same on each encoder's {s} segment (lib.constants.SEGMENTS)", digits=2)
    save(out, "fm_raw", fm_table(F, RAW_SPECS, standardize=False), "daily Fama-MacBeth, y on the forecasts in return units", digits=3)
    save(out, "spanning", spanning(F, SPAN_PAIRS), "spanning of the decile long-short returns, alone and with the factors", digits=2)
    save(out, "signals", signal_stats(S, F), "dispersion and overlap of the signals")
    save(out, "costs", costs(F, log), "EWCT transaction costs of the EW and VW decile long-short (turnover %, bp per day, SR)", digits=3)
    save(out, "legs", leg_sizes(F), "names per day in the long-short legs", digits=1)
    finish(out, a, log, stories=len(S), keys=keys, encoders=list(encoders), ess_missing_share=float(S.ess_missing.mean()))


if __name__ == "__main__":
    main()
