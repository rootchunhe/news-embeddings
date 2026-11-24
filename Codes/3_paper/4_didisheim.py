"""Section 5: the Didisheim news shock (2_replication/didisheim: z-scored firm-month embedding residualized month by month on the JKP
characteristics, MSRR) seen through the content x timing decomposition. Run from Codes/3_paper/ after the Didisheim replication and
4_monthly.py (Llama: on a compute node):
    ../../env/bin/python 4_didisheim.py --model roberta-large
        main: on the signal months of the test years, the replication's own strategy (its saved undelayed MSRR weights, the fixed
        original rule b0_{t-tau} of their Figure 10A, at every delay of TAUS_FIG) decomposed exactly into the cells cL, cD, F, uL, uD of the shock (the parts of
        4_monthly.py and U = E - m_T - C - F, m_T the mean embedding of the projection's headlines of test year T, in z units of the replication's z-score; levels over the firm's news months among the previous
        LEVEL_WINDOW months, 0 without such a month; each cell residualized with the replication's monthly projection),
        checked against the replication's returns; the same with weights re-optimized per delay (appendix; every delay of TAUS_FIG, as
        everything below). Appendix: each part
        traded on its own (the shock, its level and deviation, every cell, the level LEVEL_LAG months earlier, the firm identity
        LEVEL_LAG months earlier (F_lag of 4_monthly: the firm effect of the test year LEVEL_LAG / 12 years before, whose stories end
        37 to 48 months before the signal month; like the lagged level, scaled by the signal month's z-score and residualized on
        its characteristics), the
        category-template counterfactual, the shock with the month's own event-group indicators next to the characteristics; an
        input that exists only on some firm-months, a level or firm identity, is residualized and held only there, and a month
        needs MIN_MONTH_STOCKS such firm-months with a return: AVAIL);
        joint Fama-MacBeth of the old deviation and the current level; the level and the deviation (each along the undelayed
        weights of its month) as signals by delay, their rank IC and joint Fama-MacBeth (signals.csv) and their monthly rank ICs
        (ic_monthly.csv: signal month t, delay tau, the return of month t + tau + 1); the decomposition and each
        input's Sharpe ratio also on the signal months of each segment (SEGMENTS: attribution_<segment>.csv at every delay of TAUS_FIG, Sharpe_<segment>), and
        the strategy's monthly returns by part (monthly.csv); the fixed-rule decomposition split exactly by the Fama-French size group of
        each firm in the signal month (attribution_size.csv: the same weights, the sum over each group's firms). Output Results/paper/4_didisheim/attribution/<model>/
    ../../env/bin/python 4_didisheim.py --model roberta-large --timing
        appendix: the replication on firm-months keyed by the trade day's month (01_firm_month_embeddings.py --month trade --suffix
        _trade) instead of the month of arrival. Output 4_didisheim/timing/<model>/
    ../../env/bin/python 4_didisheim.py --model roberta-large --industry
        the main run with GICS industry-group indicators (newsshock.load_panel: the first four GICS digits) next to the characteristics
        in every residualization, as in Didisheim et al.'s Figure 21: the standalone strategies of Figure 4's lower panel (each fits its
        own weights) and the decomposition with the undelayed MSRR weights refitted on that shock as in 02_msrr.py (lag_profile); no
        template or event-group curve. Output 4_didisheim/industry/<model>/"""
from bisect import bisect_right
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lib import DIDISHEIM_RESULTS as REP, ROOT
from lib.constants import CELLS, LABELS, LEVEL_WINDOW, MIN_MONTH_STOCKS, NW_LAGS_MONTHLY, PAPER, LEVEL_LAG, SEGMENTS, STANDALONE, TAUS, TAUS_FIG, TEST_YEARS, TIMING
from lib.evaluate import fama_macbeth, ic_series
from lib.monthly import SIZE_OF_JKP, trailing_level
from lib.panel import SIZE_GROUPS
from lib.run import finish, save, start

from common.jkp import monthly_sharpe
from common.portfolio import nw_tstat
from common.runinfo import require_same_sample
from newsshock import (factors, lag_profile, load_embeddings, load_panel, managed_factors, msrr, recursive_zscore,
                       residualize)

INDUSTRY = False                                                # --industry: industry-group indicators in every residualization (main)
START, END = "2000-01", f"{TEST_YEARS[-1]}-12"                 # the window of the replication on the extended sample
FIRST = f"{TEST_YEARS[0]}-01"                                   # the decomposition exists from the first test year
JOINT_LAGS = TAUS_FIG[1:]                                       # every lag k = 1..35: the return of month t + 1 is the (k+1)-th month after the deviation month


# the inputs a standalone strategy holds only where they exist (shock_where): the level cells and the level with a history or, for
# firm identity, a firm effect in the test year's projection
AVAIL = {"cL": lambda x: x.hasL, "uL": lambda x: x.hasL, "yF": lambda x: x.seen, "lev": lambda x: x.seen | x.hasL}


def section(a):
    assert not (a.timing and a.industry), "--timing and --industry are separate runs"   # before start() creates a staging folder
    return "4_didisheim/timing" if a.timing else "4_didisheim/industry" if a.industry else "4_didisheim/attribution"


def news_shock(Z, P, extra=None):
    """newsshock.news_shock (the z-scored embedding residualized month by month on the JKP characteristics [+ extra columns]), with
    the industry-group indicators as well under --industry"""
    return residualize(Z, P.M, P.chars, True, INDUSTRY, extra, fit=False)[0]


def replication(model):
    """the replication's manifest, checked against this script's window, and its characteristic list"""
    m = require_same_sample(REP / model, f"Didisheim replication of {model}", needs=("monthly",))
    assert (m["args"]["start"], m["args"]["end"]) == (START, END), f"the replication ran on {m['args']['start']}..{m['args']['end']}, not {START}..{END}"
    return m["characteristics"]


def delay_curve(name, X, P, log, segments, avail=None):
    """the strategy on X at every delay; also its Sharpe ratio on the signal months of each segment, {name: (first, last) position
    among P.months}. avail (one boolean per row of X and of P.M, None: every row): the firm-months where the input exists; a
    month's factors need MIN_MONTH_STOCKS of them with a return (newsshock.factors)"""
    lags, opt = lag_profile(X, P, TAUS_FIG, avail)[:2]
    for k, (i, j) in segments.items():
        seg = [opt[tau].iloc[i + tau:j + tau + 1].dropna() for tau in lags.tau]
        lags = lags.assign(**{f"Sharpe_{k}": [monthly_sharpe(s) for s in seg], f"t_{k}": [nw_tstat(s.values, NW_LAGS_MONTHLY) for s in seg],
                              f"months_{k}": [len(s) for s in seg]})
    log(f"{name}: " + " ".join(f"tau {int(r.tau)} {r.Sharpe:.2f}/{r.Sharpe_fixed_b:.2f}" for r in lags.itertuples()))
    return lags.assign(object=name)


# ---- appendix run on the replication's sample ------------------------------------------------------------------------------------
def timing_run(a, log, out):
    require_same_sample(ROOT / f"Data/clean/run_firm_month_{a.model}_trade.json", "firm-month embeddings by trade month")
    P = load_panel(a.model, "_trade", START, END, chars=replication(a.model))
    save(out, "F10_lags", delay_curve("trade month", news_shock(recursive_zscore(P.E, P.M.t.values, P.n_months), P), P, log, {}))
    return {"firm_months": len(P.M)}


def group_indicators(P):
    """the month's own event-group indicators of every firm-month of P (4_groups.py)"""
    require_same_sample(ROOT / "Data/clean/run_firm_month_groups.json", "event-group indicators (4_groups.py)", needs=PAPER)
    g = np.load(ROOT / "Data/clean/firm_month_groups.npz")
    G = pd.DataFrame(g["D"], columns=g["groups"].astype(str)).assign(permno=g["permno"].astype(int), month=g["month"].astype(str))
    G = P.M[["permno", "month"]].merge(G, on=["permno", "month"], how="left", validate="1:1").drop(columns=["permno", "month"])
    assert G.notna().all().all(), "firm-months of the panel without their event-group indicators"
    return G.values.astype(np.float64)


# ---- the decomposition of the original strategy -----------------------------------------------------------------------------------
def rows_of(P, keys, mask):
    """positions in `keys` (DataFrame permno, month) of the rows of P.M[mask]"""
    pos = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_arrays([keys.permno.astype(np.int64).values, keys.month.astype(str).values]))
    p = pos.reindex(pd.MultiIndex.from_arrays([P.M.permno.values[mask].astype(np.int64), P.M.month.values[mask]])).values
    assert not np.isnan(p).any(), "firm-months of the panel missing from the parts"
    return p.astype(int)


def inputs(model, P, r04):
    """firm-month pieces for the panel rows of the test years: E, the centered C and F of 4_monthly with the projection mean of each
    test year (window_mean), the levels LE and LC (equal news months), hasL, the level LEVEL_LAG months earlier, Llag, with its flag, and the firm
    identity LEVEL_LAG months earlier, F_lag (4_monthly), with its flag seen_lag"""
    require_same_sample(ROOT / f"Data/clean/run_firm_month_parts_{model}.json", "firm-month parts (4_monthly.py)", needs=PAPER)
    z = np.load(ROOT / f"Data/clean/firm_month_parts_{model}.npz")
    keys = pd.DataFrame({"permno": z["permno"], "month": z["month"].astype(str)})
    i = rows_of(P, keys, r04)
    C, LC, F, hasL, seen = (z["C"][i].astype(np.float32), z["LC"][i].astype(np.float32), z["F"][i].astype(np.float32), z["hasL"][i],
                            z["seen"][i])
    F_lag, seen_lag = z["F_lag"][i].astype(np.float32), z["seen_lag"][i]
    window_mean = dict(zip(z["window_year"].tolist(), z["window_mean"].astype(np.float32)))
    del z
    first = (pd.Period(FIRST, "M") - LEVEL_LAG - LEVEL_WINDOW).strftime("%Y-%m")   # history the lagged level needs
    k, E_all = load_embeddings(model, "", first, END)
    LE, hasE = trailing_level(E_all, k)
    Llag, hasLlag = trailing_level(E_all, k, lag=LEVEL_LAG + 1)          # the level of month t - LEVEL_LAG (a level's window ends a month before)
    j = rows_of(P, k, r04)
    assert (hasE[j] == hasL).all(), "the levels of E and of its parts use different news months"
    year = P.M.month.astype(str).str[:4].astype(int).values[r04]
    return SimpleNamespace(E=P.E[r04], C=C, F=F, LC=LC, LE=LE[j], hasL=hasL, seen=seen, Llag=Llag[j], hasLlag=hasLlag[j],   # rows of the test years
                           F_lag=F_lag, seen_lag=seen_lag, year=year, window_mean=window_mean)


def cell(name, x, idx, mu, sd):
    """one cell in z units for the rows idx of one month (mean mu, sd of the replication's z-score; m the projection mean of the month's
    test year); the cells add up to (E - mu) / sd, with the offset (m - mu) / sd, common to the month's rows, in cD"""
    lev = x.hasL[idx, None]
    m = x.window_mean[x.year[idx[0]]]
    cL = np.where(lev, x.LC[idx] / sd, 0)
    if name == "cL":
        return cL
    if name == "cD":
        return (x.C[idx] + m - mu) / sd - cL
    if name == "yF":
        return x.F[idx] / sd
    uL = np.where(lev, (x.LE[idx] - m - x.LC[idx] - x.F[idx]) / sd, 0)
    return uL if name == "uL" else (x.E[idx] - m - x.C[idx] - x.F[idx]) / sd - uL


def shock_where(X, P, r04, avail):
    """the news shock of X (one row per row of P.M) for a standalone strategy on an input that exists only on some firm-months:
    residualized month by month among the test-year rows where it exists (avail: one boolean per test-year row, in the order of
    P.M[r04]), zero elsewhere, so the strategy holds only firms with the input"""
    rows = np.zeros(len(P.M), bool)
    rows[np.where(r04)[0][avail]] = True
    out = np.zeros_like(X)
    out[rows] = news_shock(X[rows], SimpleNamespace(M=P.M[rows].reset_index(drop=True), chars=P.chars))
    return out


def by_month(P, r04, MU, SD, fn):
    """full-row matrix (zeros before the test years) filled month by month with fn(rows of the test-year block, mu, sd)"""
    t04 = P.M.t.values[r04]
    out = np.zeros((len(P.M), MU.shape[1]), np.float32)
    pos = np.where(r04)[0]
    for t, idx in pd.Series(np.arange(len(t04))).groupby(t04).indices.items():
        out[pos[idx]] = fn(idx, MU[t].astype(np.float32), SD[t].astype(np.float32))
    return out


def delayed_returns(P):
    """{tau: the return the stock of each row of P.M (signal month t) earns in month t + tau + 1} for the delays of TAUS_FIG: the
    panel's return at tau = 0, otherwise the JKP return of signal month t + tau (no news needed there); NaN without one or beyond
    the panel's months; each delay uses the signal months and stocks with that delay's return"""
    t, permno, months = P.M.t.values, P.M.permno.values, np.array(P.months)
    out = {0: P.ret.astype(np.float64)}
    for tau in TAUS_FIG[1:]:
        r, inside = np.full(len(t), np.nan), t + tau < P.n_months
        r[inside] = P.Jret.reindex(pd.MultiIndex.from_arrays([permno[inside], months[t[inside] + tau]])).values
        out[tau] = r
    return out


def scores(eps, P, B0, RET):
    """each row's score along the undelayed weights of its month, eps_i,t' b0_t (0 in a month without weights), and {tau: by signal
    month t, whether the managed-factor vector of the month's rows with a return, sum_i eps_i,t r_i (RET = delayed_returns(P); the
    sum of newsshock.factors for formation month t + tau), has a nonzero coordinate}"""
    s, nonzero = np.zeros(len(P.M)), np.zeros((P.n_months, len(TAUS_FIG)), bool)
    R = np.column_stack([np.where(np.isfinite(RET[tau]), RET[tau], 0.0) for tau in TAUS_FIG])   # rows x delays, 0 without a return
    for t, idx in P.M.groupby("t").indices.items():
        e = eps[idx].astype(np.float64)
        s[idx] = e @ np.nan_to_num(B0[t])
        nonzero[t] = np.abs(e.T @ R[idx]).sum(0) > 0
    return s, dict(zip(TAUS_FIG, nonzero.T))


def fixed_payoffs(s, nonzero, P, B0, RET, size=None, groups=()):
    """the fixed-rule return F_t' b0_{t-tau} of a signal at every delay of TAUS_FIG from its rows' scores (s, nonzero = scores(eps, P,
    B0, RET); RET = delayed_returns(P)): per formation month t, the sum of score x return over the rows of signal month t - tau with a
    return (= newsshock.fixed_weight_returns(factors(eps, P, tau), B0, tau), on the same months). NaN before the delay, without
    weights in the signal month, with fewer than MIN_MONTH_STOCKS such rows, or with a zero managed-factor vector. Returns ({tau:
    returns by formation month}, {tau: {group: the same sum over the group's rows}}); size, groups: the rows' size groups. A
    group's sum is kept where the signal's factor vector is zero (groups that cancel); on_strategy_months sets the months"""
    t, n = P.M.t.values, P.n_months
    has_b = ~np.isnan(B0).any(1)
    out, split = {}, {}
    for tau in TAUS_FIG:
        r = RET[tau]
        ok = np.isfinite(r)
        c = np.where(ok, s * np.nan_to_num(r), 0.0)
        enough = has_b & (np.bincount(t, ok, minlength=n) >= MIN_MONTH_STOCKS)                            # by signal month, any signal
        shift = lambda v, live: np.concatenate([np.full(tau, np.nan), np.where(live, v, np.nan)[:n - tau]])   # signal month t - tau -> formation month t
        out[tau] = shift(np.bincount(t, c, minlength=n), enough & nonzero[tau])
        split[tau] = {g: shift(np.bincount(t, np.where(size == g, c, 0.0), minlength=n), enough) for g in groups}
    return out, split


def on_strategy_months(total, v):
    """the return v of a part (a cell, or a cell within a size group) on the months of the strategy's return `total`: NaN where the
    strategy has none, 0 where the part has none"""
    return np.where(np.isnan(total), np.nan, np.nan_to_num(v))


def reoptimized_payoffs(eps, P, Bopt):
    """the re-optimized return F_t' b_tau,t of signal eps at the delays with re-optimized weights (the keys of Bopt)"""
    out = {}
    for tau, B in Bopt.items():
        F = factors(eps, P, tau)
        live, has_b = np.abs(F).sum(1) > 0, ~np.isnan(B).any(1)
        out[tau] = np.where(live & has_b, (F * np.nan_to_num(B)).sum(1), np.nan)
    return out


def payoffs(eps, P, B0, Bopt, RET, size=None, groups=()):
    """of signal eps: {(weights, tau): returns} with the fixed rule at every delay of TAUS_FIG (fixed_payoffs) and re-optimized
    weights at the delays of Bopt; the rows' scores on the undelayed weights; the fixed rule's split by size group"""
    s, nonzero = scores(eps, P, B0, RET)
    fixed, split = fixed_payoffs(s, nonzero, P, B0, RET, size, groups)
    out = {**{("fixed", tau): v for tau, v in fixed.items()}, **{("re-optimized", tau): v for tau, v in reoptimized_payoffs(eps, P, Bopt).items()}}
    return out, s, split


def group_factors(X, P, tau, size, groups):
    """newsshock.factors(X, P, tau) split by the size group of the signal month's firm: {group: F_g}, the managed factors over each
    group's stocks alone (the stocks and months of factors); used to check the size split of fixed_payoffs"""
    by_t = P.M.groupby("t").indices
    out = {g: np.zeros((P.n_months, X.shape[1])) for g in groups}
    for t in range(tau, P.n_months):
        idx = by_t.get(t - tau)
        if idx is None:
            continue
        r = P.ret[idx] if tau == 0 else P.Jret.reindex(list(zip(P.M.permno.values[idx], [P.months[t]] * len(idx)))).values
        ok = np.isfinite(r)
        if ok.sum() < MIN_MONTH_STOCKS:
            continue
        Xi, ri, gi = X[idx[ok]].astype(np.float64), r[ok], size[idx[ok]]
        for g in groups:
            out[g][t] = Xi.T @ np.where(gi == g, ri, 0.0)
    return out


def check_split(eps, P, B0, split, size, groups, first_t):
    """the size split of fixed_payoffs (split) against the managed factors of each group, on the months of the test years: at
    tau = 0 and 1 every group's return is its factors (group_factors) times the signal month's weights, and the micro factors at
    tau = 0 are newsshock.managed_factors on the micro rows alone"""
    tol = lambda F: dict(rtol=1e-7, atol=1e-7 * np.abs(F).max())
    for tau in (0, 1):
        G = group_factors(eps, P, tau, size, groups)
        if tau == 0:
            mine = G["micro"]
        W = np.concatenate([np.zeros((tau, B0.shape[1])), np.nan_to_num(B0)[:P.n_months - tau]])      # the weights of signal month t - tau
        for g in groups:
            ref, got = (G[g] * W).sum(1)[first_t + tau:], split[tau][g][first_t + tau:]
            has = ~np.isnan(got)
            assert has.any() and np.allclose(got[has], ref[has], **tol(ref[has])), f"tau {tau}, {g}: the size split differs from the group's managed factors"
    m, t = size == "micro", P.M.t.values
    ref = managed_factors(eps[m], t[m], P.ret[m], P.n_months)
    live = (np.bincount(t[m], minlength=P.n_months) >= MIN_MONTH_STOCKS) & (np.bincount(t, minlength=P.n_months) >= MIN_MONTH_STOCKS)
    live[:first_t] = False                                                  # test-year months where both rules keep the month
    assert live.any() and np.allclose(mine[live], ref[live], **tol(ref[live])), "the micro factors differ from managed_factors on the micro rows"


def size_table(R, S, months, start, groups):
    """the fixed-rule decomposition by size group (S: cell -> the size split of fixed_payoffs) on the formation months of attribution_table, with each
    delay's total strategy scale; checks that the groups add up to every cell"""
    rows = []
    for tau in TAUS_FIG:
        parts = R[("fixed", tau)]
        tot = pd.Series(parts["total"], index=months).iloc[start + tau:].dropna()
        scale = 0.10 / (tot.std() * np.sqrt(12))
        G = {c: {g: on_strategy_months(parts["total"], S[c][tau][g]) for g in groups} for c in CELLS}
        for c in CELLS:
            gap = np.nanmax(np.abs(sum(G[c].values()) - parts[c])[start + tau:])
            assert gap < 1e-7 * np.nanmax(np.abs(parts["total"][start + tau:])), f"tau {tau} {c}: the size groups do not add up (gap {gap:.1e})"
        G.update({k: {g: sum(G[c][g] for c in v) for g in groups} for k, v in TIMING.items()})
        G["total"] = {g: sum(G[c][g] for c in CELLS) for g in groups}
        for k, by_size in G.items():
            for g, v in by_size.items():
                r = pd.Series(v, index=months).reindex(tot.index)
                rows.append({"weights": "fixed", "tau": tau, "part": k, "size": g, "months": len(tot), "mean_%/yr": 1200 * r.mean(),
                             "mean_%/yr_at_10%vol": 1200 * scale * r.mean(), "t": nw_tstat(r.values, NW_LAGS_MONTHLY), "share": r.mean() / tot.mean()})
    return pd.DataFrame(rows)


def attribution_table(R, months, start, end):
    """per weighting and delay of R: mean (annualized; raw, and at 10% volatility of that delay's strategy), Newey-West t and share of every part; months
    of the formation months whose signal month is at position start..end"""
    rows = []
    for (w, tau), parts in R.items():
        tot = pd.Series(parts["total"], index=months).iloc[start + tau:end + tau + 1].dropna()
        scale = 0.10 / (tot.std() * np.sqrt(12))
        for k, v in parts.items():
            r = pd.Series(v, index=months).reindex(tot.index)
            rows.append({"weights": w, "tau": tau, "part": k, "months": len(tot), "mean_%/yr": 1200 * r.mean(), "mean_%/yr_at_10%vol": 1200 * scale * r.mean(),
                         "t": nw_tstat(r.values, NW_LAGS_MONTHLY), "share": r.mean() / tot.mean(),
                         "Sharpe": monthly_sharpe(r)})
    return pd.DataFrame(rows)


def attribution_run(a, log, out):
    P = load_panel(a.model, "", START, END, chars=replication(a.model))
    w = np.load(REP / a.model / "weights.npz")
    assert list(w["months"].astype(str)) == list(P.months), "the replication's weights are on other months"
    B0 = w["msrr"].astype(np.float64)
    r04 = (P.M.month >= FIRST).values
    first_t = P.months.index(FIRST)
    unknown = set(P.M.size_grp.dropna().unique()) - set(SIZE_OF_JKP)
    assert not unknown, f"JKP size groups without a mapping: {unknown}"
    size = P.M.size_grp.map(SIZE_OF_JKP).fillna("none").to_numpy(dtype=object)            # the signal month's size group of every row (attribution_size)
    groups = SIZE_GROUPS + (("none",) if (size[r04] == "none").any() else ())   # "none": no JKP size group; kept so the split stays exact
    log("size groups of the test-year firm-months: " + ", ".join(f"{g} {100 * (size[r04] == g).mean():.1f}%" for g in groups))
    segments = {k: (P.months.index(f"{lo}-01"), bisect_right(P.months, f"{hi}-12") - 1) for k, (lo, hi) in SEGMENTS[a.model].items()}   # the last panel month is the last with a next-month return (2025-11)
    x = inputs(a.model, P, r04)
    Z, MU, SD = recursive_zscore(P.E, P.M.t.values, P.n_months, moments=True)
    P.E = None
    P4 = SimpleNamespace(**{**vars(P), "M": P.M[r04].reset_index(drop=True), "ret": P.ret[r04]})
    curve = lambda name, X, avail=None: delay_curve(name, X, P4, log, segments, avail)   # X and avail on the test-year rows (P4.M)
    if not INDUSTRY:                                                        # the industry run skips the event-group and template curves
        grouped = curve(STANDALONE["groups"], news_shock(Z, P, group_indicators(P))[r04])   # the month's event groups next to the characteristics
    shock = news_shock(Z, P)
    del Z
    if INDUSTRY:                                                            # the undelayed weights of the industry shock, as 02_msrr.py fits the replication's
        B0 = msrr(factors(shock, P, 0), weights=True)[1]
    Bopt = {tau: (B0 if tau == 0 else msrr(factors(shock, P, tau), weights=True)[1]) for tau in TAUS_FIG}
    RET = delayed_returns(P)
    total = payoffs(shock, P, B0, Bopt, RET)[0]
    stored = None if INDUSTRY else pd.read_csv(REP / a.model / "lag_returns_fixed.csv", index_col=0)
    for tau in TAUS_FIG if not INDUSTRY else ():                            # the rebuilt strategy is the replication's (not under --industry)
        mine, theirs = total[("fixed", tau)][first_t + tau:], stored[str(tau)].values[first_t + tau:]
        assert np.array_equal(np.isnan(mine), np.isnan(theirs)), f"tau {tau}: the rebuilt fixed-rule returns are on other months than the replication's"
        assert np.nanmax(np.abs(mine - theirs)) < 1e-6 * np.nanmax(np.abs(theirs)), f"tau {tau}: the rebuilt fixed-rule returns differ from the replication's"
    proj = {}
    shock = shock[r04]

    # appendix inputs that need the full rows once: the lagged level, the lagged firm identity and the category-template counterfactual
    lagged = by_month(P, r04, MU, SD, lambda idx, mu, sd: np.where(x.hasLlag[idx, None], (x.Llag[idx] - mu) / sd, 0))
    lagged = shock_where(lagged, P, r04, x.hasLlag)[r04]
    curves = [curve(STANDALONE["embedding"], shock), curve(STANDALONE["lagged"], lagged, x.hasLlag)]
    del lagged, shock
    x.Llag = None
    lagged = by_month(P, r04, MU, SD, lambda idx, mu, sd: np.where(x.seen_lag[idx, None], x.F_lag[idx] / sd, 0))
    lagged = shock_where(lagged, P, r04, x.seen_lag)[r04]
    curves.append(curve(STANDALONE["lagged_F"], lagged, x.seen_lag))
    del lagged
    x.F_lag = None
    if not INDUSTRY:
        curves.append(grouped)
        require_same_sample(ROOT / f"Data/clean/run_firm_month_emb_{a.model}_T.json", "category-template firm-months (4_monthly.py --template)", needs=PAPER)
        PT = load_panel(a.model, "_T", START, END, chars=P.chars)
        assert PT.M[["permno", "month"]].equals(P.M[["permno", "month"]]), "the template counterfactual has other firm-months"
        eT = residualize(recursive_zscore(PT.E, PT.M.t.values, PT.n_months), PT.M, P.chars, True, False, fit=False)[0][r04]
        del PT
        curves.append(curve(STANDALONE["template"], eT))
        del eT

    R = {key: {"total": v} for key, v in total.items()}
    RS = {}
    for name in CELLS:
        X = by_month(P, r04, MU, SD, lambda idx, mu, sd: cell(name, x, idx, mu, sd))
        e = news_shock(X, P)
        pay, score, RS[name] = payoffs(e, P, B0, Bopt, RET, size, groups)
        for key, v in pay.items():
            R[key][name] = on_strategy_months(total[key], v)                 # a part without factors that month earns 0
        proj[name] = score[r04]
        if name == CELLS[0]:
            check_split(e, P, B0, RS[name], size, groups, first_t)
        if name in AVAIL:                                                  # as its own strategy: on the rows where the cell exists
            del e
            e = shock_where(X, P, r04, AVAIL[name](x))
        curves.append(curve(LABELS[name], e[r04], AVAIL[name](x) if name in AVAIL else None))
        del X, e
        log(f"cell {name}: fixed-rule share at tau 0 {np.nanmean(R[('fixed', 0)][name][first_t:]) / np.nanmean(total[('fixed', 0)][first_t:]):.2f}")
    for key in R:                                                           # from the first month whose signal month is in the test years
        R[key].update({k: sum(R[key][c] for c in v) for k, v in TIMING.items()})
        s = first_t + key[1]
        gap = np.nanmax(np.abs(sum(R[key][c] for c in CELLS) - R[key]["total"])[s:])
        assert gap < 1e-4 * np.nanmax(np.abs(R[key]["total"][s:])), f"{key}: the cells do not add up to the strategy (gap {gap:.1e})"
    A = save(out, "attribution", attribution_table(R, P.months, first_t, len(P.months) - 1), "the original strategy decomposed (fixed rule; re-optimized: appendix)")
    save(out, "attribution_size", size_table(R, RS, P.months, first_t, groups), "the fixed-rule decomposition by size group of the signal month")
    del RS, RET
    for k, (i, j) in segments.items():
        save(out, f"attribution_{k}", attribution_table({key: v for key, v in R.items() if key[1] in TAUS_FIG}, P.months, i, j), f"the same on the signal months {P.months[i]}..{P.months[j]} (Tables C4, B4)")
    save(out, "monthly", pd.DataFrame([{"weights": w, "tau": tau, "month": P.months[t], "part": k, "ret": v[t]}
                                       for (w, tau), parts in R.items() for k, v in parts.items() for t in range(first_t + tau, len(P.months))]))
    # the strategy's monthly return by part (formation month t, signal month t - tau), not printed
    log("fixed rule, share of the level: " + " ".join(f"tau {int(r.tau)} {r.share:.2f}" for r in A[(A.weights == "fixed") & (A.part == "lev") & A.tau.isin(TAUS)].itertuples()))
    for k, members in TIMING.items():                                       # the level and the deviation as their own strategies
        X = by_month(P, r04, MU, SD, lambda idx, mu, sd: sum(cell(c, x, idx, mu, sd) for c in members))
        e = (shock_where(X, P, r04, AVAIL[k](x)) if k in AVAIL else news_shock(X, P))[r04]
        curves.append(curve(LABELS[k], e, AVAIL[k](x) if k in AVAIL else None))
        del X, e
    save(out, "standalone", pd.concat(curves, ignore_index=True), "each part as its own MSRR strategy (weights re-optimized / fixed at tau 0)")

    D = P4.M[["permno", "month"]].assign(month=lambda d: pd.PeriodIndex(d.month, freq="M"), **{k: proj[k] for k in CELLS})
    D["lev"], D["dev"] = D[TIMING["lev"]].sum(axis=1), D[TIMING["dev"]].sum(axis=1)
    ret = P4.M[["permno", "month"]].assign(month=lambda d: pd.PeriodIndex(d.month, freq="M"), ret=P4.ret)
    jr = []
    for k in JOINT_LAGS:                                                    # the deviation of month t - k and the level of month t, same firm
        J = D[["permno", "month", "dev"]].assign(month=D.month + k).merge(D[["permno", "month", "lev"]], on=["permno", "month"]).merge(ret, on=["permno", "month"])
        t, _ = fama_macbeth(J, "ret", ["dev", "lev"], by="month", min_names=MIN_MONTH_STOCKS, scale=100, lags=NW_LAGS_MONTHLY)
        jr += [{"lag": k, "x": r.x, "coef_pct_per_sd": r.coef, "t": r.t, "months": r.periods} for r in t.itertuples()]
    save(out, "joint_fm", pd.DataFrame(jr), "month-ahead return on the deviation of month t - k and the level of month t (% per sd)")
    jret = P.Jret.rename("ret").reset_index().assign(month=lambda d: pd.PeriodIndex(d.month, freq="M"))   # every JKP firm-month, news or not
    sr, series = [], []
    for tau in TAUS_FIG:                                                    # the level and the deviation of month t, the return of month t + tau + 1
        J = D[["permno", "month", "lev", "dev"]].assign(month=D.month + tau).merge(jret, on=["permno", "month"])
        ics = {k: ic_series(J, k, ycol="ret", by="month", min_names=MIN_MONTH_STOCKS) for k in ["lev", "dev"]}
        series.append(pd.DataFrame(ics).rename_axis("month").reset_index().assign(month=lambda d: d.month - tau, tau=tau))
        for k, ic in ics.items():
            sr.append({"tau": tau, "signal": k, "stat": "IC_%", "value": 100 * ic.mean(), "t": nw_tstat(ic.values, NW_LAGS_MONTHLY), "months": len(ic)})
        t, _ = fama_macbeth(J, "ret", ["lev", "dev"], by="month", min_names=MIN_MONTH_STOCKS, scale=100, lags=NW_LAGS_MONTHLY)
        sr += [{"tau": tau, "signal": r.x, "stat": "FM_%_per_sd", "value": r.coef, "t": r.t, "months": r.periods} for r in t.itertuples()]
    save(out, "signals", pd.DataFrame(sr), "the level and the deviation as signals by delay: rank IC (%), joint Fama-MacBeth (% per sd)")
    save(out, "ic_monthly", pd.concat(series, ignore_index=True)[["month", "tau", "lev", "dev"]])
    return {"firm_months": int(r04.sum()), "level_share": float(x.hasL.mean())}


def main():
    global INDUSTRY
    a, log, out = start(section, timing=False, industry=False)
    INDUSTRY = a.industry
    run = timing_run if a.timing else attribution_run
    facts = run(a, log, out)
    finish(out, a, log, **facts)


if __name__ == "__main__":
    main()
