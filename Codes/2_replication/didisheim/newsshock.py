"""The Didisheim–Kelly–Pourmohammadi–Tian pipeline, shared by 02_msrr.py and 3_paper: the firm-month panel, the recursive z-score,
the month-by-month residualization on the rank-standardized JKP characteristics (the residual is the news shock), the managed
factors, maximum Sharpe ratio regression (MSRR), the CAPM alpha, the MSE approach (Table 7) and the delayed strategies
(Figure 10). Importing the module reads nothing and writes nothing."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common.config import MIN_MONTH_STOCKS, MSRR_GRID, MSRR_INIT, NW_LAGS_MONTHLY
from common.factors import nw_ols
from common.jkp import ROOT, characteristics, month_ranks, monthly_market, monthly_panel, monthly_sharpe, next_month
from common.portfolio import nw_tstat

FF_CHARS = {"CAPM": ["beta_60m"], "FF3": ["beta_60m", "market_equity", "be_me"],          # the characteristics behind each factor model
            "FF6": ["beta_60m", "market_equity", "be_me", "ope_be", "at_gr1", "ret_12_1"]}   # (their Figure 2): beta; size, value; profitability, investment, momentum


# ---- the firm-month panel -------------------------------------------------------------------------------------------------
def load_embeddings(model, suffix, start, end):
    """firm-month embeddings Data/clean/firm_month_emb_<model><suffix>.npz in [start, end] (YYYY-MM): keys (permno, month) and E (float32)"""
    z = np.load(ROOT / f"Data/clean/firm_month_emb_{model}{suffix}.npz")
    keys = pd.DataFrame({"permno": z["permno"].astype(int), "month": z["month"].astype(str)})
    keep = (keys.month >= start) & (keys.month <= end)
    return keys[keep].reset_index(drop=True), z["E"][keep.values].astype(np.float32)


def load_jkp(start, end, chars=None, n_chars=132):
    """JKP firm-months with a next-month excess return (not winsorized), size group, GICS industry group, market equity with the month's
    NYSE 80th percentile and the characteristics: the given list, or the n_chars characteristics with the highest coverage over the
    window (the paper's rule), rank-standardized each month across all JKP stocks of the month (Gu-Kelly-Xiu). Returns (J, chars)"""
    J = monthly_panel(start=start, end=end)
    nyse80 = J[J.crsp_exchcd == 1].groupby("month").me.quantile(0.8)        # the cap of the capped VW sorts, from every NYSE stock of the month
    allc = [c for c in characteristics() if c in J.columns]
    if chars is None:
        coverage = J.loc[J.ret_exc_lead1m.notna(), allc].notna().mean().sort_values(ascending=False)
        chars = list(coverage.index[:n_chars])
    cols = list(dict.fromkeys(list(chars) + FF_CHARS["FF6"]))
    J[cols] = month_ranks(J, cols)                                          # before the return filter: the month's whole universe
    J = J[J.ret_exc_lead1m.notna()].copy()   # as in DKPT; precedes standardization and residualization, so a missing
                                             # next-month return also drops the stock from that month's cross-section
    J["ind"] = J.gics.astype(str).str.slice(0, 4).where(J.gics.notna(), "NA")
    J["ret_exc_lead1m"] = J.ret_exc_lead1m.astype(float)
    J["me"], J["nyse80"] = J.me.astype(float), J.month.map(nyse80).astype(float)
    return J[["permno", "month", "ret_exc_lead1m", "size_grp", "ind", "me", "nyse80"] + cols].reset_index(drop=True), list(chars)


def load_panel(model, suffix, start, end, chars=None, n_chars=132):
    """the firm-months with an embedding and a JKP row with a next-month return: M (permno, month, t, return, size group, industry,
    characteristics), E aligned with M, the months, the returns, the full JKP return panel (for delayed strategies), the market"""
    keys, E_all = load_embeddings(model, suffix, start, end)
    J, chars = load_jkp(start, end, chars, n_chars)
    M = keys.reset_index().merge(J, on=["permno", "month"], how="inner")
    E = E_all[M["index"].values]
    M = M.drop(columns="index").reset_index(drop=True)
    months = sorted(M.month.unique())
    assert months == [str(p) for p in pd.period_range(months[0], months[-1], freq="M")], "the panel's months are not contiguous"
    M["t"] = M.month.map({m: i for i, m in enumerate(months)}).values
    return SimpleNamespace(M=M, E=E, months=months, n_months=len(months), ret=M.ret_exc_lead1m.values, chars=chars,
                           Jret=J.set_index(["permno", "month"]).ret_exc_lead1m, mkt=monthly_market(), keys=len(keys))


# ---- the news shock ---------------------------------------------------------------------------------------------------------------
def recursive_zscore(E, t_index, n_months, moments=False):
    """coordinate-wise z-score with moments of all firm-months through t-1 (the first month uses its own); moments=True also
    returns the mean and sd used in every month (n_months x D)"""
    Z = np.zeros_like(E)
    D = E.shape[1]
    cs, css, n = np.zeros(D), np.zeros(D), 0
    MU, SD = np.zeros((n_months, D)), np.ones((n_months, D))
    for t in range(n_months):
        idx = np.where(t_index == t)[0]
        X = E[idx].astype(np.float64)
        if n == 0:
            mu, sd = X.mean(0), np.sqrt(np.maximum(X.var(0), 1e-12))
        else:
            mu = cs / n
            sd = np.sqrt(np.maximum(css / n - mu ** 2, 1e-12))
        Z[idx] = ((X - mu) / sd).astype(np.float32)
        MU[t], SD[t] = mu, sd
        cs += X.sum(0)
        css += (X ** 2).sum(0)
        n += len(idx)
    return (Z, MU, SD) if moments else Z


def residualize(Z, M, chars, use_chars, use_ind, extra=None, fit=True):
    """month-by-month OLS of every coordinate on a constant [+ characteristics] [+ industry dummies] [+ extra columns]; returns
    residual and fit (None with fit=False)"""
    R, F = np.zeros_like(Z), (np.zeros_like(Z) if fit else None)
    for t in range(M.t.max() + 1):
        idx = np.where(M.t.values == t)[0]
        if len(idx) == 0:
            continue
        d = M.iloc[idx]
        cols = [np.ones(len(idx))]
        if use_chars:
            cols.append(d[chars].to_numpy(np.float64))                  # ranks of load_jkp (the month's JKP universe)
        if use_ind:
            cols.append(pd.get_dummies(d.ind, drop_first=True).values.astype(float))
        if extra is not None:
            cols.append(extra[idx])
        X = np.column_stack(cols)
        assert len(idx) >= 2 * X.shape[1], f"month {t}: {len(idx)} rows for {X.shape[1]} regressors"
        Y = Z[idx].astype(np.float64)
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        Yhat = X @ B
        R[idx] = (Y - Yhat).astype(R.dtype)
        if F is not None:
            F[idx] = Yhat.astype(F.dtype)
    return R, F


# ---- MSRR -------------------------------------------------------------------------------------------------------------------------
def managed_factors(X, t_index, ret, n_months, avail=None):
    """F_{t+1} = X_t' R_{t+1}: one managed portfolio per coordinate, summed over the stocks of month t (paper eq. 5-6); months with
    fewer than MIN_MONTH_STOCKS stocks stay empty, as for the delayed factors. avail (boolean per row of X; None:
    every row): the rows that count towards MIN_MONTH_STOCKS; the sum runs over all the month's rows"""
    F = np.zeros((n_months, X.shape[1]))
    for t in range(n_months):
        idx = np.where(t_index == t)[0]
        if (len(idx) if avail is None else avail[idx].sum()) >= MIN_MONTH_STOCKS:
            F[t] = X[idx].astype(np.float64).T @ ret[idx]
    return F


def lagged_factors(X, M, Jret, tau, n_months, months, avail=None):
    """F_t = X_{t-tau}' R_{t+1} over every stock with a shock at t-tau and a JKP return at t+1 (no news needed at t). avail (boolean
    per row of M; None: every row): MIN_MONTH_STOCKS counts the rows of month t-tau with avail and a return at t+1; the sum runs
    over all its rows with a return"""
    by_t = M.groupby("t").indices
    F = np.zeros((n_months, X.shape[1]))
    for t in range(tau, n_months):
        idx = by_t.get(t - tau)
        if idx is None:
            continue
        r = Jret.reindex(list(zip(M.permno.values[idx], [months[t]] * len(idx)))).values
        ok = np.isfinite(r)
        if (ok if avail is None else ok & avail[idx]).sum() < MIN_MONTH_STOCKS:
            continue
        F[t] = X[idx[ok]].astype(np.float64).T @ r[ok]
    return F


def factors(X, P, tau, avail=None):
    """the managed factors of signal X on panel P at trading delay tau (tau = 0: the month after the signal); avail: the rows of P.M that
    count towards MIN_MONTH_STOCKS (None: every row)"""
    if tau == 0:
        return managed_factors(X, P.M.t.values, P.ret, P.n_months, avail)
    return lagged_factors(X, P.M, P.Jret, tau, P.n_months, P.months, avail)


def msrr(F, init=MSRR_INIT, grid=MSRR_GRID, weights=False):
    """expanding-window MSRR with leave-one-out ridge penalty, dual form: b = F'(FF' + lam I)^-1 1 and the LOO hat values are the
    diagonal of K (K + lam I)^-1 with K = FF' (t x t); returns the OOS return earned after each signal month (and, with
    weights=True, also the T x D array of the weights b_t applied to F[t])"""
    T, D = F.shape
    out = np.full(T, np.nan)
    B = np.full((T, D), np.nan)
    for t in range(init, T):
        Ftr = F[:t]
        Ftr = Ftr[np.abs(Ftr).sum(1) > 0]
        if len(Ftr) < init or np.abs(F[t]).sum() == 0:     # at least `init` months with observations
            continue
        K = Ftr @ Ftr.T
        scale = max(np.trace(K) / D, 1e-18)
        w, U = np.linalg.eigh(K)
        w = np.maximum(w, 0)
        U1 = U.T @ np.ones(len(Ftr))
        best = None
        for lam in grid:
            dinv = 1.0 / (w + lam * scale)
            alpha = U @ (U1 * dinv)
            H = np.minimum((U ** 2 * (w * dinv)).sum(1), 1 - 1e-9)
            loo = (((1 - K @ alpha) / (1 - H)) ** 2).mean()
            if best is None or loo < best[0]:
                best = (loo, alpha)
        B[t] = Ftr.T @ best[1]
        out[t] = F[t] @ B[t]
    return (out, B) if weights else out


def fixed_weight_returns(F, B0, tau):
    """the delayed managed factors held with the undelayed strategy's weights as they stood in the signal month: F[t] @ B0[t - tau]"""
    out = np.full(len(F), np.nan)
    for t in range(tau, len(F)):
        if np.abs(F[t]).sum() > 0 and not np.isnan(B0[t - tau]).any():
            out[t] = F[t] @ B0[t - tau]
    return out


def capm(r, months, mkt):
    """annualized Sharpe ratio, mean and CAPM alpha (at 10% volatility, Newey-West t) of OOS returns indexed by signal month, earned in
    the next month; every statistic on the months with a market return"""
    r = pd.Series(r, index=months).dropna()
    x = mkt.reindex([next_month(m) for m in r.index]).values
    ok = np.isfinite(x)
    r, x = r[ok], x[ok]
    sr = monthly_sharpe(r)
    r = r / (r.std() * np.sqrt(12)) * 0.10
    b, t, _ = nw_ols(r.values, x[:, None], NW_LAGS_MONTHLY)
    return {"months": len(r), "Sharpe": sr, "mean_%/yr_at_10%vol": 1200 * r.mean(), "t_mean": nw_tstat(r.values, NW_LAGS_MONTHLY),
            "alpha_%/yr_at_10%vol": 1200 * b[0], "t_alpha": t[0]}


def lag_profile(X, P, taus, avail=None):
    """the strategy on X at every delay in taus (taus[0] = 0): weights re-optimized on the delayed factors (Sharpe) and weights fixed
    at the undelayed strategy's of the signal month (Sharpe_fixed_b, with its mean and CAPM alpha at 10% volatility); returns (table, re-optimized returns, fixed-weight returns, B0)
    with the returns indexed by formation month t (signal month t - tau, return earned in t + 1) and B0[t] the undelayed weights;
    avail: as in factors, at every delay"""
    rows, opt, fix = [], {}, {}
    for tau in taus:
        F = factors(X, P, tau, avail)
        if tau == 0:
            r, B0 = msrr(F, weights=True)                                    # B0[t]: the weights of the undelayed strategy as of month t
            rf = r
        else:
            r, rf = msrr(F), fixed_weight_returns(F, B0, tau)
        opt[tau], fix[tau] = pd.Series(r, index=P.months), pd.Series(rf, index=P.months)
        s, sf = opt[tau].dropna(), fix[tau].dropna()
        c = capm(rf, P.months, P.mkt)
        rows.append({"tau": tau, "months": len(s), "Sharpe": monthly_sharpe(s), "t": nw_tstat(s.values, NW_LAGS_MONTHLY),
                     "months_fixed_b": len(sf), "Sharpe_fixed_b": monthly_sharpe(sf), "t_fixed_b": nw_tstat(sf.values, NW_LAGS_MONTHLY),
                     **{f"{k}_fixed_b": c[k] for k in ["mean_%/yr_at_10%vol", "t_mean", "alpha_%/yr_at_10%vol", "t_alpha"]}})
    return pd.DataFrame(rows), pd.DataFrame(opt).rename_axis("formation_month"), pd.DataFrame(fix).rename_axis("formation_month"), B0


# ---- their Table 7: quintile sorts on the MSE forecast and on the MSRR weights ----------------------------------------------------------
def mse_scores(X, M, min_train_months=36):
    """expanding window with annual refit from the first year with min_train_months training months: features standardized on
    the training rows, ridge with the GCV penalty (closed-form leave-one-out approximation) of the next-month excess return (not
    winsorized), forecast of the next year's firm-months; returns the forecast, NaN outside the test years"""
    yr = M.month.str[:4].astype(int).values
    ret = M.ret_exc_lead1m.values
    pred = np.full(len(M), np.nan)
    for Y in range(yr.min() + 1, yr.max() + 1):
        tr, te = yr < Y, yr == Y   # December Y-1 carries January Y's return: realized by the first forecast (end of January Y)
        if M.t.values[tr].max() - M.t.values[tr].min() + 1 < min_train_months or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = ((X[tr] - mu) / sd).astype(np.float64)
        y = ret[tr] - ret[tr].mean()
        n = len(Z)
        G = Z.T @ Z / n
        w, V = np.linalg.eigh(G)
        g = V.T @ (Z.T @ y / n)
        best = None
        for lam in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
            b = V @ (g / (w + lam))
            gcv = ((y - Z @ b) ** 2).mean() / (1 - (w / (w + lam)).sum() / n) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, b)
        pred[te] = ((X[te] - mu) / sd) @ best[1]
    return pred


def quintile_sorts(score, M, months, mkt):
    """their Table 7 portfolios: monthly quintiles of the firm-months with a score, equal-weighted and capped value-weighted (market
    equity capped at the month's NYSE 80th percentile, as in JKP), next-month excess returns of Low ... High and High minus Low:
    annualized mean and standard deviation (the paper reports them monthly), annualized CAPM alpha with its Newey-West t, annualized
    Sharpe ratio, all on the months with a market return"""
    d = M[["t", "ret_exc_lead1m", "me", "nyse80"]].assign(s=score).dropna(subset=["s"])
    d = d[d.groupby("t").t.transform("size") >= MIN_MONTH_STOCKS]
    d["q"] = d.groupby("t").s.rank(method="first", pct=True).mul(5).apply(np.ceil).clip(1, 5).astype(int)
    d["w"] = np.minimum(d.me, d.nyse80).fillna(0.0)
    ew = d.groupby(["t", "q"]).ret_exc_lead1m.mean().unstack()
    vw = (d.ret_exc_lead1m * d.w).groupby([d.t, d.q]).sum().unstack() / d.w.groupby([d.t, d.q]).sum().unstack()
    rows = []
    for weighting, R in [("EW", ew), ("capped VW", vw)]:
        R = R.assign(HL=R[5] - R[1]).rename(columns={1: "Low (L)", 5: "High (H)", "HL": "H-L"})
        x = mkt.reindex([next_month(months[t]) for t in R.index]).values
        for col in R:
            ok = np.isfinite(R[col].values) & np.isfinite(x)
            r = R[col].values[ok]
            b, t, _ = nw_ols(r, x[ok][:, None], NW_LAGS_MONTHLY)
            rows.append({"weighting": weighting, "portfolio": str(col), "months": int(ok.sum()), "avg": 12 * r.mean(),
                         "std": np.sqrt(12) * r.std(ddof=1), "alpha": 12 * b[0], "t_alpha": t[0], "SR": monthly_sharpe(r)})
    return pd.DataFrame(rows)


# ---- the specifications of their Figure 4 ------------------------------------------------------------------------------------------
FIG4_SPECS = [("Raw", None, False), ("Constant", [], False), ("CAPM", FF_CHARS["CAPM"], False), ("FF3", FF_CHARS["FF3"], False),
              ("FF6", FF_CHARS["FF6"], False), ("JKP", "jkp", False), ("IND", [], True), ("JKP + IND", "jkp", True)]   # (label, regressors, industry)
SIZE_GROUPS = [(["mega", "large"], "Mega + Large"), (["small", "micro"], "Small + Micro")]


def news_shock(Z, P, extra=None):
    """the main news shock: the z-scored embedding residualized on the JKP characteristics [+ extra columns], month by month"""
    return residualize(Z, P.M, P.chars, True, False, extra, fit=False)[0]


def figure4(Z, P, log):
    """their Figure 4 and Figure 7: MSRR on the JKP characteristics themselves (the control) and on the z-scored embedding under every
    specification of FIG4_SPECS (raw; residual on a constant, on the characteristics of CAPM, FF3, FF6 or all JKP, with or without
    industry groups) with its predicted part, and, for the specifications without industry, within each size group. Returns the
    table, the size table, the monthly returns of the raw embedding, the JKP predicted part and the JKP news shock, the JKP news
    shock's monthly returns by size group, and the JKP news shock itself"""
    M, rows, size, rets, size_rets, main = P.M, [], [], {}, {}, None
    Xj = (M[P.chars] - M.groupby("t")[P.chars].transform("mean")).to_numpy(np.float32)   # load_jkp's ranks, demeaned among the month's firms
    rows.append({"spec": "JKP characteristics (control)", "portfolio": "control", **capm(msrr(managed_factors(Xj, M.t.values, P.ret, P.n_months)), P.months, P.mkt)})
    log(f"control: Sharpe {rows[-1]['Sharpe']:.2f}")
    for label, regs, use_ind in FIG4_SPECS:
        if regs is None:
            Xs, Fit = Z, None
        else:
            chars = P.chars if regs == "jkp" else regs
            Xs, Fit = residualize(Z, M, chars, bool(chars), use_ind)
        r = msrr(managed_factors(Xs, M.t.values, P.ret, P.n_months))
        rows.append({"spec": label, "portfolio": "raw embedding" if regs is None else "news shock", **capm(r, P.months, P.mkt)})
        log(f"{label}: news shock Sharpe {rows[-1]['Sharpe']:.2f}")
        if Fit is not None:
            rp = msrr(managed_factors(Fit, M.t.values, P.ret, P.n_months))
            rows.append({"spec": label, "portfolio": "predicted news", **capm(rp, P.months, P.mkt)})
        if regs is None:
            rets["raw embedding"] = r
        if label == "JKP":
            main, rets["predicted news"], rets["news shock"] = Xs, rp, r
        if regs is not None and not use_ind:
            for groups, name in SIZE_GROUPS:
                keep = M.size_grp.isin(groups).values
                rs = msrr(managed_factors(Xs[keep], M.t.values[keep], P.ret[keep], P.n_months))
                size.append({"spec": label, "size": name, "firm_months": int(keep.sum()), **capm(rs, P.months, P.mkt)})
                if label == "JKP":
                    size_rets[name] = rs
        del Xs, Fit
    as_frame = lambda d: pd.DataFrame(d, index=pd.Index(P.months, name="signal_month"))
    return SimpleNamespace(table=pd.DataFrame(rows), size=pd.DataFrame(size), returns=as_frame(rets), size_returns=as_frame(size_rets), shock=main)
