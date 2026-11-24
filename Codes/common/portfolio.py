"""Daily portfolio construction and performance statistics for the CKX tables (one implementation of each convention).

Conventions (CKX): on each trade day the stocks with a signal and a non-missing next open-to-open return are sorted into
quantiles; the long leg is the top q, the short leg the bottom q; weights are equal or proportional to the previous-day cap
(column cap_prev);
returns are open-to-open. Transaction costs: 10 bp (20 bp below the NYSE 20th cap percentile) per 100% turnover with
turnover_t = 0.5 * sum_i |w_it - w_i,t-1|. Alphas use the strategy valued at the close (close_valued) so that the holding
window matches close-to-close factor returns."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common.config import NW_LAGS_DAILY
from common.crsp import ROOT, daily_panel, o2o, open_to_close
from common.factors import dmrs_daily, ff_daily, q5_daily
from common.universe import nyse_breakpoints

ANN = 252
KMAX = 10


def nw_tstat(x, lags=NW_LAGS_DAILY):
    """Newey-West t-statistic of the mean of a daily series"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    e = x - x.mean()
    v = e @ e / len(x)
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / len(x)
    return x.mean() / np.sqrt(v / len(x))


def sharpe(r):
    r = pd.Series(r).dropna()
    return r.mean() / r.std() * np.sqrt(ANN)


def weights(sd, score, q=0.2, vw=False, min_names=20, ret="o2o_1"):
    """long / short weights per (day0, permno). The frame keeps every column of sd, so delayed returns can be read later."""
    d = sd.dropna(subset=[score, ret]).copy()
    d = d[d.groupby("day0").permno.transform("size") >= min_names]
    assert not vw or d.cap_prev.notna().all(), "value weights need the previous-day cap of every member"
    rank = d.groupby("day0")[score].rank(method="first")
    n = d.groupby("day0")[score].transform("size")
    k = np.floor(q * n + 1e-9)                                             # names per leg
    base = d.cap_prev.clip(lower=1) if vw else pd.Series(1.0, index=d.index)
    for side, member in [("long", rank > n - k), ("short", rank <= k)]:
        w = base.where(member, 0.0)
        d[f"w_{side}"] = (w / w.groupby(d.day0).transform("sum").replace(0, np.nan)).fillna(0.0)
    d["w_ls"] = d.w_long - d.w_short
    return d


def legs(W, ret="o2o_1"):
    """daily long, short and long-short returns of a weights frame; for a delayed return (o2o_k) the day-0 members are kept and
    the leg return averages over the members whose return on that day exists (fixed composition)"""
    r = W[ret]
    ok = r.notna()
    out = {}
    for leg in ["long", "short"]:
        w = W[f"w_{leg}"].where(ok, 0.0)
        out[leg] = (w * r.fillna(0)).groupby(W.day0).sum() / w.groupby(W.day0).sum().replace(0, np.nan)
    out = pd.DataFrame(out).dropna()
    out.columns = ["L", "S"]
    out["LS"] = out.L - out.S
    out["n_long"] = (W.w_long > 0).groupby(W.day0).sum().reindex(out.index)
    out["n_short"] = (W.w_short > 0).groupby(W.day0).sum().reindex(out.index)
    return out


def close_valued(W, calendar):
    """close-to-close return of the open-to-open long-short strategy, leg by leg as a self-financing position marked at the close:
    A_t = sum_i w_it (1 + intra_it) is the leg value at close_t per dollar invested at open_t,
    B_t = sum_i w_it (1 + intra_it)(1 + gap_i,t+1) / A_t its overnight gross return,
    leg return from close_t to close_{t+1} = B_t * A_{t+1} - 1 on the trading calendar (no position: A = B = 1)."""
    cal = pd.DatetimeIndex(calendar)
    cal = cal[cal >= W.day0.min()]
    cal = cal[: int((cal <= W.day0.max()).sum()) + 1]          # through the trading day after the last position (its overnight leg)
    assert W.intra0.notna().all() and W.gap1.notna().all(), "a position without its intraday or overnight return"
    intra, gap = 1 + W.intra0, 1 + W.gap1
    out = {}
    for leg in ["long", "short"]:
        v1 = (W[f"w_{leg}"] * intra).groupby(W.day0).sum()
        v2 = (W[f"w_{leg}"] * intra * gap).groupby(W.day0).sum()
        A = v1.reindex(cal).fillna(1.0)
        B = (v2 / v1).reindex(cal).fillna(1.0)
        out[leg] = B.shift(1).fillna(1.0) * A - 1                   # the first day: its open-to-close leg only
    return out["long"] - out["short"]


def ewct(target, ret, small, delist_pos, delist_ret, gammas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), large_bp=10, small_bp=20):
    """exponentially weighted calendar-time rebalancing (CKX Sec. 3.5): w_t = (1 - gamma) * drifted w_{t-1} + gamma * target_t.
    target, ret, small: (T days x N stocks) arrays of target weights, returns of held positions (NaN if missing) and small-cap flags; a day
    whose target is all zero (no portfolio that day) is not traded.
    Exit rule: a stock with a CRSP delisting event is closed on the first portfolio day after the delisting date (delist_pos)
    earning the delisting return (CRSP CIZ daily returns do not include it; a missing delisting return counts as 0); a missing
    daily return without delisting is carried at 0. The grid of days is the trading calendar of the period."""
    T, N = target.shape
    rate = np.where(small, small_bp, large_bp) / 1e4
    rows = []
    for th in gammas:
        w_prev = np.zeros(N)
        gross, net, turn = [], [], []
        for t in range(T):
            gone = t >= delist_pos
            exit_now = gone & (t == delist_pos)
            r0 = np.where(exit_now, delist_ret, np.nan_to_num(ret[t], nan=0.0))
            liq = np.abs(np.where(gone, w_prev * (1 + np.where(exit_now, delist_ret, 0.0)), 0.0))
            gain_liq = (w_prev * np.where(exit_now, delist_ret, 0.0)).sum()
            w_prev = np.where(gone, 0.0, w_prev)
            w = (1 - th) * w_prev + th * target[t] if target[t].any() else w_prev
            dw = np.abs(w - w_prev)
            turn.append(0.5 * (dw.sum() + liq.sum()))
            cost = 0.5 * ((dw + liq) * rate[t]).sum()
            g = (w * r0).sum() + gain_liq
            gross.append(g)
            net.append(g - cost)
            w_prev = w * (1 + r0) / (1 + g)                                   # the day's return incl. the liquidated positions
        gross, net = pd.Series(gross), pd.Series(net)
        rows.append({"gamma": th, "turnover_%": 100 * np.mean(turn), "gross_ret_bp": 1e4 * gross.mean(), "gross_SR": sharpe(gross),
                     "net_ret_bp": 1e4 * net.mean(), "net_SR": sharpe(net)})
    return pd.DataFrame(rows).set_index("gamma")


def market_data(log):
    """CRSP common-stock panel (next o2o return, past 5-day return, previous-day cap, NYSE breakpoints), trading calendar, factors,
    delisting events. ret_hold, the return of a held position, is o2o_1 and on a stock's last CRSP day its open-to-close return
    (the delisting return then runs from that close)"""
    crsp, tdays = daily_panel(["permno", "dlycaldt", "dlyret", "dlyopen", "dlyclose", "dlycap", "primaryexch"])
    g = crsp.groupby("permno")
    crsp["o2o_1"] = o2o(crsp, 1, g)
    cum = np.log1p(crsp.dlyret).groupby(crsp.permno).cumsum()
    crsp["past5"] = np.expm1(cum.groupby(crsp.permno).shift(1) - cum.groupby(crsp.permno).shift(6)).where(g.pos.shift(6) == crsp.pos - 6)
    crsp["cap_prev"] = g.dlycap.shift(1).where(g.pos.shift(1) == crsp.pos - 1)
    crsp["ret_hold"] = crsp.o2o_1.where(g.pos.shift(-1).notna(), open_to_close(crsp))
    mkt = crsp[["permno", "dlycaldt", "o2o_1", "ret_hold", "past5", "cap_prev"]].rename(columns={"dlycaldt": "day0"}).dropna(subset=["ret_hold"])
    del crsp, g, cum
    bp = nyse_breakpoints().set_index("day0")
    mkt = mkt.merge(bp[["nyse20", "nyse50"]], left_on="day0", right_index=True, how="left")
    ff = ff_daily()
    dl = pd.read_parquet(ROOT / "Data/raw/crsp/stkdelists.parquet")           # CIZ delisting events, one per permno
    dl = pd.DataFrame({"dlstdt": pd.to_datetime(dl.delistingdt).values, "dlret": pd.to_numeric(dl.delret).fillna(0.0).values}, index=dl.permno.astype(int).values)
    assert dl.index.is_unique, "several delisting events of one permno"
    log(f"market panel {len(mkt):,} stock-days")
    return SimpleNamespace(MKT=mkt, TDAYS=tdays, BP=bp, FF=ff, ALLF=ff.join(dmrs_daily(), how="inner").join(q5_daily(), how="inner"), DL=dl)


def ewct_table(W, mk):
    """EWCT (ewct) on the daily target long-short weights W (columns day0, permno, w_ls), with the market panel of market_data:
    the returns of held positions (ret_hold), the size class of each held stock (its last known previous-day cap against the NYSE 20th
    percentile, carried forward over days without a return) and the delisting exits"""
    TDAYS, MKT, DL = mk.TDAYS, mk.MKT, mk.DL
    days = TDAYS[(TDAYS >= np.datetime64(W.day0.min())) & (TDAYS <= np.datetime64(W.day0.max()))]
    perm = np.array(sorted(W.permno.unique()))
    pid = {x: i for i, x in enumerate(perm)}
    di = {d: i for i, d in enumerate(pd.DatetimeIndex(days))}
    T, N = len(days), len(perm)
    target, ret, small = np.zeros((T, N)), np.full((T, N), np.nan), np.full((T, N), np.nan)
    target[W.day0.map(di).values, W.permno.map(pid).values] = W.w_ls.values
    M = MKT[MKT.day0.isin(days) & MKT.permno.isin(pid)]
    ti, pi = M.day0.map(di).values, M.permno.map(pid).values
    ret[ti, pi] = M.ret_hold.values
    small[ti, pi] = np.where(M.cap_prev.notna() & M.nyse20.notna(), M.cap_prev < M.nyse20, np.nan)   # unknown: the last known class
    small = pd.DataFrame(small).ffill().fillna(0).values.astype(bool)
    delist_pos = np.full(N, T, np.int64)
    dd = DL.dlstdt.reindex(perm)
    has = dd.notna().values
    delist_pos[has] = np.searchsorted(days, dd.values[has].astype("datetime64[ns]"), side="right")
    delist_ret = DL.dlret.reindex(perm).fillna(0.0).values
    return ewct(target, ret, small, delist_pos, delist_ret).round(3)
