"""Tables and figures of paper_results.ipynb and latex.py, read from Results/paper (R = that folder) after each folder's manifest
check. Figures are saved to R/figures/<name>.pdf. Order: Section 3 (content), Section 4 (horizons), Section 5 (pricing), appendix."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from lib import DIDISHEIM_RESULTS
from lib.constants import (BANDS, CELLS, COMP, DRIFT_TABLE_MIN_STORIES, LABELS, MODELS, PAPER, PERIOD, LEVEL_LAG, POST_YEAR, PRETRAIN_CUTOFF, ROBUST_MODELS, SEGMENT_PERIOD, SEGMENTS, SHORT, FORECAST_LABEL, STANDALONE,
                           TEST_YEARS, TIMING, ARRIVAL_MONTH, TRADE_MONTH, TS_DAILY, ENCODERS, COVERAGE_GROUPS)
from lib.category import LABELS as CAT_LABELS, NEGATIVE, POSITIVE, direction
from lib.panel import SIZE_GROUPS
from lib.run import years

from common.config import FIRST_YEAR, NW_LAGS_DAILY, NW_LAGS_MONTHLY
from common.runinfo import require_same_sample

DAILY_H, MONTHLY_H = [0, 1, 2, 3, 5], [1, 2, 4, 6, 12, 18, 24]                   # the horizons h of Tables 7 and 8 (days after the trade, months after the signal month; the figures show all)
SEGMENT_MIN_MONTHS = 24                                                # months a monthly statistic on a segment's signal months needs to be shown (the B and C tables)
TIMING_PARTS = list(TIMING)
Z95 = 1.96                                                             # shaded 95% confidence bands (Newey-West), as in CKX and DKPT
CELL_COLORS = {"cL": "#1f77b4", "cD": "#aec7e8", "yF": "#ff7f0e", "uL": "#2ca02c", "uD": "#98df8a"}
PROBE_NAMES = {"headline embedding E": "Headline embedding $E$", "category template t_c alone": "Category template $E^{\\mathrm{tpl}}_c$", "E - t_c": "$E-E^{\\mathrm{tpl}}_c$",
               "category centroid alone": "Category centroid $\\bar E_c$", "E - centroid (within-category)": "$E-\\bar E_c$"}
PARTS_TEX = {"between input (z_C)": "Category centroid $\\beta'\\bar E_c$", "within input (z_perp)": "Within category $\\beta'(E-\\bar E_c)$",
             "event category": "Event category $\\hat y^C$", "firm identity": "Firm identity $\\hat y^F$", "idiosyncratic": "Idiosyncratic $\\hat y^I$",
             "forecast": "Total forecast $\\hat y$", "event category: level": "Event category, level $\\hat y^C_L$",
             "event category: deviation": "Event category, deviation $\\hat y^C_D$", "idiosyncratic: level": "Idiosyncratic, level $\\hat y^I_L$",
             "idiosyncratic: deviation": "Idiosyncratic, deviation $\\hat y^I_D$", "historical level": "Level $\\hat y_L$",
             "deviation from the level": "Deviation $\\hat y_D$", "idiosyncratic: own 20-day mean": "20-day mean $\\hat y^I_H$",
             "idiosyncratic: minus own 20-day mean": "Remainder $\\hat y^I-\\hat y^I_H$"}   # display names of the forecast's components (tables and figures)
EMBEDDING_TEX = {LABELS[k]: v for k, v in [("cL", "Event category, level $C_L$"), ("cD", "Event category, deviation $C_D$"), ("yF", "Firm identity $F$"),
                                            ("uL", "Idiosyncratic, level $I_L$"), ("uD", "Idiosyncratic, deviation $I_D$"), ("lev", "Level $E_L$"),
                                            ("dev", "Deviation $E_D$")]}   # Section 5 (Didisheim): parts of the residualized embedding, not of the forecast
INFO_GROUPS = ("earnings", "revenues", "analyst-ratings", "price-targets", "insider-trading", "dividends", "labor-issues", "equity-actions")


# ---- plumbing ------------------------------------------------------------------------------------------------------------------
def show(fig, R, name):
    """save a figure as R/figures/<name>.pdf and return it (the notebook displays the returned figure)"""
    d = R / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")             # legends below the axes stay inside the file
    return fig


def checked(d, what):
    """the manifest of a results folder, or an error: another specification or a partial --years run never feeds the paper"""
    s = str(d)
    info = require_same_sample(d, what, needs=PAPER if "Results/paper" in s else ("ridge",) if "replication/ckx" in s else ("monthly",) if "replication/didisheim" in s else ())
    yrs = info.get("args", {}).get("years")
    if yrs and list(years(yrs)) != list(TEST_YEARS):
        raise RuntimeError(f"{what} is a partial run (--years {yrs}); rerun it on every test year")
    return info


def models(R, part, encoders=MODELS):
    """{model: label} of R/<part>/<model>/, each folder's manifest checked; every model must have results"""
    for m in encoders:
        checked(R / part / m, f"{part}/{m}")
    return dict(encoders)


def segment_label(m, s):
    """the signal years of model m's segment s (lib.constants.SEGMENTS)"""
    return "{}--{}".format(*SEGMENTS[m][s])


def runs(R):
    """every run.json under R: analysis, finished"""
    rows = [{"analysis": str(f.parent.relative_to(R)), "finished": json.loads(f.read_text())["finished"]} for f in sorted(R.rglob("run.json"))]
    return pd.DataFrame(rows).set_index("analysis")


def read(path, manifest=None, **kw):
    """pd.read_csv of a file inside a results folder, after the check of that folder's manifest (run.json, or `manifest`)"""
    checked(manifest or path.parent, str(path.parent))
    return pd.read_csv(path, **kw)


def table(R, part, name, m=None):
    return read(R / part / (m or "") / f"{name}.csv")


def per_model(R, part, fn, encoders=MODELS):
    """fn(model) for every model, stacked under a 'model' level"""
    return pd.concat({lab: fn(m) for m, lab in models(R, part, encoders).items()}, names=["model"])


def _legend(ax, n=None):
    """inside the axes for up to four entries, below it otherwise"""
    n = n if n is not None else len(ax.get_legend_handles_labels()[0])
    if n <= 4:
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))


def _row_legend(fig, row, ncol):
    """one legend centered under a row of axes, handles of its first axes; call after the last layout change (show's tight_layout
    ignores figure legends)"""
    fig.tight_layout()
    l, r = row[0].get_position(), row[-1].get_position()
    bottom = min(ax.get_tightbbox(fig.canvas.get_renderer()).y0 for ax in row)   # below the x labels, in pixels
    fig.legend(*row[0].get_legend_handles_labels(), fontsize=7, frameon=False, ncol=ncol, loc="upper center",
               bbox_to_anchor=((l.x0 + r.x1) / 2, fig.transFigure.inverted().transform((0, bottom))[1]), columnspacing=2.5)


def _horizon_ticks(ax, hs, x):
    """ticks at the positions x of the horizons hs: every horizon labelled up to twelve of them, otherwise the first and every sixth"""
    keep = [i for i, h in enumerate(hs) if len(hs) <= 12 or i == 0 or h % 6 == 0]
    ax.set_xticks([x[i] for i in keep])
    ax.set_xticklabels([hs[i] for i in keep])


def _years_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _h_axis(ax, hs):
    """a line plot over horizons hs: the axis spans the first to the last horizon, ticks as _horizon_ticks"""
    hs = sorted(set(int(h) for h in hs))
    ax.set_xlim(hs[0], hs[-1])
    _horizon_ticks(ax, hs, hs)


def _regression(f, col, reg, coef, t, regs=None, digits=2):
    """academic regression layout: one column per regression (values of `col`), a coefficient row per regressor with its t below
    in parentheses"""
    cols = list(f[col].drop_duplicates())
    rows, index = [], []
    for r in regs or list(f[reg].drop_duplicates()):
        d = f[f[reg] == r].set_index(col).reindex(cols)
        rows += [[f"{v:.{digits}f}" if pd.notna(v) else "" for v in d[coef]], [f"({v:.2f})" if pd.notna(v) else "" for v in d[t]]]
        index += [LABELS.get(r, r), ""]
    return pd.DataFrame(rows, index=index, columns=cols)


# ---- Section 3: what predicts --------------------------------------------------------------------------------------------------
COST_GAMMAS = (1.0, 0.5)                                               # Table 1's net SR: daily rebalancing, and EWCT with a one-day half-life
TABLE1 = {"A": ["Category mean (8-yr)", "Category mean (1-yr)", *[FORECAST_LABEL[m] for m in MODELS], "RavenPack ESS"],
          "B": ["Event-group mean", "Direction mean", "RavenPack-type mean", "Group x direction mean", "LMMD sentiment"]}


def table1(R):
    """Table 1 on all firm-days, {A: the category means against the embedding forecasts and ESS, B: the coarser event definitions},
    the same columns: cells, IC, the EW and VW decile long-shorts with their net Sharpe ratios under EWCT at COST_GAMMAS, the
    names per decile leg on an average day"""
    b = table(R, "1_category", "benchmark")
    lv = table(R, "1_category", "levels")
    t = pd.concat([b[(b["sample"] == "all") & (b.period == PERIOD)], lv[~lv.signal.isin(b.signal) & (lv.period == PERIOD)]]).set_index("signal")
    c = table(R, "1_category", "costs")
    t["cells"] = lv.set_index("signal").cells.reindex(t.index)
    t.loc["Category mean (1-yr)", "cells"] = t.loc["Category mean (8-yr)", "cells"]
    net = c.set_index(["weight", "gamma", "signal"]).net_SR.sort_index()
    for w in ["EW", "VW"]:
        for th in COST_GAMMAS:
            t[f"net_{w}_{th:g}"] = net[w, th]
    n = leg_sizes(R)
    t["names_leg"] = ((n.long + n.short) / 2).reindex(t.index).round()
    t = t[["cells", "IC_%", "t_IC"] + [c for w in ["EW", "VW"] for c in [f"{w}_LS_bp", f"{w}_Sharpe"] + [f"net_{w}_{th:g}" for th in COST_GAMMAS]] + ["names_leg"]]
    return {k: t.loc[rows] for k, rows in TABLE1.items()}


def leg_sizes(R):
    """names on an average day of each signal's decile long-short: all, in the long and in the short leg"""
    return table(R, "1_category", "legs").set_index("signal")


def table_tests(R):
    """each Table 1 signal's difference from the 8-year category mean on all firm-days: IC (Diebold-Mariano t), EW and VW Sharpe (Jobson-Korkie z)"""
    b = table(R, "1_category", "benchmark")
    b = b[(b["sample"] == "all") & (b.period == PERIOD)].set_index("signal")
    return b[["d_IC_%", "DM_t", "d_Sharpe", "JK_z", "d_Sharpe_VW", "JK_z_VW"]].dropna(how="all")


def table_fm(R, name="fm", encoders=MODELS):
    """Table 2 Panel A in the regression layout: daily Fama-MacBeth of the next-day return on each embedding forecast alone, with the
    category mean and with the group x direction mean (within-day standardized, bp per sd), one column block per encoder, each
    coefficient with its Newey-West t below in parentheses"""
    f = table(R, "1_category", name)
    label = {FORECAST_LABEL[m]: SHORT[m] for m in encoders}
    f["model"] = f.spec.map(lambda s: next((v for k, v in label.items() if k in s.split(" + ")), None))   # the spec joins its signals' labels
    f = f[f.model.notna()].assign(regressor=lambda d: d.regressor.replace({k: "Embedding forecast" for k in label}))
    out = pd.concat({m: _regression(d, "spec", "regressor", "coef", "t_NW", regs=["Embedding forecast", "Category mean (8-yr)", "Group x direction mean"])
                     for m, d in f.groupby("model", sort=False)}, axis=1)
    out.columns = pd.MultiIndex.from_tuples([(m, f"({k + 1})") for k, (m, _) in enumerate(out.columns)])
    return out


def table_residual(R, name="residual", encoders=MODELS):
    """Table 2 Panel B: the 8-year category mean and, for every model, the category mean plus the ridge forecast of its residual, with
    the tests against the category mean (DM t, JK z), the out-of-sample R2 against a zero forecast (bp) and the Newey-West t of the
    squared-error difference from the category mean"""
    cols = ["IC_%", "DM_t", "EW_Sharpe", "JK_z", "VW_Sharpe", "JK_z_VW", "R2_bp", "t_mse"]
    res = {lab: table(R, "2_probe", name, m).set_index("signal") for m, lab in models(R, "2_probe", encoders).items()}
    rows = {"Category mean (8-yr)": next(iter(res.values())).loc["category mean (8-yr)", cols]} | {f"+ residual ridge, {lab}": r.loc["category mean + residual ridge", cols] for lab, r in res.items()}
    return pd.DataFrame(rows).T


def table_signals(R):
    """Appendix: one row per signal: a return forecast's Mincer-Zarnowitz slope alone (daily Fama-MacBeth in return units) with
    the t of b = 1, and the overlap with the classifications (the share of the signal's variance between event categories, its
    correlations with the 8-year category and group x direction means)"""
    f = table(R, "1_category", "fm_raw").set_index("regressor")
    s = table(R, "1_category", "signals").set_index("signal")
    t = pd.DataFrame({"b": f.coef, "t_b1": f.t_vs_1}).join(s, how="outer")
    rows = [FORECAST_LABEL[m] for m in MODELS] + ["Category mean (8-yr)", "Group x direction mean", "RavenPack ESS"]   # as in Panel A, then ESS
    return t.loc[rows, ["b", "t_b1", "between_category_share_%", "corr_with_m8", "corr_with_b8"]]


def table_split(R, seg=None, encoders=MODELS):
    """Table 6: the forecast's content parts C, F, I (two-way projection) with their shares of its covariance with the return and of
    its variance, and each part as a signal; seg: on that segment's signal years (Tables C2, B2)"""
    def one(m):
        t = table(R, "3_decompose", "decomposition", m)
        return (t[t.period == (SEGMENT_PERIOD[m][seg] if seg else PERIOD)].set_index("component")
                .loc[COMP + ["yhat"], ["A_j", "var_share", "IC_%", "EW_Sharpe", "VW_Sharpe"]].rename(index=LABELS))
    return per_model(R, "3_decompose", one, encoders)


def table_centroid(R):
    """appendix: the one-way split of the forecast into the part on the story's category centroid and the rest"""
    return per_model(R, "2_ridge", lambda m: table(R, "2_ridge", "decomposition", m).query("period == @PERIOD").set_index("label")
                     .loc[["between input (z_C)", "within input (z_perp)", "forecast"], ["A_j", "var_share", "IC_%", "EW_Sharpe", "VW_Sharpe"]])


SUE_READ = {"C": ("$K$", "OLS"), "C + forecast": ("$K+\\hat y$", "OLS"), "C + E (ridge)": ("$K+E$", "Ridge"), "C + ED (ridge)": ("$K+E_D$", "Ridge"),
            "C + EL (ridge)": ("$K+E_L$", "Ridge"), "C + ED + EL (ridge)": ("$K+E_D+E_L$", "Ridge")}
SUE_BASE = {k: SUE_READ[k] for k in ["C", "C + forecast", "C + E (ridge)", "C + ED + EL (ridge)"]}   # Panel B's rows without the surprise
SUE_PREDICT = {**SUE_BASE, "C + ES": ("$K+\\mathrm{SUE}$", "OLS"), "C + ES + forecast": ("$K+\\mathrm{SUE}+\\hat y$", "OLS"),
               "C + ES + E (ridge)": ("$K+\\mathrm{SUE}+E$", "Ridge"), "C + ES + ED + EL (ridge)": ("$K+\\mathrm{SUE}+E_D+E_L$", "Ridge")}


def table_sue(R):
    """Table 3, the models side by side, one row per specification and its method: {read: the surprise rank recovered from the
    headline (OOS R2 in percent, Spearman; fitted on earlier years); predict: the first trading day after the announcement (OOS R2 and rank IC of the rolling fit)}"""
    def side(name, rows, cols):
        t = pd.concat({lab: table(R, "3_surprise", name, m).set_index("spec").loc[list(rows), cols] for m, lab in models(R, "3_surprise").items()}, axis=1)
        t.index = pd.MultiIndex.from_tuples(rows.values(), names=["Specification", "Method"])
        return t
    read = side("panel_a", SUE_READ, ["R2_oos", "spearman"])
    read.loc[:, (slice(None), "R2_oos")] *= 100                        # in percent, as Panel B
    return {"read": read.rename(columns={"R2_oos": "R2_oos_%"}, level=1), "predict": side("panel_b", SUE_PREDICT, ["R2_%", "IC_%"])}


def plot_drift_group(R, group, name):
    """one figure per event group: the cumulative excess return from the first open after the story of each category
    with enough stories; the jump to that open (from the last close before the story; from the day's open for news during the
    session) is in the label"""
    c = table(R, "1_drift", "drift_category")
    dg = c[c.group == group]
    assert not dg.empty, f"no category of {group} has a drift path"
    h0 = dg[dg.h == 0].set_index("category")
    cats = h0.car_bp.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 4))                          # as wide as the three legend columns below it
    colors = plt.cm.tab10.colors + plt.cm.Dark2.colors
    for i, cat in enumerate(cats):
        p = dg[dg.category == cat].sort_values("h")
        p = p.assign(car_bp=np.where(p.h == 0, 0.0, p.car_bp), t=np.where(p.h == 0, np.nan, p.t))   # the path starts at 0 at the first open
        se = (p.car_bp / p.t.replace(0, np.nan)).abs().fillna(0).values
        ax.plot(p.h, p.car_bp, marker="o", ms=3, lw=1.4, color=colors[i % len(colors)], label=f"{cat} ({h0.stories[cat] / 1000:.0f}k, jump {h0.car_bp[cat]:+.0f} bp)")
        ax.fill_between(p.h, p.car_bp - Z95 * se, p.car_bp + Z95 * se, color=colors[i % len(colors)], alpha=0.08)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(0, dg.h.max())
    ax.set_xlabel("trading days after the first open after the headline", fontsize=8)
    ax.set_ylabel("cumulative excess return (bp)", fontsize=8)
    ax.tick_params(labelsize=8)
    _years_axis(ax)
    _legend(ax, len(cats))
    return show(fig, R, name)


def plot_drift(R):
    return [plot_drift_group(R, g, f"fig1_drift_{g}") for g in INFO_GROUPS]


# ---- Section 4: horizons -------------------------------------------------------------------------------------------------------
FORECASTS = {"daily": "Daily forecast", "bridge": "Monthly average of the daily forecast", "ridge": "Monthly-trained forecast"}   # Table 7's panels
HORIZON_AXIS = {"daily": "$h$ (trading days)", "bridge": "$h$ (months)", "ridge": "$h$ (months)", "strategy": "$h$ (months)"}
LEGEND = {"lev": "Level", "dev": "Deviation", "cL": "Event category, level", "cD": "Event category, deviation", "yF": "Firm identity",
          "uL": "Idiosyncratic, level", "uD": "Idiosyncratic, deviation"}   # figure legends in words: one picture serves different maps of the embedding


def horizons(R, m, seg=None):
    """every part's contribution to each forecast's covariance with the return (x 1e6), long: forecast, h, part, mean, t, share
    (daily: 4_horizon, h days after the trade; monthly: 4_monthly_ridge, h = tau + 1 months after the signal month, every month
    of the first 36); seg: on that segment's signal years (Tables C3, B3), the monthly rows with at least SEGMENT_MIN_MONTHS months"""
    suffix = f"_{seg}" if seg else ""
    d = table(R, "4_horizon", "term_structure" + suffix, m).assign(forecast="daily")
    mo = table(R, "4_monthly_ridge", "contributions" + suffix, m)
    if seg:
        mo = mo[mo.periods >= SEGMENT_MIN_MONTHS]
    T = pd.concat([d, mo.assign(forecast=mo.predictor, h=mo.tau + 1)], ignore_index=True)
    return T.assign(part=T.part.replace({"yhat": "total"}))


def horizon_table(R, parts, total, seg=None, encoders=MODELS):
    """Table 7 (timing parts and the forecast; seg: Tables C3, B3) and Table A14 (the cells): {forecast: rows (model, h), columns (part,
    Cov / t / Share (%))}"""
    def one(T, f, hs):
        T = T[(T.forecast == f) & T.h.isin(hs)]
        get = lambda k: T[T.part == k].set_index("h")
        cols = {("Forecast", "Cov"): get("total")["mean"], ("Forecast", "$t$"): get("total").t} if total else {}
        for k in parts:
            cols.update({(PARTS_TEX[LABELS[k]], "Cov"): get(k)["mean"], (PARTS_TEX[LABELS[k]], "$t$"): get(k).t,
                         (PARTS_TEX[LABELS[k]], "Share (%)"): (100 * get(k).share).where(get("total").t.abs() >= 2)})   # blank where the total is indistinguishable from zero
        return pd.DataFrame(cols)
    T = {m: horizons(R, m, seg) for m in models(R, "4_horizon", encoders)}
    return {f: per_model(R, "4_horizon", lambda m: one(T[m], f, DAILY_H if f == "daily" else MONTHLY_H), encoders) for f in FORECASTS}


def decomposition_figure(R, data, xlabel, ylabel, total, name):
    """Figures 2 and 3, one column per model: the level and the deviation as stacked bars with the total as diamonds (top) and the
    cells with 95% bands (bottom) by horizon h; data {model label: long table part, h, mean, t}"""
    fig, axes = plt.subplots(2, len(data), figsize=(6.5 * len(data), 7.4), squeeze=False, gridspec_kw={"hspace": 0.35})
    for j, (lab, A) in enumerate(data.items()):
        hs = sorted(A.h.unique())
        x = np.arange(len(hs))
        col = lambda k, c: A[A.part == k].set_index("h").reindex(hs)[c].values
        top, bot = axes[0, j], axes[1, j]
        lev, dev = col("lev", "mean"), col("dev", "mean")
        top.bar(x, lev, 0.6, color=CELL_COLORS["yF"], label=LEGEND["lev"])
        top.bar(x, dev, 0.6, bottom=np.where(dev * lev > 0, lev, 0), color=CELL_COLORS["cL"], label=LEGEND["dev"])
        top.plot(x, col("total", "mean"), "kD", ms=4, label=total)
        for k in CELLS:
            v, se = col(k, "mean"), np.abs(col(k, "mean") / col(k, "t"))
            bot.plot(x, v, marker="o", ms=3, color=CELL_COLORS[k], label=LEGEND[k])
            bot.fill_between(x, v - Z95 * se, v + Z95 * se, color=CELL_COLORS[k], alpha=0.08)
        for ax, title in [(top, "timing decomposition"), (bot, "full decomposition")]:
            ax.axhline(0, color="grey", lw=0.5)
            _horizon_ticks(ax, hs, x)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_title(f"{lab}: {title}", fontsize=9, loc="left")
            ax.tick_params(labelsize=8)
        top.set_ylabel(ylabel, fontsize=8)
        bot.set_ylabel(ylabel, fontsize=8)
    _legend(axes[0, 0], 3)
    _row_legend(fig, axes[1], len(CELLS))
    return show(fig, R, name)


def plot_horizons(R):
    """Figure 2, one file per forecast of Table 7 (fig2a daily from h = 1, fig2b its monthly average, fig2c the monthly-trained forecast)"""
    T = {lab: horizons(R, m) for m, lab in models(R, "4_horizon").items()}
    return [decomposition_figure(R, {lab: t[(t.forecast == f) & (t.h >= 1)] for lab, t in T.items()}, HORIZON_AXIS[f], "Cov ($\\times 10^6$)",
                                 "Forecast", f"fig2{letter}") for letter, f in zip("abc", FORECASTS)]


# ---- Section 5: pricing --------------------------------------------------------------------------------------------------------
def attribution(R, m, weights="fixed", name="attribution"):
    """4_didisheim's decomposition of the replication's strategy by horizon h = tau + 1 months after the signal month (fixed
    weights: every month of the first 36; name "attribution_<segment>": on that segment's signal months, the horizons with at least
    SEGMENT_MIN_MONTHS months)"""
    A = table(R, "4_didisheim/attribution", name, m)
    if name != "attribution":
        A = A[A.months >= SEGMENT_MIN_MONTHS]
    return A[A.weights == weights].assign(h=lambda d: d.tau + 1, mean=lambda d: d["mean_%/yr_at_10%vol"])


def table_attribution(R, weights="fixed", name="attribution", encoders=MODELS):
    """Table 8: the replication's strategy by horizon, decomposed exactly by timing (level, deviation) and by the full decomposition;
    means in %/yr at 10% annual volatility of the strategy, shares in %"""
    def one(m, parts, strategy):
        A = attribution(R, m, weights, name)
        A = A[A.h.isin(MONTHLY_H)]
        tot = A[A.part == "total"].set_index("h")
        cols = {("Embedding $E$", "SR"): tot.Sharpe, ("Embedding $E$", "Ret"): tot["mean"], ("Embedding $E$", "$t$"): tot.t} if strategy else {}
        for k in parts:
            d = A[A.part == k].set_index("h")
            cols.update({(EMBEDDING_TEX[LABELS[k]], "Ret"): d["mean"], (EMBEDDING_TEX[LABELS[k]], "$t$"): d.t,
                         (EMBEDDING_TEX[LABELS[k]], "Share (%)"): (100 * d.share).where(tot.t.abs() >= 2)})   # blank where the total is indistinguishable from zero
        return pd.DataFrame(cols)
    return {"timing": per_model(R, "4_didisheim/attribution", lambda m: one(m, TIMING_PARTS, True), encoders),
            "full": per_model(R, "4_didisheim/attribution", lambda m: one(m, CELLS, False), encoders)}


def plot_attribution(R):
    """Figure 3: Table 8 at every horizon, one column per model"""
    data = {lab: attribution(R, m) for m, lab in models(R, "4_didisheim/attribution").items()}
    return decomposition_figure(R, data, HORIZON_AXIS["bridge"], "Return (%/yr, MSRR strategy at 10% volatility)", "MSRR strategy", "fig3_attribution")


# ---- appendix: size groups -----------------------------------------------------------------------------------------------------
SIZE_NAMES = {"micro": "Micro", "small": "Small", "big": "Big"}       # Fama and French (2008) groups on NYSE breakpoints (lib.panel.size_group)
SIZE_COLORS = {"micro": "#d62728", "small": "#ff7f0e", "big": "#1f77b4"}


def _size_lines(ax, d, pos, value):
    """one line per size group with its 95% band; d: rows size, h, `value`, t; pos: {h: x position}"""
    for g in SIZE_GROUPS:
        p = d[d["size"] == g].set_index("h").sort_index()
        v, x = p[value].values, [pos[h] for h in p.index]
        se = np.abs(v / p.t.replace(0, np.nan).values)
        ax.plot(x, v, marker="o", ms=3, lw=1.4, color=SIZE_COLORS[g], label=SIZE_NAMES[g])
        ax.fill_between(x, v - Z95 * se, v + Z95 * se, color=SIZE_COLORS[g], alpha=0.10, lw=0)
    ax.axhline(0, color="grey", lw=0.5)
    ax.tick_params(labelsize=8)


def plot_drift_size(R):
    """the top-minus-bottom decile path of the 8-year category mean within each size group (1_drift, drift_ls_size): from the first
    open after the story, the jump in the label"""
    d = table(R, "1_drift", "drift_ls_size")
    d = d[d.signal == CAT_LABELS["m8"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    jump = d[d.h == 0].set_index("size").ls_bp
    p = d.assign(ls_bp=np.where(d.h == 0, 0.0, d.ls_bp), t=np.where(d.h == 0, np.nan, d.t))     # the path starts at 0 at the first open
    hs = sorted(p.h.unique())
    _size_lines(ax, p.assign(t=p.t.fillna(np.inf)), {h: h for h in hs}, "ls_bp")
    for line, g in zip(ax.get_lines(), SIZE_GROUPS):
        line.set_label(f"{SIZE_NAMES[g]} (jump {jump[g]:+.0f} bp)")
    ax.set_xlim(0, max(hs))
    ax.set_xlabel("trading days after the first open after the headline", fontsize=8)
    ax.set_ylabel("cumulative top-minus-bottom decile return (bp)", fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return show(fig, R, "figA_drift_size")


def size_figure(R, data, value, ylabel, names, name):
    """one row per cell and one column per model, a line per size group; rows share the vertical axis; data {model label: long
    table size, part, h, value, t}"""
    fig, axes = plt.subplots(len(CELLS), len(data), figsize=(5 * len(data), 3 * len(CELLS)), squeeze=False, sharex=True, sharey="row")
    for j, (lab, A) in enumerate(data.items()):
        hs = sorted(A.h.unique())
        x = np.arange(len(hs))
        for i, k in enumerate(CELLS):
            ax = axes[i, j]
            _size_lines(ax, A[A.part == k], dict(zip(hs, x)), value)
            _horizon_ticks(ax, hs, x)
            ax.set_title(f"{lab}: {names[LABELS[k]]}", fontsize=9, loc="left")
            if i == len(CELLS) - 1:
                ax.set_xlabel(HORIZON_AXIS["bridge"], fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    return show(fig, R, name)


def plot_horizons_size(R):
    """Table 7's monthly average of the daily forecast within each size group of the signal month (4_monthly_ridge, contributions_size)"""
    data = {lab: table(R, "4_monthly_ridge", "contributions_size", m).assign(h=lambda d: d.tau + 1)
            for m, lab in models(R, "4_monthly_ridge").items()}
    return size_figure(R, data, "mean", "Cov ($\\times 10^6$)", PARTS_TEX, "figA_bridge_size")


def plot_attribution_size(R):
    """Figure 3 split exactly by the size group of the signal month (4_didisheim, attribution_size): the groups add up to Figure 3"""
    data = {lab: table(R, "4_didisheim/attribution", "attribution_size", m).assign(h=lambda d: d.tau + 1, mean=lambda d: d["mean_%/yr_at_10%vol"])
            for m, lab in models(R, "4_didisheim/attribution").items()}
    assert all(set(d["size"]) == set(SIZE_GROUPS) for d in data.values()), "firms without a size group: the three groups would not add up to Figure 3"
    return size_figure(R, data, "mean", "Return (%/yr, MSRR strategy at 10% volatility)", EMBEDDING_TEX, "figA_attribution_size")


# ---- appendix: Section 3 -------------------------------------------------------------------------------------------------------
def table_subsamples(R):
    """the rank IC and the EW and VW Sharpe ratios of Table 1's upper signals by sample: {measure: samples x (the sample's share of the
    stories, signals)}"""
    b = table(R, "1_category", "benchmark")
    share = b.groupby("sample", sort=False).stories_pct.first()
    return {m: b.pivot_table(index="sample", columns="signal", values=m, sort=False)[TABLE1["A"]].assign(stories_pct=share)[["stories_pct"] + TABLE1["A"]]
            for m in ["IC_%", "EW_Sharpe", "VW_Sharpe"]}


def table_costs(R):
    """the daily signals' EW and VW decile long-short under EWCT rebalancing: {gross, net}: Sharpe ratio by signal (rows) and
    weighting x gamma (columns)"""
    c = table(R, "1_category", "costs")
    c = c[c.gamma.round(2).isin([0.1, 0.2, 0.3, 0.5, 1.0])]
    return {v: c.pivot(index="signal", columns=["weight", "gamma"], values=f"{v}_SR").loc[pd.unique(c.signal)] for v in ["gross", "net"]}


def table_spanning(R):
    t = table(R, "1_category", "spanning")
    return t.set_index(["factors", "weight", "dependent", "on"])[["beta", "alpha_bp_day", "t_alpha", "R2", "appraisal_ratio"]]


def table_drift(R):
    """the drift of the categories with at least DRIFT_TABLE_MIN_STORIES stories and of the decile long-short portfolios: the jump,
    day 1 and days 1-10 from the first open after the story; within a group by event type, then direction (+, none, -)"""
    c = table(R, "1_drift", "drift_category")
    c = c[c.stories >= DRIFT_TABLE_MIN_STORIES]
    h0, h1, hH = (c[c.h == h].set_index("category") for h in (0, 1, c.h.max()))
    t = pd.DataFrame({"group": h0.group, "stories": h0.stories, "jump_bp": h0.car_bp, "t_jump": h0.t, "day1_bp": h1.car_bp, "t_day1": h1.t,
                      "days_1_10_bp": hH.car_bp, "t_1_10": hH.t})
    ls = table(R, "1_drift", "drift_ls")
    l0, l1, lH = (ls[ls.h == h].set_index("signal") for h in (0, 1, ls.h.max()))
    rows = pd.DataFrame({"group": "decile long-short", "stories": np.nan, "jump_bp": l0.ls_bp, "t_jump": l0.t, "day1_bp": l1.ls_bp, "t_day1": l1.t,
                         "days_1_10_bp": lH.ls_bp, "t_1_10": lH.t})
    t["type"] = ["-".join(w for w in c.split("-") if w not in POSITIVE | NEGATIVE | {"neutral"}) for c in t.index]
    t["sign"] = t.index.map(lambda c: "+0-".index(direction(c)))
    t = pd.concat([t.sort_values(["group", "type", "sign"]).drop(columns=["type", "sign"]), rows])
    return t.reset_index().rename(columns={"index": "category"}).set_index(["group", "category"])


def plot_stability(R, rows="A", name="figA_stability"):
    """rank IC and EW decile long-short Sharpe ratio by test year of Table 1's upper rows (rows="A") or of its coarser classifications
    with the 8-year category mean for reference (rows="B")"""
    y = table(R, "1_category", "by_year")
    keep = TABLE1["A"] if rows == "A" else ["Category mean (8-yr)"] + TABLE1["B"]
    y = y[y.signal.isin(keep)].assign(order=lambda d: d.signal.map(keep.index)).sort_values(["order", "year"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for col, ax, title in [("IC_%", axes[0], "Rank IC by year (%)"), ("EW_Sharpe", axes[1], "EW decile long-short Sharpe ratio by year")]:
        for s, d in y.groupby("signal", sort=False):
            ax.plot(d.year, d[col], marker="o", ms=3, label=s.replace(" x ", " × "))
        ax.set_title(title, fontsize=9, loc="left")
        ax.axhline(0, color="grey", lw=0.5)
        _years_axis(ax)
    _row_legend(fig, axes, y.signal.nunique())
    return show(fig, R, name)


def table_decomposition_full(R):
    """every component of the daily decomposition: content, cells, timing and the 20-day split"""
    sigs = COMP + CELLS + list(TIMING) + TS_DAILY + ["yhat"]
    return per_model(R, "3_decompose", lambda m: table(R, "3_decompose", "decomposition", m).query("period == @PERIOD")
                     .set_index("component").loc[lambda d: d.index.isin(sigs), ["A_j", "score_share", "var_share", "IC_%", "EW_Sharpe", "VW_Sharpe"]]
                     .rename(index=LABELS))


REWRITES = ["magnitude x10", "magnitude /10", "direction flipped", "name swapped (same industry)", "name removed"]   # 2_perturb's order


def table_rewrites(R):
    def one(m):
        p = table(R, "2_perturb", "summary", m).set_index("variant").loc[REWRITES, ["n", "d_dir", "abs_d_yhat_sd", "correct_sign",
                                                                                              "abs_d_rank", "decile_change", "cosine"]]
        a = read(R / "2_spectrum" / m / "axes.csv").groupby("axis")[["in_category_subspace", "in_firm_subspace"]].mean()
        return p.join(a.reindex([k.replace("magnitude /", "magnitude x") for k in p.index]).set_axis(p.index))   # /k shares the axis of xk
    return per_model(R, "2_perturb", one)


def spectrum_bands(R):
    """by band of principal directions and over all of them (Total): the band's share of the variance and of the forecast's test-year
    covariance with the return, the mean shrinkage, and each part's share of the band's variance and covariance, in percent"""
    def one(m):
        b = table(R, "2_spectrum", "bands", m).set_index("band").drop(columns="directions")
        pct = b.columns.drop("shrinkage")
        b[pct] *= 100
        return b.rename(index=lambda k: "Total" if k == "all" else k.replace("-", "--"))
    return per_model(R, "2_spectrum", one)


def table_groups(R):
    """by event group: stories, rank IC of the category mean and the forecast, shares (%) of the forecast's predictive
    covariance (blank where that covariance has |t| < 2, a denominator indistinguishable from zero) and the
    contribution to the overall EW decile long-short"""
    def one(m):
        t = table(R, "3_by_group", "by_group", m).set_index("group")
        shares = ["A_yC", "A_yF", "A_yU", "A_lev", "A_dev"]
        t[shares] = (100 * t[shares]).where(t.t_forecast.abs() >= 2)
        return t[["stories", "IC_category_mean_%", "IC_forecast_%"] + shares + ["contribution_bp_day"]]
    return per_model(R, "3_by_group", one)


def table_factors(R):
    """alpha (t), R2 and the profitability and value loadings of the decile long-short on the forecast and its parts, EW and VW"""
    cols = ["alpha_%/yr", "t_alpha", "R2", "b_rmw", "b_hml"]
    def one(m):
        f = table(R, "3_factors", "factors", m).set_index(["part", "weight"])[cols].unstack("weight").swaplevel(axis=1)
        return f[[(w, c) for w in ["EW", "VW"] for c in cols]].rename(index=LABELS)
    return per_model(R, "3_factors", one)


# ---- appendix: Sections 4 and 5 ------------------------------------------------------------------------------------------------
def persistence(R, m, f):
    """the level and the deviation of forecast f (Table 7's panels, or "strategy": DKPT's strategy of Table 8) as signals by horizon h:
    (their rank IC, their joint Fama-MacBeth coefficient after the JKP residualization), each long: signal, h, value, t"""
    if f == "strategy":
        s = table(R, "4_didisheim/attribution", "signals", m).assign(h=lambda d: d.tau + 1)
        ic, fm = s[s.stat == "IC_%"], s[s.stat == "FM_%_per_sd"]
    elif f == "daily":
        T = table(R, "4_horizon", "term_structure", m)
        ic = T[T.part.isin(TIMING_PARTS) & (T.h >= 1)].drop(columns="t").rename(columns={"part": "signal", "IC_%": "value", "t_IC": "t"})
        fm = table(R, "4_horizon", "jkp_fm", m).rename(columns={"x": "signal", "coef_bp_per_sd": "value"})
    else:
        p, j = table(R, "4_monthly_ridge", "performance", m), table(R, "4_monthly_ridge", "jkp_fm", m)
        ic = p[(p.predictor == f) & p.signal.isin(TIMING_PARTS)].assign(h=p.tau + 1).rename(columns={"IC_%": "value", "t_IC": "t"})
        fm = j[j.predictor == f].assign(h=j.tau + 1).rename(columns={"x": "signal", "coef_pct_per_sd": "value"})
    return ic[["signal", "h", "value", "t"]], fm[["signal", "h", "value", "t"]]


def plot_persistence(R):
    """Figure A4, one file per forecast of Table 7 (a daily, b its monthly average, c monthly-trained), and Figure A5 for DKPT's
    strategy, one column per model: top the rank IC of the level and the deviation as signals, bottom their joint Fama-MacBeth
    coefficients after both are residualized on the replication's JKP characteristics, by horizon h (95% bands); each row shares its y axis"""
    figs = []
    for name, f in [*((f"figA_persistence_{letter}", f) for letter, f in zip("abc", FORECASTS)), ("figA_persistence_strategy", "strategy")]:
        daily, ms = f == "daily", models(R, "4_didisheim/attribution" if f == "strategy" else "4_monthly_ridge")
        fig, axes = plt.subplots(2, len(ms), figsize=(11, 7.2), squeeze=False, sharey="row")
        for j, (col, (m, lab)) in enumerate(zip(axes.T, ms.items())):
            for ax, d, title, ylab in zip(col, persistence(R, m, f), ["rank IC", "Fama-MacBeth after the JKP characteristics"],
                                          ["IC (%)", "bp per sd" if daily else "% per sd"]):
                for k, color in [("lev", CELL_COLORS["yF"]), ("dev", CELL_COLORS["cL"])]:
                    e = d[d.signal == k].sort_values("h")
                    se = (e.value / e.t.replace(0, np.nan)).abs().fillna(0)
                    ax.plot(e.h, e.value, marker="o", ms=3, color=color, label=LEGEND[k])
                    ax.fill_between(e.h, e.value - Z95 * se, e.value + Z95 * se, color=color, alpha=0.12)
                ax.axhline(0, color="grey", lw=0.5)
                ax.set_title(f"{lab}: {title}", fontsize=9, loc="left")
                ax.set_xlabel(HORIZON_AXIS[f], fontsize=8)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                ax.tick_params(labelsize=8)
                _h_axis(ax, d.h)
        _legend(axes[0, 0])
        figs.append(show(fig, R, name))
    return figs


STANDALONE_NAMES = {**EMBEDDING_TEX, STANDALONE["embedding"]: "Embedding $E$", STANDALONE["lagged"]: f"Level {LEVEL_LAG} months earlier, $E_{{L,\\,\\tau-{LEVEL_LAG}}}$",
                    STANDALONE["lagged_F"]: f"Firm identity {LEVEL_LAG} months earlier, $F_{{\\tau-{LEVEL_LAG}}}$",
                    STANDALONE["template"]: "Category templates, $E^{\\mathrm{tpl}}$", STANDALONE["groups"]: "Embedding $E$, event-group control"}


def standalone(R, col="Sharpe", encoders=MODELS, part="4_didisheim/attribution"):
    """the strategy on each input (MSRR weights re-optimized at every horizon): Sharpe ratio (col "Sharpe_<segment>": on that
    segment's signal months, blank below SEGMENT_MIN_MONTHS months) by object (rows) and horizon h = tau + 1; part
    "4_didisheim/industry": with the industry-group indicators in the residualization"""
    def one(m):
        t = table(R, part, "standalone", m).assign(h=lambda d: d.tau + 1)
        if col != "Sharpe":
            t[col] = t[col].where(t[col.replace("Sharpe", "months")] >= SEGMENT_MIN_MONTHS)
        return t.pivot(index="object", columns="h", values=col)
    return per_model(R, part, one, encoders)


REPRESENTATION_OUTCOMES = {"log_pb": "$\\log(P/B)$", "log_pe": "$\\log(P/E)$", "ltg": "LTG"}
REPRESENTATION_INPUTS = {"F": "$F$", "average": "Average embedding", "controls": "Controls", "F + controls": "$F$ + controls",
                         "average + controls": "Average embedding + controls"}    # Tables C6 and B6
VALUATION_INPUTS = {"controls": "Controls", "average": "$E^{\\mathrm{avg}}$", "category mix": "$C^{\\mathrm{mix}}$", "F": "$F$",
                    "category mix + F": "$C^{\\mathrm{mix}}+F$", "average + controls": "Controls + $E^{\\mathrm{avg}}$",
                    "category mix + F + controls": "Controls + $C^{\\mathrm{mix}}+F$"}          # Table 4, in row order


def representation(R, sample="split", first=FIRST_YEAR, encoders=MODELS, inputs=REPRESENTATION_INPUTS):
    """Table 4 (inputs=VALUATION_INPUTS), C6 and B6: R2 (%) of the firms' valuations and growth forecasts on each input, averaged over
    the held-out folds and the windows of the Decembers of the signal years; sample "split" (Sarkar's split sample) or "previous window"
    (fitted on the previous window); first: the windows' first year, or {model: first year} (Table C6: the year after each cutoff)"""
    def one(m):
        r = table(R, "2_representation", "r2", m)
        r = r[r.first_year == (first[m] if isinstance(first, dict) else first)]
        t = 100 * r[r["sample"] == sample].groupby(["inputs", "outcome"]).R2.mean().unstack()
        return t.loc[list(inputs), list(REPRESENTATION_OUTCOMES)].rename(index=inputs, columns=REPRESENTATION_OUTCOMES)
    return per_model(R, "2_representation", one, encoders)


NETWORK_CHARACTERISTICS = {"ebitda_sale": ("Firm characteristics", "OI/Sales"), "ebitda_at": ("Firm characteristics", "OI/Assets"),
                           "sale_gr1": ("Firm characteristics", "$\\Delta$Sales"), "beta_60m": ("Firm characteristics", "Market $\\beta$"),
                           "asset_beta": ("Firm characteristics", "Asset $\\beta$"),
                           **{k: ("Valuations and expectations", v) for k, v in REPRESENTATION_OUTCOMES.items()}}
NETWORK_TABLE = ["ebitda_sale", "ebitda_at", "sale_gr1", "beta_60m", "asset_beta"]   # Tables 5, C7, B7 (the valuations only in Table 10)
NETWORK_ROWS = {"SIC-3": "SIC-3", "NAICS-4": "NAICS-4", "GICS-6": "GICS-6", "average": "$E^{\\mathrm{avg}}$", "F": "$F$",
                "category mix": "$C^{\\mathrm{mix}}$", "average, average": "$E^{\\mathrm{avg}}$, $E^{\\mathrm{avg}}$", "F, category mix": "$F$, $C^{\\mathrm{mix}}$"}
OVERLAP = {k: ("Overlap", k) for k in ["SIC-3", "NAICS-4", "GICS-6"]}


def network_run(encoders, first=FIRST_YEAR):
    """the folder of a 2_networks.py run under Results/paper"""
    return f"2_networks/{'+'.join(encoders)}/{first}"


def networks(R, encoders, first=FIRST_YEAR, display=False):
    """Tables 5, C7, B7 (Table 10): adjusted R2 (%) of each characteristic on each row's industry controls and the row's overlap with each
    industry classification (%) (display=True: only NETWORK_TABLE's characteristics); blank where the run has no result:
    {"": the industry classifications, encoder label: that encoder's rows} of the run of `encoders` (a dict) from `first`"""
    run = network_run(encoders, first)
    r, o = table(R, run, "r2"), table(R, run, "overlap")
    r["encoder"], o["encoder"] = r.encoder.fillna(""), o.encoder.fillna("")
    codes, rows = ["SIC-3", "NAICS-4", "GICS-6"], ["average", "F", "category mix", "average, average", "F, category mix"]
    t = 100 * r.pivot_table(index=["encoder", "row"], columns="characteristic", values="adj_R2")
    t = t.reindex(index=pd.MultiIndex.from_tuples([("", k) for k in codes] + [(m, k) for m in encoders for k in rows], names=["encoder", "row"]),
                  columns=list(NETWORK_CHARACTERISTICS))
    t.columns = pd.MultiIndex.from_tuples(list(NETWORK_CHARACTERISTICS.values()))
    for k, col in OVERLAP.items():
        t[col] = 100 * o.groupby(["encoder", "row"])["overlap_with_" + k].mean()
        t.loc[("", k), col] = np.nan                                        # a classification with itself
    if display:
        t = t[[NETWORK_CHARACTERISTICS[k] for k in NETWORK_TABLE] + list(OVERLAP.values())]
    out = {"": t.loc[""].loc[codes].rename(index=NETWORK_ROWS)}
    for m, lab in encoders.items():
        out[lab] = t.loc[m].loc[rows].rename(index=NETWORK_ROWS)
    return out


def encoder_summary(R, seg=None):
    """Table 10: the main results of every encoder, one column each, on the main tables' signal years or (seg) on the robustness
    encoder's segment (each encoder on its segment with those signal years): the daily forecast (Tables 2, 6), the monthly strategy at
    h = 1 and 12 (Tables 8, A16) and the firm component (Table 4, on the windows from the segment's first year); blank where a segment's
    horizon has fewer than SEGMENT_MIN_MONTHS months, the level shares also where the strategy's mean return has |t| < 2 (as Table 8)"""
    (robust,) = ROBUST_MODELS
    order = ROBUST_MODELS | MODELS                                         # by the end of the pretraining text
    cols = {}
    for m, lab in order.items():
        s = seg and next(k for k, v in SEGMENTS[m].items() if v == SEGMENTS[robust][seg])   # the encoder's segment with these signal years
        one = {m: lab}
        sp = table_split(R, s, one).loc[lab]
        fm = table(R, "1_category", f"fm_{s}" if s else "fm")
        b = fm[(fm.spec == f"Category mean (8-yr) + {FORECAST_LABEL[m]}") & (fm.regressor == FORECAST_LABEL[m])].iloc[0]
        A = attribution(R, m, name=f"attribution_{s}" if s else "attribution").set_index(["h", "part"])
        A = A.reindex(pd.MultiIndex.from_product([[1, 12], ["total", "lev"]]))
        day, mon = "CKX daily forecast", "DKPT monthly MSRR strategy"
        pct = lambda v: f"{v:.1f}" if pd.notna(v) else v                    # shares with one decimal, as in Tables 6 and 8
        lev = lambda h: pct(100 * A.loc[(h, "lev"), "share"]) if abs(A.loc[(h, "total"), "t"]) >= 2 else np.nan   # blank where the total is indistinguishable from zero
        c = {(day, "Rank IC (%)"): sp.loc[LABELS["yhat"], "IC_%"], (day, "EW SR"): sp.loc[LABELS["yhat"], "EW_Sharpe"],
             (day, "VW SR"): sp.loc[LABELS["yhat"], "VW_Sharpe"], (day, "Coefficient with the category mean"): b.coef, (day, ""): f"({b.t_NW:.2f})",
             **{(day, f"Covariance share, {LABELS[k]} (%)"): pct(100 * sp.loc[LABELS[k], "A_j"]) for k in COMP},
             (mon, "SR, $h=1$"): A.loc[(1, "total"), "Sharpe"], (mon, "Share of the level $E_L$, $h=1$ (%)"): lev(1),
             (mon, "SR, $h=12$"): A.loc[(12, "total"), "Sharpe"], (mon, "Share of the level $E_L$, $h=12$ (%)"): lev(12),
             (mon, f"SR, level {LEVEL_LAG} months earlier, $h=1$"): standalone(R, f"Sharpe_{s}" if s else "Sharpe", one).reindex(columns=[1]).loc[(lab, STANDALONE["lagged"]), 1]}
        r = representation(R, first=SEGMENTS[robust][seg][0] if seg else FIRST_YEAR, encoders=one).loc[lab, REPRESENTATION_OUTCOMES["log_pb"]]
        c[("Firm identity $F$", "$R^2$ of $\\log(P/B)$ (%)")] = r[REPRESENTATION_INPUTS["F"]]
        c[("Firm identity $F$", "Increment over the controls (pp)")] = r[REPRESENTATION_INPUTS["F + controls"]] - r[REPRESENTATION_INPUTS["controls"]]
        first = SEGMENTS[robust][seg][0] if seg else FIRST_YEAR               # the F network of the encoder's network run from that first year
        run = list(MODELS) if m in MODELS and first == FIRST_YEAR else [m]   # Table 5's joint run of the main encoders, otherwise the encoder alone
        net = networks(R, {e: ENCODERS[e] for e in run}, first)
        pb = REPRESENTATION_OUTCOMES["log_pb"]
        f_net = net[lab].loc[NETWORK_ROWS["F"], ("Valuations and expectations", pb)]
        c[("Firm identity $F$", "$F$ network: adjusted $R^2$ of $\\log(P/B)$ (%)")] = f_net
        c[("Firm identity $F$", "$F$ network minus SIC-3 (pp)")] = f_net - net[""].loc["SIC-3", ("Valuations and expectations", pb)]
        cols[lab] = pd.Series(c)
    return pd.DataFrame(cols).reindex(columns=list(order.values()))


CUTOFF_WINDOW = 252                                                     # Figure 6's rolling window (trading days)
CUTOFF_NW_LAGS = 40                                                     # Newey-West lags of the window means' standard errors
CUTOFF_WINDOW_MONTHLY = 36                                              # Figure A6's rolling window (signal months)
CUTOFF_COLORS = {m: f"C{i}" for i, m in enumerate([*reversed(MODELS), *ROBUST_MODELS])}   # matplotlib's default colors in legend order, Llama-2 first


def _nw_se(x, lags):
    """Newey-West (Bartlett) standard error of the mean of x"""
    x = x - x.mean()
    n = len(x)
    g = [x[k:] @ x[:n - k] / n for k in range(lags + 1)]
    return np.sqrt((g[0] + 2 * sum((1 - k / (lags + 1)) * g[k] for k in range(1, lags + 1))) / n)


def plot_cutoff(R):
    """Figure 6: the CUTOFF_WINDOW-day rolling mean of each encoder's daily next-day rank IC (4_horizon ic_daily.csv.gz, h = 1) on the
    trading days every encoder shares, the level (top) and the deviation (bottom), with 95% Newey-West bands for the main encoders and
    their pretraining cutoffs marked"""
    enc = {**MODELS, **ROBUST_MODELS}
    ic = {m: read(R / "4_horizon" / m / "ic_daily.csv.gz", parse_dates=["day0"]).query("h == 1").set_index("day0") for m in enc}
    return show(rolling_ic_figure(ic, CUTOFF_WINDOW, CUTOFF_NW_LAGS, [PARTS_TEX["historical level"], PARTS_TEX["deviation from the level"]]), R, "fig6_cutoff")


def plot_cutoff_monthly(R):
    """Figure A6, the monthly counterpart of Figure 6: the CUTOFF_WINDOW_MONTHLY-month rolling mean of each encoder's monthly rank IC of
    the DKPT level and deviation signals with the next month's return (4_didisheim ic_monthly.csv, tau = 0), dated at the end of the
    signal month, with NW_LAGS_MONTHLY Newey-West lags"""
    enc = {**MODELS, **ROBUST_MODELS}
    ic = {}
    for m in enc:
        d = read(R / "4_didisheim/attribution" / m / "ic_monthly.csv").query("tau == 0")
        ic[m] = d.set_index(pd.PeriodIndex(d.month, freq="M").to_timestamp(how="end").normalize())
    return show(rolling_ic_figure(ic, CUTOFF_WINDOW_MONTHLY, NW_LAGS_MONTHLY, [LEGEND[k] for k in TIMING_PARTS]), R, "figA_cutoff_monthly")


CUTOFF_BLOCK, CUTOFF_DRAWS, CUTOFF_SEED = 40, 2000, 20261109           # Table 11's bootstrap: block, draws and seed as Framework/calibration.py; its
                                                                        # days are those RoBERTa and Llama-2 share, the framework's also need ChronoGPT's
CUTOFF_GROUPS = {"all": ("", "All"), **{g: ("Size", SIZE_NAMES[g]) for g in SIZE_GROUPS},
                 **{g: ("News intensity", g.capitalize()) for g in COVERAGE_GROUPS}}   # Table 11's columns


def ic_ratio(A, B, post):
    """A's mean IC over B's on the days after the split, relative to the same ratio before (Framework/calibration.py ic_ratio)"""
    return (np.nanmean(A[post]) / np.nanmean(B[post])) / (np.nanmean(A[~post]) / np.nanmean(B[~post]))


def block_draws(A, B, post, L, rng, n):
    """moving-block bootstrap of ic_ratio with blocks of L trading days, drawn separately within the days before and after the split
    (each period keeps its length; Framework/calibration.py block_draws)"""
    per = [np.flatnonzero(~post), np.flatnonzero(post)]
    out = []
    for _ in range(n):
        ii = []
        for d in per:
            s = rng.integers(0, len(d) - L + 1, size=len(d) // L + 1)
            ii.append(d[np.concatenate([np.arange(k, k + L) for k in s])[:len(d)]])
        ii = np.concatenate(ii)
        out.append(ic_ratio(A[ii], B[ii], post[ii]))
    return np.array(out)


def table_cutoff_groups(R):
    """Table 11: the main encoders' mean next-day rank ICs (%) of the level and the deviation before and from the January after Llama-2's
    pretraining cutoff, for all firm-days (4_horizon ic_daily.csv.gz, h = 1) and within each size group and news-intensity tercile
    (ic_daily_groups.csv.gz), on the trading days both encoders share; and Llama-2's change relative to RoBERTa (ic_ratio)
    with its 90% moving-block bootstrap interval, blank when Llama-2's IC before or RoBERTa's after has a Newey-West |t| below 2 (a
    denominator indistinguishable from zero)"""
    RB, LM = MODELS
    split = pd.Timestamp(f"{POST_YEAR[LM]}-01-01")
    ic = {}
    for m in MODELS:
        a = read(R / "4_horizon" / m / "ic_daily.csv.gz", parse_dates=["day0"]).query("h == 1").assign(group="all")
        g = read(R / "4_horizon" / m / "ic_daily_groups.csv.gz", parse_dates=["day0"])
        ic[m] = pd.concat([a[["day0", "group", *TIMING]], g], ignore_index=True)
    rows = []
    for k in TIMING:
        for grp in CUTOFF_GROUPS:
            S = {m: d[d.group == grp].set_index("day0")[k].dropna() for m, d in ic.items()}
            idx = S[LM].index.intersection(S[RB].index)
            A, B, post = S[LM].loc[idx].values, S[RB].loc[idx].values, np.asarray(idx >= split)
            t = lambda x: x.mean() / _nw_se(x, NW_LAGS_DAILY)
            r = {"part": k, "group": grp, "days_pre": int((~post).sum()), "days_post": int(post.sum()),
                 **{f"{m}_{p}": 100 * x[q].mean() for m, x in ((LM, A), (RB, B)) for p, q in (("pre", ~post), ("post", post))},
                 "t_lm_pre": t(A[~post]), "t_rb_post": t(B[post]), "ratio": np.nan, "lo": np.nan, "hi": np.nan}
            if abs(r["t_lm_pre"]) >= 2 and abs(r["t_rb_post"]) >= 2:
                o = block_draws(A, B, post, CUTOFF_BLOCK, np.random.default_rng(CUTOFF_SEED), CUTOFF_DRAWS)
                r.update(ratio=ic_ratio(A, B, post), lo=np.nanpercentile(o, 5), hi=np.nanpercentile(o, 95))
            rows.append(r)
    return pd.DataFrame(rows)


def cutoff_control_ratio(R, control=None):
    """Section 7.2's footnote: Table 11's relative IC ratio over all firm-days (ic_daily.csv.gz, h = 1) with the robustness encoder
    (control None: ChronoGPT) in place of RoBERTa, for the level and the deviation, on the trading days Llama-2 and the control share;
    with Table 11's 90% moving-block bootstrap interval (same block, draws and seed) and the Newey-West t of Llama-2's IC before and
    of the control's IC from the split, computed even when the control's t is below 2"""
    (control,) = ROBUST_MODELS if control is None else (control,)
    _, LM = MODELS                                                        # Llama-2, as in table_cutoff_groups
    split = pd.Timestamp(f"{POST_YEAR[LM]}-01-01")
    ic = {m: read(R / "4_horizon" / m / "ic_daily.csv.gz", parse_dates=["day0"]).query("h == 1").set_index("day0") for m in (LM, control)}
    t = lambda x: x.mean() / _nw_se(x, NW_LAGS_DAILY)
    rows = []
    for k in TIMING:
        S = {m: d[k].dropna() for m, d in ic.items()}
        idx = S[LM].index.intersection(S[control].index)
        A, B, post = S[LM].loc[idx].values, S[control].loc[idx].values, np.asarray(idx >= split)
        o = block_draws(A, B, post, CUTOFF_BLOCK, np.random.default_rng(CUTOFF_SEED), CUTOFF_DRAWS)
        rows.append({"part": k, "control": control, "days_pre": int((~post).sum()), "days_post": int(post.sum()),
                     "ratio": ic_ratio(A, B, post), "lo": np.nanpercentile(o, 5), "hi": np.nanpercentile(o, 95),
                     "t_lm_pre": t(A[~post]), "t_control_post": t(B[post])})
    return pd.DataFrame(rows)


def rolling_ic_figure(ic, W, lags, names):
    """Figures 6 and A13: ic = {encoder: rank ICs indexed by date, columns TIMING_PARTS}; on the dates every encoder shares, the W-period
    rolling mean (%) of the level (top) and the deviation (bottom), y labels `names`, 95% Newey-West bands (lags within each window) for
    the main encoders, pretraining cutoffs marked, Llama-2 first in the legend"""
    enc = list(ic)
    days = pd.DatetimeIndex(sorted(set.intersection(*(set(d.index) for d in ic.values()))))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    for ax, part, name in zip(axes, TIMING_PARTS, names):
        for m in enc:
            x = 100 * ic[m].loc[days, part].to_numpy()
            mu = pd.Series(x, index=days).rolling(W).mean()
            if m in MODELS:
                se = np.r_[np.full(W - 1, np.nan), [_nw_se(x[e - W:e], lags) for e in range(W, len(x) + 1)]]
                ax.fill_between(days, mu - 1.96 * se, mu + 1.96 * se, color=CUTOFF_COLORS[m], alpha=0.15, lw=0)
            ax.plot(days, mu, color=CUTOFF_COLORS[m], lw=1.2, label=ROBUST_MODELS.get(m, SHORT[m]))
        ax.axhline(0, color="grey", lw=0.5)
        for m, c in PRETRAIN_CUTOFF.items():
            ax.axvline(pd.Period(c, "M").end_time, color="0.6", lw=0.8, ls="--")
        ax.set_ylabel(f"{name}: rank IC (%)", fontsize=8)
        ax.tick_params(labelsize=8)
    for m, c in PRETRAIN_CUTOFF.items():
        axes[0].text(pd.Period(c, "M").end_time, 0.97, f"{SHORT[m]} cutoff", transform=axes[0].get_xaxis_transform(), fontsize=7,
                     ha="right", va="top", rotation=90, color="0.4")
    h, l = axes[0].get_legend_handles_labels()
    order = [list(enc).index(m) for m in [*reversed(MODELS), *ROBUST_MODELS]]   # Llama-2 first
    axes[0].legend([h[i] for i in order], [l[i] for i in order], fontsize=7, frameon=False, loc="upper right")
    return fig



def plot_cutoff_returns(R):
    """Figure 5: cumulative monthly returns of each encoder's MSRR strategy and of its level and deviation (4_didisheim monthly.csv,
    undelayed fixed weights, tau = 0), dated at the end of the signal month"""
    ret = {}
    for m in {**MODELS, **ROBUST_MODELS}:
        d = read(R / "4_didisheim/attribution" / m / "monthly.csv").query("weights == 'fixed' and tau == 0").pivot(index="month", columns="part", values="ret")
        ret[m] = d.set_index(pd.PeriodIndex(d.index, freq="M").to_timestamp(how="end").normalize())
    return show(cumulative_figure(ret, ["MSRR strategy", LEGEND["lev"], LEGEND["dev"]], 12), R, "fig5_cutoff_returns")


def plot_cutoff_returns_daily(R):
    """Figure A7, the daily counterpart of Figure 5: cumulative returns of the long-short portfolio weighted by the within-day demeaned
    forecast (its return is the forecast's within-day covariance with the next open-to-open return, 4_horizon cov_daily.csv.gz, h = 1)
    and of its level and deviation"""
    ret = {m: read(R / "4_horizon" / m / "cov_daily.csv.gz", parse_dates=["day0"]).query("h == 1").set_index("day0").rename(columns={"yhat": "total"})
           for m in {**MODELS, **ROBUST_MODELS}}
    return show(cumulative_figure(ret, [PARTS_TEX["forecast"], PARTS_TEX["historical level"], PARTS_TEX["deviation from the level"]], 252), R,
                "figA_cutoff_returns_daily")


def cumulative_figure(ret, names, per_year):
    """Figures 5 and A7: ret = {encoder: returns indexed by date, columns total, lev, dev}; on the dates every encoder shares, the
    cumulative sum (%) of the total (top), the level and the deviation, the total scaled to 10% annual volatility and its components by
    the same factor, y labels `names`, pretraining cutoffs marked, Llama-2 first in the legend"""
    enc = list(ret)
    days = pd.DatetimeIndex(sorted(set.intersection(*(set(d.index) for d in ret.values()))))
    parts = ["total", *TIMING_PARTS]
    fig, axes = plt.subplots(len(parts), 1, figsize=(8, 6.4), sharex=True)
    for m in enc:
        d = ret[m].loc[days, parts]
        d = d * 0.10 / (d["total"].std() * np.sqrt(per_year))
        for ax, part in zip(axes, parts):
            ax.plot(days, 100 * d[part].cumsum(), color=CUTOFF_COLORS[m], lw=1.2, label=ROBUST_MODELS.get(m, SHORT[m]))
    for ax, name in zip(axes, names):
        ax.axhline(0, color="grey", lw=0.5)
        for c in PRETRAIN_CUTOFF.values():
            ax.axvline(pd.Period(c, "M").end_time, color="0.6", lw=0.8, ls="--")
        ax.set_ylabel(f"{name}\ncumulative return (%)", fontsize=8)
        ax.tick_params(labelsize=8)
    for m, c in PRETRAIN_CUTOFF.items():                                 # above the top panel, clear of the curves
        axes[0].text(pd.Period(c, "M").end_time, 1.02, f"{SHORT[m]}\ncutoff", transform=axes[0].get_xaxis_transform(), fontsize=7,
                     ha="center", va="bottom", color="0.4")
    h, l = axes[0].get_legend_handles_labels()
    order = [enc.index(m) for m in [*reversed(MODELS), *ROBUST_MODELS]]   # Llama-2 first
    axes[0].legend([h[i] for i in order], [l[i] for i in order], fontsize=7, frameon=False, loc="upper left")
    fig.align_ylabels(axes)
    return fig


SPECTRUM_LINES = {"All": "k", "C": CELL_COLORS["cL"], "F": CELL_COLORS["yF"], "I": CELL_COLORS["uL"]}


def plot_spectrum(R):
    """Figure A3: by principal direction k (log scale), the cumulative share along directions 1..k, top: of the embedding's variance (All)
    and of each content component's, bottom: of the forecast's test-year covariance with the return and of its components' contributions"""
    ms = models(R, "2_spectrum")
    fig, axes = plt.subplots(2, len(ms), figsize=(6.5 * len(ms), 6.4), squeeze=False)
    for j, (m, lab) in enumerate(ms.items()):
        d = table(R, "2_spectrum", "by_k", m)
        for i, title in enumerate(["cumulative share of each component's variance (%)", "cumulative share of the forecast's covariance (%)"]):
            ax = axes[i, j]
            for part, color in SPECTRUM_LINES.items():
                y = d[f"cum_{'lambda' if part == 'All' else 'var_' + part}"] if i == 0 else d[f"cov_{part}"].cumsum()
                ax.plot(d.k, 100 * y, color=color, lw=1.2, label=part if part == "All" else f"${part}$")
            for lo, _ in BANDS[1:]:
                ax.axvline(lo - 0.5, color="grey", lw=0.4, ls=":")
            ax.set_xscale("log")
            ax.set_xlim(1, d.k.max())
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(f"{lab}: {title}", fontsize=9, loc="left")
            ax.tick_params(labelsize=8)
            if i == 1:
                ax.set_xlabel("principal direction $k$ (by decreasing variance)", fontsize=8)
            if j == 0:
                ax.legend(fontsize=7, frameon=False)
    return show(fig, R, "figA_spectrum")


FIG4_UPPER = [STANDALONE["embedding"], LABELS["lev"], LABELS["dev"], STANDALONE["lagged"], STANDALONE["lagged_F"]]   # JKP residualization
FIG4_LOWER = [STANDALONE["embedding"], LABELS["lev"], STANDALONE["lagged_F"]]   # JKP residualization and with industry indicators as well
FIG4_STYLE = {STANDALONE["embedding"]: ("k", 1.5), LABELS["lev"]: (CELL_COLORS["yF"], 1.5), LABELS["dev"]: (CELL_COLORS["cL"], 1.5),
              STANDALONE["lagged"]: ("#d62728", 2.5), STANDALONE["lagged_F"]: ("#9467bd", 2.5)}   # (color, width): one color per input in both rows
INDUSTRY_CONTROL = ", industry control"                                  # the legend suffix of a 4_didisheim --industry line


def plot_standalone(R):
    """Figure 4: Table A16 at every horizon, one column per model with one vertical axis: the inputs of FIG4_UPPER residualized on the
    JKP characteristics (upper panel), and those of FIG4_LOWER so (solid) and with industry-group indicators as well (dashed,
    4_didisheim --industry; lower panel); 95% bands on the solid lines (fig4_inputs) or none (fig4_inputs_noci, for slides)"""
    ms = models(R, "4_didisheim/attribution")
    models(R, "4_didisheim/industry")
    titles = ["alternative inputs", "with and without industry control"]
    figs = []
    for bands, name in [(True, "fig4_inputs"), (False, "fig4_inputs_noci")]:
        fig, axes = plt.subplots(2, len(ms), figsize=(6.5 * len(ms), 9.0), squeeze=False, sharey="col", gridspec_kw={"hspace": 0.4})
        for j, (m, lab) in enumerate(ms.items()):
            jkp = dict(tuple(table(R, "4_didisheim/attribution", "standalone", m).groupby("object", sort=False)))
            ind = dict(tuple(table(R, "4_didisheim/industry", "standalone", m).groupby("object", sort=False)))
            lines = [(0, obj, jkp[obj], "-", STANDALONE_NAMES[obj]) for obj in FIG4_UPPER]
            for obj in FIG4_LOWER:
                lines += [(1, obj, jkp[obj], "-", STANDALONE_NAMES[obj]), (1, obj, ind[obj], "--", STANDALONE_NAMES[obj] + INDUSTRY_CONTROL)]
            for i, obj, d, ls, label in lines:
                ax, (color, lw) = axes[i, j], FIG4_STYLE[obj]
                lw = min(lw, 1.5) if ls == "--" else lw                     # a thick dashed line reads as solid
                h = d.tau + 1
                ax.plot(h, d.Sharpe, marker="o", ms=3, color=color, ls=ls, lw=lw, label=label)
                if bands and ls == "-":
                    se = (d.Sharpe / d.t).abs()                                # the mean's Newey-West band in Sharpe units
                    ax.fill_between(h, d.Sharpe - Z95 * se, d.Sharpe + Z95 * se, color=color, alpha=0.08)
            for i, title in enumerate(titles):
                ax = axes[i, j]
                ax.axhline(0, color="grey", lw=0.5)
                _h_axis(ax, jkp[STANDALONE["embedding"]].tau + 1)
                ax.set_xlabel(HORIZON_AXIS["bridge"], fontsize=8)
                ax.set_title(f"{lab}: {title}", fontsize=9, loc="left")
                ax.set_ylabel("Sharpe ratio", fontsize=8)
        _row_legend(fig, axes[0], len(FIG4_UPPER))
        _row_legend(fig, axes[1], len(FIG4_LOWER))                             # three columns: each input's solid and dashed line together
        figs.append(show(fig, R, name))
    return figs


JOINT_FM_TEX = {LABELS["dev"]: "Deviation $E_{D,\\,\\tau-k}$", LABELS["lev"]: "Level $E_{L,\\,\\tau}$"}


JOINT_FM_LAGS = [h - 1 for h in MONTHLY_H if h > 1]               # k = h - 1: the return month is the h-th month after the deviation month, Table 8's horizons


def joint_fm(R):
    """deviation of month t - k and level of month t: one column per lag k of JOINT_FM_LAGS (every k is stored), % per sd with t below"""
    return per_model(R, "4_didisheim/attribution", lambda m: _regression(table(R, "4_didisheim/attribution", "joint_fm", m).loc[lambda d: d.lag.isin(JOINT_FM_LAGS)], "lag", "x", "coef_pct_per_sd", "t", digits=3))


def plot_old_news(R, encoders=MODELS, name="figA_old_news"):
    """Figure A12 (with encoders = {**ROBUST_MODELS, **MODELS} and another name: the same with ChronoGPT, not in the paper): the deviation's coefficient in Table 9's regressions at every horizon
    h = k + 1 (the return month's distance from the deviation month), the encoders in one panel, 95% bands; h = 1 is the same
    regression on the deviation of month t (signals.csv, the Fama-MacBeth of figA_persistence_strategy at tau 0); filled markers at
    multiples of three months"""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, lab in models(R, "4_didisheim/attribution", encoders).items():
        j = table(R, "4_didisheim/attribution", "joint_fm", m).query("x == 'dev'").assign(h=lambda x: x.lag + 1)
        s = table(R, "4_didisheim/attribution", "signals", m).query("stat == 'FM_%_per_sd' and signal == 'dev' and tau == 0")
        d = pd.concat([s.assign(h=1, coef_pct_per_sd=s.value), j])[["h", "coef_pct_per_sd", "t"]].sort_values("h")
        color = CUTOFF_COLORS[m]
        se = (d.coef_pct_per_sd / d.t.replace(0, np.nan)).abs().fillna(0)
        ax.fill_between(d.h, d.coef_pct_per_sd - Z95 * se, d.coef_pct_per_sd + Z95 * se, color=color, alpha=0.12, lw=0)
        ax.plot(d.h, d.coef_pct_per_sd, color=color, lw=1.3, label=lab)
        q = d.h % 3 == 0
        ax.plot(d.h[q], d.coef_pct_per_sd[q], ls="none", marker="o", ms=4, color=color)
        ax.plot(d.h[~q], d.coef_pct_per_sd[~q], ls="none", marker="o", ms=4, mfc="white", color=color)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(HORIZON_AXIS["bridge"], fontsize=8)
    ax.set_ylabel("% per sd", fontsize=8)
    ax.tick_params(labelsize=8)
    _h_axis(ax, range(1, 37))
    ax.set_xlim(0.5, 36.5)
    ax.legend(fontsize=8, frameon=False)
    return show(fig, R, name)


def timing(R):
    """the replication's MSRR Sharpe ratio by horizon h = tau + 1, firm-months keyed by the month of arrival (the replication) and by
    the trade month"""
    def one(m):
        f = {ARRIVAL_MONTH: DIDISHEIM_RESULTS / m / "F10_lags.csv", TRADE_MONTH: R / "4_didisheim/timing" / m / "F10_lags.csv"}
        return pd.DataFrame({k: read(v).assign(h=lambda d: d.tau + 1).set_index("h").Sharpe for k, v in f.items()}).T
    return per_model(R, "4_didisheim/attribution", one)
