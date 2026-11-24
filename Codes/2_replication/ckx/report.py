"""Tables and figures of the CKX replication notebook, computed from Results/replication/ckx. Each function takes the results
folder R and returns a DataFrame (or draws a figure), so the notebook cells are one-line calls that can be checked statically."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.runinfo import require_same_sample
from common.paper_numbers import CKX, CKX_EWCT, CKX_LLAMA3, CKX_NET

NAMES = {"Llama-2-13b-hf": "LLaMA2", "roberta-large": "RoBERTa"}            # the paper's names and order
PAPER_END = "2019-06-30"


def show(R, name):
    """save the current figure to R/figures/<name>.pdf, then display it"""
    d = Path(R) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(d / f"{name}.pdf")
    plt.show()


def models(R):
    """the models with results, each manifest checked against the current sample and ridge specification"""
    out = {m: l for m, l in NAMES.items() if (R / m / "paper" / "run.json").exists()}
    for m in out:
        require_same_sample(R / m / "paper" / "run.json", f"{m} results", needs=("ridge",))
    return out


def table(R, m, name, per="paper", **kw):
    f = R / m / per / f"{name}.csv"
    if not f.exists():
        return None
    require_same_sample(f.parent / "run.json", f"{m}/{per} results", needs=("ridge",))
    return pd.read_csv(f, **kw)


def runs(R):
    rows = []
    for m in models(R):
        for per in ["paper", "extended"]:
            f = R / m / per / "run.json"
            if f.exists():
                j = json.loads(f.read_text())
                rows.append({"model": m, "period": per, "finished": j["finished"]})
    return pd.DataFrame(rows).set_index(["model", "period"])


def summary(R):
    """paper vs ours for every headline statistic (paper period); LLaMA3-only paper numbers are in a separate column"""
    rows = []
    for m, lab in models(R).items():
        c = table(R, m, "T03_correlation", index_col=0)["value"]
        q5 = table(R, m, "T05_quintile", header=[0, 1], index_col=0)
        q10 = table(R, m, "T06_decile", header=[0, 1], index_col=0)
        al = table(R, m, "T20_21_alphas", index_col=0)
        ew = table(R, m, "T08_ewct", index_col=0)
        dl = table(R, m, "T18_delay").set_index(["delay", "weight"])
        cnt = table(R, m, "counts", index_col=0)
        ours = {"Pearson corr %": c["Mean (Pearson) %"], "Pearson corr, over NYSE median %": c["Over NYSE median %"],
                "EW quintile L-S Sharpe": q5.loc["SR", ("EW", "L-S")], "EW quintile L-S return %/yr": q5.loc["Ret_%/yr", ("EW", "L-S")],
                "VW quintile L-S Sharpe": q5.loc["SR", ("VW", "L-S")], "VW quintile L-S return %/yr": q5.loc["Ret_%/yr", ("VW", "L-S")],
                "EW decile L-S Sharpe": q10.loc["SR", ("EW", "L-S")], "VW decile L-S Sharpe": q10.loc["SR", ("VW", "L-S")],
                "EW alpha, all factors %/yr": al.loc["EW", "alpha_ann_%"], "t(EW alpha)": al.loc["EW", "alpha_ann_%"] / al.loc["EW", "se_alpha_ann_%"],
                "VW alpha, all factors %/yr": al.loc["VW", "alpha_ann_%"], "t(VW alpha)": al.loc["VW", "alpha_ann_%"] / al.loc["VW", "se_alpha_ann_%"], "EW loading on R_ROE": al.loc["EW", "b_R_ROE"],
                "EWCT gamma=0.1 gross Sharpe": ew.loc[0.1, "gross_SR"], "EWCT gamma=0.1 net Sharpe": ew.loc[0.1, "net_SR"], "EWCT gamma=0.3 net Sharpe": ew.loc[0.3, "net_SR"],
                "delay-2 EW Sharpe": dl.loc[(2, "EW"), "LS_SR"], "delay-3 EW Sharpe": dl.loc[(3, "EW"), "LS_SR"],
                "stocks with news per day": np.average(cnt.firms_per_day, weights=cnt.days), "stories per month": cnt.stories.sum() / cnt.months.sum()}
        acc = table(R, m, "v2024_T05_accuracy", index_col=0)
        if acc is not None:
            sent = table(R, m, "v2024_T06_sentiment", header=[0, 1], index_col=0)
            ours["sentiment accuracy %"] = acc.loc["overall", "accuracy_%"]
            ours["EW sentiment L-S Sharpe"] = sent.loc["SR", ("EW", "L-S")]
        for k, v in ours.items():
            rows.append({"statistic": k, "model": lab, "paper": CKX[m].get(k, np.nan), "ours": v})
    order = list(dict.fromkeys(r["statistic"] for r in rows))
    S = pd.DataFrame(rows).pivot(index="statistic", columns="model", values=["paper", "ours"]).reindex(order)
    S.columns = [f"{b}: {a}" for a, b in S.columns]
    S.insert(0, "paper, LLaMA3 only", pd.Series(CKX_LLAMA3).reindex(order))
    return S


def plot_correlation_by_year(R):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for m, lab in models(R).items():
        c = table(R, m, "T03_correlation", index_col=0)["value"]
        yrs = c[c.index.str.isdigit()]
        ax.plot(yrs.index.astype(int), yrs.values, marker="o", label=f"{lab} (mean {c['Mean (Pearson) %']:.2f}%)")
    for m, ls in [("Llama-2-13b-hf", "--"), ("roberta-large", ":")]:
        ax.axhline(CKX[m]["Pearson corr %"], ls=ls, color="gray", label=f"paper {NAMES[m]} mean {CKX[m]['Pearson corr %']}%")
    ax.set_ylabel("daily Pearson corr, %")
    ax.set_title("Table 3: mean daily cross-sectional correlation by year")
    ax.legend(fontsize=8)
    show(R, "correlation_by_year")


def plot_cumulative(R):
    """their Figure 5: cumulative log returns of the EW and VW quintile long-short strategies over the extended period, the paper's
    period ending at the dashed line"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m, lab in models(R).items():
        c = table(R, m, "F05_cumret", "extended", index_col=0)
        c.index = pd.to_datetime(c.index)
        for ax, w in zip(axes, ["EW", "VW"]):
            ax.plot(c.index, c[f"{w}_LS"], lw=1, label=lab)
    for ax, w in zip(axes, ["EW", "VW"]):
        ax.plot(c.index, c.market, color="gray", lw=1, label="Market")
        ax.axvline(pd.Timestamp(PAPER_END), color="k", ls="--", lw=0.6)
        ax.set_title(f"{w} long-short", fontsize=10, loc="left")
        ax.set_ylabel("Cumulative log return")
        ax.legend(fontsize=8)
    show(R, "fig5_cumulative")


def ewct(R):
    """EWCT gross / net Sharpe and turnover by gamma, with the paper's net Sharpe of the same model (Table 9: articles, alerts) and
    its LLaMA3 numbers (Table 8)"""
    tab = {}
    for m, lab in models(R).items():
        e = table(R, m, "T08_ewct", index_col=0)
        tab[f"{lab} gross"], tab[f"{lab} net"], tab[f"{lab} turnover %"] = e.gross_SR, e.net_SR, e["turnover_%"]
        tab.update({f"paper {lab} net, {k}": pd.Series(v) for k, v in CKX_NET[m].items()})
    paper = pd.DataFrame({f"paper LLaMA3 {k}": v for k, v in CKX_EWCT.items()})
    return pd.DataFrame(tab).join(paper, how="outer")


def plot_ewct(R):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, lab in models(R).items():
        e = table(R, m, "T08_ewct", index_col=0)
        ax.plot(e.index, e.gross_SR, marker="o", label=f"{lab} gross")
        ax.plot(e.index, e.net_SR, marker="s", ls="--", label=f"{lab} net")
    g, n = pd.Series(CKX_EWCT["articles gross"]), pd.Series(CKX_EWCT["articles net"])
    ax.plot(g.index, g.values, "k^", label="paper LLaMA3 gross")
    ax.plot(n.index, n.values, "kv", label="paper LLaMA3 net")
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("Sharpe")
    ax.set_title("Table 8: EWCT gross and net Sharpe")
    ax.legend(fontsize=8)
    show(R, "ewct")


def plot_assimilation(R):
    """their Figure 10: the EW quintile long-short traded 1..6 days after the news, by size (NYSE median), bp per day with 95%
    confidence bands (Newey-West), the paper's period"""
    ms = models(R)
    fig, axes = plt.subplots(1, len(ms), figsize=(6 * len(ms), 3.8), squeeze=False, sharey=True)
    for ax, (m, lab) in zip(axes[0], ms.items()):
        s = table(R, m, "F10_size_delay").set_index(["size", "delay"])
        for size, color in [("large", "tab:blue"), ("small", "tab:orange")]:
            x = s.loc[size]
            se = (x.LS_bp / x.t_nw).abs()
            ax.plot(x.index, x.LS_bp, marker="o", color=color, label=size.capitalize())
            ax.fill_between(x.index, x.LS_bp - 1.96 * se, x.LS_bp + 1.96 * se, color=color, alpha=0.15)
        ax.axhline(0, color="k", ls="--", lw=0.6)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([f"Day{k}" for k in range(1, 7)])
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Return (bps)")
    show(R, "fig10_assimilation")


def plot_loadings(R):
    ms = models(R)
    fig, axes = plt.subplots(1, len(ms), figsize=(7 * len(ms), 3.5), squeeze=False)
    for ax, (m, lab) in zip(axes[0], ms.items()):
        al = table(R, m, "T20_21_alphas", index_col=0)
        b = al.loc[:, al.columns.str.startswith("b_")]
        b.columns = b.columns.str[2:]
        b.T.plot.bar(ax=ax)
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"Tables 20-21 loadings, {lab}")
        ax.tick_params(axis="x", labelsize=8)
    show(R, "loadings")


def periods(R):
    rows = []
    for m, lab in models(R).items():
        for per in ["paper", "extended"]:
            q5 = table(R, m, "T05_quintile", per, header=[0, 1], index_col=0)
            if q5 is None:
                continue
            c = table(R, m, "T03_correlation", per, index_col=0)["value"]
            cnt = table(R, m, "counts", per, index_col=0)
            rows.append({"model": lab, "period": per, "years": f"{cnt.index.min()}-{cnt.index.max()}", "Pearson corr %": c["Mean (Pearson) %"],
                         "EW L-S Sharpe": q5.loc["SR", ("EW", "L-S")], "EW L-S return %/yr": q5.loc["Ret_%/yr", ("EW", "L-S")],
                         "VW L-S Sharpe": q5.loc["SR", ("VW", "L-S")], "stocks/day": np.average(cnt.firms_per_day, weights=cnt.days)})
    return pd.DataFrame(rows).set_index(["model", "period"])


def cross_model(R):
    """Table 4 (Diebold-Mariano, row RoBERTa, column LLaMA2, positive = the column model better) and Figure 7 (strategy correlation), one row per period"""
    f = R / "cross_model.csv"
    if not f.exists():
        return None
    require_same_sample(R / "run_cross_model.json", "cross-model tables", needs=("ridge",))
    return pd.read_csv(f)
