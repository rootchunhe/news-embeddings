"""Tables and figures of the Didisheim replication notebook, from Results/replication/didisheim."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.runinfo import require_same_sample
from common.paper_numbers import DKPT

NAMES = {"Llama-2-13b-hf": "LLaMA2", "roberta-large": "RoBERTa"}
SPECS = ["Constant", "CAPM", "FF3", "FF6", "JKP"]                       # the bars of their Figures 4 and 7


def show(R, name):
    """save the current figure to R/figures/<name>.pdf, then display it"""
    d = Path(R) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(d / f"{name}.pdf")
    plt.show()


def models(R):
    """the models with results, each manifest checked against the current sample and monthly specification"""
    out = {m: l for m, l in NAMES.items() if (R / m / "run.json").exists()}
    for m in out:
        require_same_sample(R / m / "run.json", f"{m} results", needs=("monthly",))
    return out


def table(R, m, name, **kw):
    """R/<m>/<name>.csv, None when absent; the seasonality table (04_seasonality.py) is checked against its own manifest"""
    f = R / m / f"{name}.csv"
    if not f.exists():
        return None
    if name == "seasonality":
        require_same_sample(R / m / f"run_{name}.json", f"{m} {name}", needs=("monthly",))
    return pd.read_csv(f, **kw)


def runs(R):
    rows = []
    for m in models(R):
        j = json.loads((R / m / "run.json").read_text())
        rows.append({"model": m, "finished": j["finished"], "firm_months": j["firm_months"], "months": j["months"]})
    return pd.DataFrame(rows).set_index("model")


def summary(R):
    """paper vs ours for the headline statistics (replication period)"""
    rows = []
    for m, lab in models(R).items():
        f4 = table(R, m, "F04_msrr").set_index(["spec", "portfolio"]).Sharpe
        t7 = table(R, m, "T07_sorts").set_index(["panel", "weighting", "portfolio"]).SR
        f7 = table(R, m, "F07_size").set_index(["spec", "size"]).Sharpe
        f10 = table(R, m, "F10_lags").set_index("tau")
        ours = {"Raw": f4[("Raw", "raw embedding")], **{k: f4[(k, "news shock")] for k in SPECS + ["IND", "JKP + IND"]},
                "control: JKP characteristics": f4[("JKP characteristics (control)", "control")], "Table 7 MSE, EW H-L": t7[("MSE", "EW", "H-L")],
                "Table 7 MSE, capped VW H-L": t7[("MSE", "capped VW", "H-L")], "Table 7 MSRR, EW H-L": t7[("MSRR", "EW", "H-L")],
                "size: Mega + Large": f7[("JKP", "Mega + Large")], "size: Small + Micro": f7[("JKP", "Small + Micro")]}
        for tau in [1, 6, 12, 18]:
            ours[f"delay {tau} months, weights fixed at tau 0"] = f10.loc[tau, "Sharpe_fixed_b"]
            ours[f"delay {tau} months, weights re-optimized"] = f10.loc[tau, "Sharpe"]
        for k, v in ours.items():
            rows.append({"statistic": k, "model": lab, "paper": DKPT.get(k, np.nan), "ours": v})
    order = list(dict.fromkeys(r["statistic"] for r in rows))
    S = pd.DataFrame(rows).pivot(index="statistic", columns="model", values=["paper", "ours"]).reindex(order)
    S.columns = [f"{b}: {a}" for a, b in S.columns]
    return S


def _cumulative(ax, rets, colors):
    """cumulative sum of the monthly returns, each series scaled to 10% annual volatility, by the month the return is earned"""
    idx = pd.PeriodIndex(rets.index, freq="M") + 1
    for col, color in colors.items():
        r = rets[col] / (rets[col].std() * np.sqrt(12)) * 0.10
        ax.plot(idx.to_timestamp(), r.fillna(0).cumsum(), color=color, lw=1, label=col)
    ax.set_ylabel("Cumulative Return")


def plot_fig4(R):
    """their Figure 4, one row per model: A the Sharpe ratio of the news shock and of the predicted news under each residualization
    (the raw embedding as the dashed line); B their CAPM alphas at 10% volatility with 95% intervals (Newey-West); C the cumulative
    returns of the raw embedding, the predicted news and the news shock (JKP)"""
    ms = models(R)
    fig, axes = plt.subplots(len(ms), 3, figsize=(16, 3.8 * len(ms)), squeeze=False)
    x = np.arange(len(SPECS))
    for row, (m, lab) in zip(axes, ms.items()):
        f4 = table(R, m, "F04_msrr").set_index(["spec", "portfolio"])
        for ax, col, ylab in [(row[0], "Sharpe", "Sharpe Ratio"), (row[1], "alpha_%/yr_at_10%vol", "Alpha (%/yr)")]:
            for k, (port, color) in enumerate([("predicted news", "tab:purple"), ("news shock", "tab:pink")]):
                v = f4.xs(port, level="portfolio").reindex(SPECS)
                err = 1.96 * (v[col] / v.t_alpha).abs() if col.startswith("alpha") else None
                ax.bar(x + (k - 0.5) * 0.38, v[col], 0.38, yerr=err, capsize=3, color=color, label=port)
            ax.axhline(f4.loc[("Raw", "raw embedding"), col], color="tab:green", ls="--", label="raw embedding")
            ax.set_xticks(x)
            ax.set_xticklabels(SPECS)
            ax.set_ylabel(ylab)
            ax.set_title(f"{lab}: {'Sharpe ratios' if col == 'Sharpe' else 'CAPM alpha'}", fontsize=10, loc="left")
        row[0].legend(fontsize=8)
        _cumulative(row[2], table(R, m, "F04_returns", index_col=0), {"raw embedding": "tab:green", "predicted news": "tab:purple", "news shock": "tab:pink"})
        row[2].set_title(f"{lab}: cumulative returns (JKP)", fontsize=10, loc="left")
        row[2].legend(fontsize=8)
    show(R, "fig4_msrr")


def plot_fig7(R):
    """their Figure 7, one row per model: A the news shock's Sharpe ratio within the JKP size groups under each residualization;
    B the cumulative returns of the JKP news shock by size group"""
    ms = models(R)
    fig, axes = plt.subplots(len(ms), 2, figsize=(12, 3.8 * len(ms)), squeeze=False)
    x = np.arange(len(SPECS))
    colors = {"Mega + Large": "tab:green", "Small + Micro": "tab:blue"}
    for row, (m, lab) in zip(axes, ms.items()):
        f7 = table(R, m, "F07_size").set_index(["spec", "size"]).Sharpe
        for k, (size, color) in enumerate(colors.items()):
            row[0].bar(x + (k - 0.5) * 0.38, f7.xs(size, level="size").reindex(SPECS), 0.38, color=color, label=size)
        row[0].set_xticks(x)
        row[0].set_xticklabels(SPECS)
        row[0].set_ylabel("Sharpe Ratio")
        row[0].set_title(f"{lab}: news shock Sharpe ratio by size group", fontsize=10, loc="left")
        row[0].legend(fontsize=8)
        _cumulative(row[1], table(R, m, "F07_returns", index_col=0), colors)
        row[1].set_title(f"{lab}: cumulative return by size group (JKP)", fontsize=10, loc="left")
        row[1].legend(fontsize=8)
    show(R, "fig7_size")


def plot_fig10(R):
    """their Figure 10 Panel A: the news shock traded with a delay of tau = 1..36 months with the undelayed strategy's weights of the
    signal month, annualized average return and CAPM alpha at 10% volatility with 95% bands (Newey-West)"""
    ms = models(R)
    fig, axes = plt.subplots(1, len(ms), figsize=(6.5 * len(ms), 3.8), squeeze=False, sharey=True)
    for ax, (m, lab) in zip(axes[0], ms.items()):
        f10 = table(R, m, "F10_lags").set_index("tau").loc[1:]
        for col, t, name, color in [("mean_%/yr_at_10%vol_fixed_b", "t_mean_fixed_b", "Average Return", "tab:pink"),
                                    ("alpha_%/yr_at_10%vol_fixed_b", "t_alpha_fixed_b", "Alpha", "tab:green")]:
            v, se = f10[col] / 100, (f10[col] / 100 / f10[t]).abs()
            ax.plot(f10.index, v, marker="o", ms=3, color=color, label=name)
            ax.fill_between(f10.index, v - 1.96 * se, v + 1.96 * se, color=color, alpha=0.15)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Trading Delay (months)")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Annualized Return")
    show(R, "fig10_delay")
