"""Model against data by component and horizon (framework.tex): cal_fig2a (daily forecast), cal_fig2b (monthly-averaged daily
forecast), cal_fig2c (monthly projection forecast), cal_fig3 (MSRR component returns), cal_cutoff_ic and cal_cutoff_placebo
(rank ICs around January 2023). Called from calibration.main(), which passes itself as cal (the module runs as __main__, so it
is not imported here)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELL_COLORS = {"cL": "#1f77b4", "cD": "#aec7e8", "yF": "#ff7f0e", "uL": "#2ca02c", "uD": "#98df8a"}
CELL_NAMES = {"cL": "Event category, level", "cD": "Event category, deviation", "yF": "Firm identity", "uL": "Idiosyncratic, level",
              "uD": "Idiosyncratic, deviation"}
TIMING_COLORS = {"total": "black", "lev": "#ff7f0e", "dev": "#1f77b4"}
TIMING_NAMES = {"total": "Total", "lev": "Level", "dev": "Deviation"}
STYLE = [plt.Line2D([], [], color="0.45", lw=1.2, marker="o", ms=3), plt.Line2D([], [], color="0.45", lw=1.4, ls="--")]
STYLE_NAMES = ["Data: solid line with dots", "Model: dashed line"]


def targeted(mom, freq, k, e):
    """the targeted horizons of component k at frequency freq for encoder e (filled markers)"""
    x = mom[(mom.role == "target") & (mom.block == f"{freq}band_{k}") & (mom.enc == e)]
    return {h for lo, hi in zip(x.lo, x.hi) for h in range(int(lo), int(hi) + 1)}


def ticks(ax, hs, x):
    """every horizon labelled up to twelve, otherwise the first and every sixth (as report._horizon_ticks)"""
    keep = [i for i, h in enumerate(hs) if len(hs) <= 12 or i == 0 or h % 6 == 0]
    ax.set_xticks([x[i] for i in keep])
    ax.set_xticklabels([f"{hs[i]:g}" for i in keep])


def decomposition(cal, data, model, hs, xlabel, ylabel, path, mom=None, freq=None):
    """timing split on top, components below, one column per encoder; data solid with markers (targets filled), model dashed"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), sharey="row", gridspec_kw={"hspace": 0.35})
    x = np.arange(len(hs))
    for j, m in enumerate(cal.PANELS):
        top, bot = axes[0, j], axes[1, j]
        for ax, keys, colors in [(top, ["total", "lev", "dev"], TIMING_COLORS), (bot, list(CELL_NAMES), CELL_COLORS)]:
            for k in keys:
                ax.plot(x, data[m][k], marker="o", ms=3.5, lw=1.2, color=colors[k], markerfacecolor="white")
                if freq is not None and k in CELL_NAMES:
                    hit = targeted(mom, freq, k, cal.CODE[m])
                    tg = [i for i, h in enumerate(hs) if h in hit]
                    ax.plot(x[tg], np.asarray(data[m][k])[tg], marker="o", ms=4.5, ls="none", color=colors[k])
                ax.plot(x, model[m][k], lw=1.4, ls="--", color=colors[k])
            ax.axhline(0, color="0.8", lw=0.6, ls=":")
            ticks(ax, hs, x)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=8)
        top.set_title(f"{cal.NAME[m]}: timing decomposition", fontsize=9, loc="left")
        bot.set_title(f"{cal.NAME[m]}: full decomposition", fontsize=9, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    sq = lambda c: plt.Line2D([], [], color=c, marker="s", ms=6, ls="none")
    axes[0, 0].legend([sq(TIMING_COLORS[k]) for k in TIMING_NAMES] + STYLE, list(TIMING_NAMES.values()) + STYLE_NAMES, fontsize=7, frameon=False)
    axes[1, 0].legend([sq(CELL_COLORS[k]) for k in CELL_NAMES] + STYLE, list(CELL_NAMES.values()) + STYLE_NAMES, fontsize=7, frameon=False,
                      ncol=2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw(d, p, mom, cal, out):
    PANELS, cells = cal.PANELS, cal.cells
    cols = list(CELL_NAMES) + ["lev", "dev", "total"]
    hs = list(range(1, cal.DAY_H + 1))
    data = {m: {k: d["ts"][m]["yhat" if k == "total" else k]["mean"].reindex(hs).values for k in cols} for m in PANELS}
    decomposition(cal, data, {m: cells(p, m, "daily", hs) for m in PANELS}, hs, "$h$ (trading days)", "Cov ($\\times 10^6$)",
                  out / "cal_fig2a.pdf", mom, "day")
    for letter, kind in [("b", "bridge"), ("c", "ridge")]:
        data, model = {}, {}
        for m in PANELS:
            mo = cal.paper("4_monthly_ridge", m, "contributions")
            mo = mo[mo.predictor == kind].assign(h=lambda x: x.tau + 1)
            hm = sorted(mo.h.unique())
            data[m] = {k: mo[mo.part == k].set_index("h")["mean"].reindex(hm).values for k in cols}
            model[m] = cells(p, m, kind, hm, data=data[m]["total"][0])
        decomposition(cal, data, model, hm, "$h$ (months)", "Cov ($\\times 10^6$)", out / f"cal_fig2{letter}.pdf")
    data, model = {}, {}
    for m in PANELS:
        a = cal.attribution(m)
        hm = list(a["lev"].index.values)
        t = a["total"]
        scale = (t["mean_%/yr_at_10%vol"] / t["mean_%/yr"]).reindex(hm).values   # the strategy at 10% volatility, model alike
        data[m] = {k: a[k]["mean_%/yr_at_10%vol"].reindex(hm).values for k in cols}
        model[m] = {k: v * scale for k, v in cells(p, m, "strategy", hm).items()}
    decomposition(cal, data, model, hm, "$h$ (months)", "Return (%/yr, strategy at 10% volatility)", out / "cal_fig3.pdf", mom, "mo")


COLORS = {"chrono-gpt-v1-19991231": "0.45", "roberta-large": "#1f77b4", "Llama-2-13b-hf": "#c55a11"}


def cutoff_figure(lic, cut, cal, path, panels="ABC"):
    """(A) the historical level's mean next-day rank IC before and after January 2023 by encoder; (B) Llama-2 over RoBERTa's
    relative IC ratio of the level by horizon, with 90% intervals; (C) the next-day relative IC ratio at each placebo split (hollow)
    and across January 2023 (filled), by component. framework.tex shows all three panels, Appendix D panels B and C (A repeats
    Table 11)"""
    from matplotlib.patches import Patch
    fig, ax_ = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.0), gridspec_kw={"width_ratios": [1] * (len(panels) - 1) + [1.1]})
    axes = dict(zip(panels, ax_))
    title = lambda p, t: f"{'ABC'[panels.index(p)]}. {t}"
    year = cal.SPLITS["L"]["split"] // 12                                   # the split is its January
    if "A" in panels:
        _panel_a(axes["A"], lic, cal, title("A", f"Historical level before and after {year}"), Patch, year)
    _panel_b(axes["B"], lic, cal, title("B", f"Relative IC ratio of $E_L$ across January {year}"))
    _panel_c(axes["C"], cut, cal, title("C", "Placebo splits"))
    for a in ax_:
        a.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _panel_a(ax, lic, cal, title, Patch, year):
    w = 0.36
    for i, m in enumerate(cal.NAME):
        a, b = lic["means"][m]
        ax.bar(i - w / 2, a, w, color=COLORS[m], alpha=0.45)
        ax.bar(i + w / 2, b, w, color=COLORS[m])
        ax.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.03, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([cal.NAME[m] for m in cal.NAME], fontsize=9)
    ax.set_ylabel("Mean next-day rank IC of $E_L$ (%)", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(handles=[Patch(color="0.45", alpha=0.45, label=f"Through {year - 1}"), Patch(color="0.45", label=f"From {year}")],
              fontsize=8.5, frameon=False, loc="upper left")
    ax.set_ylim(0, 1.6)


def _panel_b(ax, lic, cal, title):
    for i, h in enumerate(cal.IC_HS):
        r = lic["ratio"][h]
        ax.plot([i, i], [r["lo"], r["hi"]], color=COLORS[cal.LM], lw=1.4)
        ax.plot(i, r["split"], "o", color=COLORS[cal.LM], ms=6.5)
        ax.text(i + 0.1, r["split"], f"{r['split']:.2f}", ha="left", va="center", fontsize=9)
    ax.axhline(1, color="0.6", lw=0.9)
    ax.set_xticks(range(len(cal.IC_HS)))
    ax.set_xticklabels([str(h) for h in cal.IC_HS])
    ax.set_xlabel("$h$ (trading days)", fontsize=10)
    ax.set_ylabel("Llama-2 over RoBERTa, after relative to before", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")


def _panel_c(ax, cut, cal, title):
    comps = (("F", "$F$"), ("lev", "$E_L$"), ("dev", "$E_D$"), ("CD", "$C_D$"))
    years, split = cal.PLACEBO_YEARS, cal.SPLITS["L"]["split"] // 12
    for i, (k, lab) in enumerate(comps):
        v = list(cut["placebo"][k]["ratios"].values())
        ax.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, "o", mfc="white", mec="0.35", ms=5,
                label=f"Placebo splits, January {years[0]}\u2013{years[-1]}" if i == 0 else None)
        ax.plot(i, cut["L"][k]["ic"], "o", color=COLORS[cal.LM], ms=7, label=f"Split at January {split}" if i == 0 else None)
    ax.axhline(1, color="0.6", lw=0.9)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([lab for _, lab in comps])
    ax.set_ylabel("Next-day relative IC ratio, Llama-2 over RoBERTa", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
