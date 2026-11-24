"""LaTeX tables of the paper, its appendix and the key tables of the two replications, from Results/. Run from Codes/3_paper/:
    ../../env/bin/python latex.py
Output Results/paper/tex/<label>.tex (one booktabs table environment each: title, note, tabular) and tables.tex, which inputs
them in paper order; a table whose inputs are missing or refused stops the script. tex/exhibits.tex, a minimal document with every table
and figure in review order, is then compiled with pdflatex; errors, undefined references, floats too large for the page and
overfull boxes beyond OVERFULL_PT make the script exit with status 1."""
import fnmatch
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import report
from report import PARTS_TEX

pd.set_option("future.infer_string", False)                             # keeps the Heading index labels (a str subclass) under pandas >= 3
from lib import ROOT
from lib.constants import (FORECAST_LABEL, POST_YEAR, PRETRAIN_CUTOFF, ROBUST_MODELS, SEGMENTS, SHORT, DRIFT_MIN_STORIES, DRIFT_TABLE_MIN_STORIES, HISTORY_DAYS, LABELS, LEVEL_WINDOW,
                           CATEGORY_MEAN_SHRINKAGE, MIN_GROUP_STORIES, MIN_MONTH_STOCKS, MODELS, FULL, NW_LAGS_DAILY, NW_LAGS_MONTHLY, LEVEL_LAG, SIZE_H, SUBSPACE, SURPRISE, TAUS_FIG, TEST_YEARS, TIMING, H, PROJECTION_YEARS, MIN_STORIES, MIN_FIRM_YEARS)

from common.config import FIRST_YEAR, MSRR_INIT, TRAIN_YEARS, VAL_YEARS
from common.factors import dmrs_daily

R = ROOT / "Results/paper"
REP = ROOT / "Results/replication"
OUT = R / "tex"
OVERFULL_PT = 20
INT_COLS = {"# labels", "categories", "per leg", "days", "n", "headlines", "rewrites"}


# ---- formatting ---------------------------------------------------------------------------------------------------------------
def esc(s):
    """LaTeX-escape a header, an index label or a text cell; math ($...$) and an escaped dollar sign (\\$) are kept"""
    parts = re.split(r"(\\\$|\$[^$]*\$)", str(s))
    return "".join(p if p.startswith("$") or p == "\\$" else p.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
                   .replace("_", r"\_").replace("$", r"\$").replace("^", r"\^{}").replace("~", r"\~{}").replace(">", r"\textgreater{}").replace("<", r"\textless{}") for p in parts)


ONE_DECIMAL = re.compile(r"share|headlines \(%\)|subspace \(%\)|sign \(%\)|decile \(%\)|\(bp\)|l--s|turnover|\(%/yr|^\$\\alpha\$$|bp/day|direction \(%\)|changed \(%\)|rank change|^ret$|^overlap$", re.I)   # percent shares (also the network tables' overlap), basis points, %/yr


def fmt(v, col, digits):
    """a cell as text: integers with thousands separators, percent shares and basis points (any header level matching ONE_DECIMAL)
    with one decimal, other numbers with `digits`, a minus sign instead of a hyphen"""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v == "nan"):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    name = str(col[-1] if isinstance(col, tuple) else col).lower()
    if isinstance(v, (int, np.integer)) or (name in INT_COLS and isinstance(v, (float, np.floating)) and float(v).is_integer()):
        return str(int(v)) if name in {"year", "test_year", "tau", "lag", "h"} else f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        d = 1 if any(ONE_DECIMAL.search(str(c)) for c in (col if isinstance(col, tuple) else (col,))) else digits
        return f"{v:.{d}f}".replace("-", "$-$")                        # a minus sign, not a hyphen
    if isinstance(v, str) and re.match(r"^\(?-\d", v):                       # a preformatted number (regression layout)
        return v.replace("-", "$-$", 1)
    return esc(v)


class Heading(str):
    """an index label printed as an italic row across the table below a rule: a group of rows within one table; an empty heading
    is a rule alone"""


def tabular(df, height=0.75):
    """booktabs tabular of a DataFrame; the index levels become the first (left-aligned) columns, the numbers of a column are aligned
    on the decimal point and centered under their header (align_decimals); a multi-level column header gets one row per level, the upper
    levels grouped by \\multicolumn with \\cmidrule; shrunk to the page width and to `height` of the text height by adjustbox"""
    df = df.copy()
    levels = df.columns.nlevels
    names = [n or "" for n in df.index.names]
    idx = df.index.to_frame(index=False)
    idx.columns = pd.MultiIndex.from_tuples([n if isinstance(n, tuple) else ("",) * (levels - 1) + (n,) for n in names]) if levels > 1 else names
    df = pd.concat([idx, df.reset_index(drop=True)], axis=1)
    cols = [c if isinstance(c, tuple) else (c,) for c in df.columns]
    lines = [f"\\begin{{tabular}}{{{'l' * len(names) + 'c' * (len(cols) - len(names))}}}", "\\toprule"]
    for level in range(levels - 1):                                       # a group: adjacent columns sharing the labels up to this level
        groups = []
        for c in cols:
            if groups and groups[-1][0] == c[:level + 1]:
                groups[-1][1] += 1
            else:
                groups.append([c[:level + 1], 1])
        top, rules, pos = [], [], 1
        for key, n in groups:
            label = "\\shortstack{" + " \\\\ ".join(esc(x) for x in key[-1].split("\n")) + "}" if "\n" in key[-1] else esc(key[-1])   # a header with a line break is stacked
            top.append(f"\\multicolumn{{{n}}}{{c}}{{{label}}}" if n > 1 else label)
            if n > 1 and key[-1] != "":
                rules.append(f"\\cmidrule(lr){{{pos}-{pos + n - 1}}}")
            pos += n
        lines += [" & ".join(top) + " \\\\"] + [" ".join(rules)] * bool(rules)
    head, i = [], 0                                                      # an index name spans the unnamed index columns after it
    while i < len(cols):
        n = 1
        if 0 < i < len(names) and names[i]:
            while i + n < len(names) and not names[i + n]:
                n += 1
        head.append(f"\\multicolumn{{{n}}}{{l}}{{{esc(cols[i][-1])}}}" if n > 1 else esc(cols[i][-1]))
        i += n
    lines.append(" & ".join(head) + " \\\\")
    lines.append("\\midrule")
    prev, body = None, []                                                 # body: rule or heading lines (str) and rows of cells (list)
    grouped = len(names) > 1 and df.iloc[:, 0].duplicated().any()         # rules only when a group spans several rows
    for _, row in df.iterrows():
        vals = list(row.values)
        if isinstance(vals[0], Heading):                                   # an empty heading is a rule alone
            body.append(("" if prev is None else "\\midrule" + "\n" * bool(vals[0])) + (f"\\multicolumn{{{len(cols)}}}{{l}}{{\\textit{{{esc(vals[0])}}}}} \\\\" if vals[0] else ""))
            prev = None
            continue
        same = 0                                                            # leading index levels repeated from the previous row
        while prev is not None and same < len(names) - 1 and vals[same] == prev[same]:
            same += 1
        if grouped and prev is not None and same == 0:
            body.append("\\midrule")                                        # a rule between the groups of the first index level
        elif prev is not None and 0 < same < len(names) - 1:
            body.append(f"\\cmidrule(lr){{{same + 1}-{len(cols)}}}")        # a partial rule when a middle level changes
        body.append([""] * same + [fmt(v, k, 2) for v, k in zip(vals[same:], cols[same:])])   # repeated levels printed once
        prev = vals
    align_decimals([r for r in body if isinstance(r, list)], len(names))
    lines += [r if isinstance(r, str) else " & ".join(r) + " \\\\" for r in body] + ["\\bottomrule", "\\end{tabular}"]
    return f"\\begin{{adjustbox}}{{max width=\\linewidth, max totalheight={height:.2f}\\textheight}}\n" + "\n".join(lines) + "\n\\end{adjustbox}"


NUMBER = re.compile(r"^\(?(\$-\$)?[\d,]+(\.\d+)?\)?$")                  # a formatted number, also a t-statistic in parentheses
GLYPHS = ["$-$", "(", ")", ",", ".", "0"]                                   # the characters of a number by width class (digits share one)


def align_decimals(rows, first):
    """pads every number of a value column (from column `first` on) with invisible characters up to the column's widest integer part
    and fraction part, so that the centered column lines its numbers up on the decimal point"""
    def count(text):
        return [text.count("$-$"), text.count("("), text.count(")"), text.count(","), text.count("."), sum(ch.isdigit() for ch in text)]
    for j in range(first, len(rows[0]) if rows else 0):
        nums = [i for i, r in enumerate(rows) if NUMBER.match(r[j])]
        sides = {i: [count(rows[i][j].split(".")[0]), count(rows[i][j][len(rows[i][j].split(".")[0]):])] for i in nums}
        widest = [[max(sides[i][k][g] for i in nums) for g in range(len(GLYPHS))] for k in (0, 1)] if nums else []
        for i in nums:
            pad = ["".join(glyph * (w - c) for glyph, w, c in zip(GLYPHS, widest[k], sides[i][k])) for k in (0, 1)]
            rows[i][j] = "".join([f"\\phantom{{{pad[0]}}}" * bool(pad[0]), rows[i][j], f"\\phantom{{{pad[1]}}}" * bool(pad[1])])


def dmrs_end():
    """the last quarter of the DMRS hedged factors (the end of the extended alpha tables)"""
    d = dmrs_daily().dropna().index.max()
    return f"{d.year}Q{d.quarter}"


WIDE = 160                                                              # a tabular wider than this many characters is set sideways


def width(df):
    """the rough natural width of a tabular in characters: per column its longest header or cell, plus the gap"""
    d = df.reset_index()
    head = lambda c: re.sub(r"\\[a-zA-Z]+|[{}$^_]", "", str(c[-1] if isinstance(c, tuple) else c))   # the printed characters of a header
    cell = lambda v: len(f"{v:.2f}") if isinstance(v, float) else len(str(v))
    return sum(max([len(head(c))] + [cell(v) for v in d[c]]) + 2 for c in d.columns)


def note_names(note):
    """encoder names in a note: the full name at the first mention, the short name after; ChronoGPT always with its vintage"""
    seen = set()
    def one(m):
        e = "roberta-large" if m.group(0).startswith("R") else "Llama-2-13b-hf"
        name = MODELS[e] if e in seen else FULL[e]
        seen.add(e)
        return name
    note = re.sub(r"RoBERTa(-large)?|Llama-2( \(13B\))?", one, note)
    chrono = ROBUST_MODELS["chrono-gpt-v1-19991231"]
    note = note.replace("ChronoGPT (December 1999 vintage)", chrono).replace("the December 1999 ChronoGPT vintage", chrono)
    return re.sub(r"ChronoGPT(?!\$_\{1999\}\$)", lambda m: chrono, note)


def environment(label, title, note, body):
    """the table environment: title, label, the note (what was done), then the tabular(s); `body` is a DataFrame or {panel: DataFrame};
    a table wider than WIDE characters is a sidewaystable (package rotating)"""
    env = "sidewaystable" if max(width(df) for df in (body.values() if isinstance(body, dict) else [body])) > WIDE else "table"
    if isinstance(body, dict):
        tab = "\n\\par\\medskip\n".join(f"\\textit{{{esc(name)}}}\\\\[2pt]\n" + tabular(df, height=0.55 * (len(df) + 3) / sum(len(d) + 3 for d in body.values()))
                                         for name, df in body.items())   # the panels share the page by their lines (rows + header)
    else:
        tab = tabular(body)
    note = note_names(re.sub(r"(?<!\\)%", r"\\%", note.strip()))        # a bare % would start a TeX comment
    return "\n".join([f"\\begin{{{env}}}[htbp]", "\\centering", f"\\caption{{{title}}}", f"\\label{{tab:{label}}}",
                      "\\begin{minipage}{\\linewidth}\\footnotesize", note, "\\end{minipage}", "\\medskip", "", "{\\footnotesize", tab, "}", f"\\end{{{env}}}", ""])


def panels(**kw):
    return {f"Panel {chr(65 + i)}: {k}": v for i, (k, v) in enumerate(kw.items())}


NAMES = {"IC_%": "Rank IC (%)", "t_IC": "$t$", "EW_LS_bp": "EW L--S (bp)", "EW_Sharpe": "EW SR", "VW_LS_bp": "VW L--S (bp)", "VW_Sharpe": "VW SR",
         "days": "Days"}


def pretty(df):
    """display names for the columns (one or two levels) and the index labels"""
    return (df.rename(columns=NAMES, index=LABELS).rename(index=PARTS_TEX)
            .rename(index=lambda s: s.replace(" x ", " $\\times$ ") if isinstance(s, str) else s))


# ---- shared wording -------------------------------------------------------------------------------------------------------------
YEARS = f"{TEST_YEARS[0]}--{TEST_YEARS[-1]}"
(ROBUST,) = ROBUST_MODELS                                                # the robustness encoder of Table 10 and the B tables
SEG = "{}--{}".format(*SEGMENTS[ROBUST]["post"])                         # the B tables' second segment: after Llama-2's cutoff
CUTOFFS = " and ".join(f"{pd.Period(c, 'M').strftime('%B %Y')}" for c in PRETRAIN_CUTOFF.values())


# ---- paper tables --------------------------------------------------------------------------------------------------------------
WEIGHTS = {"EW": "Equal-weighted", "VW": "Value-weighted"}
T1_HEAD = {"cells": ("", "", "# Labels"), "IC_%": ("Rank IC (%)", "", "Mean"), "t_IC": ("Rank IC (%)", "", "$t$"),
           **{f"{w}_Sharpe": (WEIGHTS[w], "", "SR") for w in WEIGHTS},
           **{f"net_{w}_{th:g}": (WEIGHTS[w], "Net SR", f"$\\gamma={th:g}$") for w in WEIGHTS for th in report.COST_GAMMAS},
           "names_leg": ("", "Names", "per leg")}


def tab1():
    t = {k: v.rename(index=lambda s: s.replace(" x ", " $\\times$ ")) for k, v in report.table1(R).items()}
    lower = pd.DataFrame(index=[Heading("Coarser classifications and dictionary tone")], columns=t["A"].columns, dtype=float)
    body = pd.concat([t["A"], lower, t["B"]]).drop(columns=["EW_LS_bp", "VW_LS_bp"])
    return body.set_axis(pd.MultiIndex.from_tuples([T1_HEAD[c] for c in body.columns]), axis=1).rename_axis("Signal")


FM_PANEL, RESID_PANEL = "Fama--MacBeth regressions, standardized signals (bp per standard deviation)", "Out of sample: the category mean plus a ridge forecast of its residual"


def fm_panel(name, encoders=MODELS):
    return report.table_fm(R, name, encoders).rename(index=lambda s: s.replace(" x ", " $\\times$ ")).rename_axis("")


def residual_panel(name, encoders=MODELS):
    r = report.table_residual(R, name, encoders)
    r.columns = pd.MultiIndex.from_tuples([RESID_HEAD[c] for c in r.columns])
    return r


def tab2(post=False):
    s = "_post" if post else ""
    return panels(**{FM_PANEL: fm_panel("fm" + s), RESID_PANEL: residual_panel("residual" + s).rename_axis("Forecast")})


SHARES_HEAD = {"A_j": ("Share (%)", "$\\mathrm{Cov}(\\hat y^j,r)/\\mathrm{Cov}(\\hat y,r)$"),
               "var_share": ("Share (%)", "$\\mathrm{Var}(\\hat y^j)/\\mathrm{Var}(\\hat y)$"),
               "IC_%": ("Standalone performance", "Rank IC (%)"), "EW_Sharpe": ("Standalone performance", "EW SR"),
               "VW_Sharpe": ("Standalone performance", "VW SR")}


def shares(t):
    """a decomposition table with the shares and the part-as-signal columns under two-level headers"""
    t = t.rename(index=PARTS_TEX).rename_axis(["Model", "Decomposition"])
    t[["A_j", "var_share"]] *= 100
    return t.set_axis(pd.MultiIndex.from_tuples([SHARES_HEAD[c] for c in t.columns]), axis=1)


def tab6(seg=None, encoders=MODELS):
    return shares(report.table_split(R, seg, encoders))


def a_centroid():
    return shares(report.table_centroid(R))


SUE_HEAD = {"R2_oos_%": "OOS $R^2$ (%)", "spearman": "Spearman", "R2_%": "OOS $R^2$ (%)", "IC_%": "Rank IC (%)"}


def tab3():
    t = {k: v.rename(columns=SUE_HEAD, level=1) for k, v in report.table_sue(R).items()}
    p, n = t["predict"], len(report.SUE_BASE)
    rule = pd.DataFrame(index=pd.MultiIndex.from_tuples([(Heading(""), "")], names=p.index.names), columns=p.columns, dtype=float)
    p = pd.concat([p.iloc[:n], rule, p.iloc[n:]])                       # a rule before the rows with the surprise
    return panels(**{"Explaining the SUE rank": t["read"], "Predicting the next-day return": p})


def horizon_panels(parts, total, seg=None, encoders=MODELS):
    """Table 7 (timing; seg: Tables C3, B3) or Table A14 (full decomposition): one panel per forecast, rows model x horizon h"""
    return panels(**{report.FORECASTS[f]: t.rename_axis(["Model", report.HORIZON_AXIS[f]]) for f, t in report.horizon_table(R, parts, total, seg, encoders).items()})


def tab7():
    return horizon_panels(report.TIMING_PARTS, True)


def attribution(weights="fixed", name="attribution", encoders=MODELS):
    t = report.table_attribution(R, weights, name, encoders)
    return panels(**{"Timing decomposition": t["timing"].rename_axis(["Model", "$h$ (months)"]),
                     "Full decomposition": t["full"].rename_axis(["Model", "$h$ (months)"])})


def tab8():
    return attribution()


# ---- appendix --------------------------------------------------------------------------------------------------------------------
SIGNAL_HEAD = {"stories_pct": ("", "Headlines (%)"), "Category mean (8-yr)": ("Category mean", "8-yr"), "Category mean (1-yr)": ("Category mean", "1-yr"),
               **{FORECAST_LABEL[m]: ("Forecast", SHORT[m]) for m in MODELS}, "RavenPack ESS": ("RavenPack", "ESS")}
SAMPLES = {"all": "All firm-days", "size: micro": "Micro", "size: small": "Small", "size: big": "Big", "ex insider-trading": "Without insider trading",
           "ex price-derived groups": "Without price-derived news", "headline-only news types": "Headline-only news",
           "FULL-ARTICLE news type": "Full-article news"}


def a_subsamples():
    t = report.table_subsamples(R)
    rows = []
    for m, name in [("IC_%", "Rank IC (%)"), ("EW_Sharpe", "EW SR"), ("VW_Sharpe", "VW SR")]:
        rows += [pd.DataFrame(index=[Heading(name)], columns=t[m].columns, dtype=float), t[m].rename(index=SAMPLES)]
    body = pd.concat(rows)
    body.columns = pd.MultiIndex.from_tuples([SIGNAL_HEAD[c] for c in body.columns])
    tests = report.table_tests(R)
    tests.columns = pd.MultiIndex.from_tuples([("Rank IC (%)", "$\\Delta$"), ("Rank IC (%)", "DM $t$"), ("EW SR", "$\\Delta$"), ("EW SR", "JK $z$"),
                                               ("VW SR", "$\\Delta$"), ("VW SR", "JK $z$")])
    return panels(**{"By subsample": body.rename_axis("Sample"), "Differences from the 8-year category mean, all firm-days": tests.rename_axis("Signal")})


def a_costs():
    t = report.table_costs(R)
    return panels(**{f"{k.capitalize()} Sharpe ratio": pretty(t[k]).rename_axis("Signal").rename(columns=lambda x: f"$\\gamma={x:g}$" if isinstance(x, float) else WEIGHTS.get(x, x))
                     for k in ["gross", "net"]})


def windows_table():
    t = report.table(R, "1_category", "windows").set_index("signal").drop(columns="period")
    assert t.days.nunique() == 1, "the windows are not on the same days"
    return t


def a_windows():
    return pretty(windows_table().drop(columns="days").rename(index={"Category mean (through T-3)": "Category mean (through $T-3$)"})).rename_axis("Window")


def a_mz():
    s = report.table_signals(R).rename(index=lambda s: s.replace(" x ", " $\\times$ "))
    s.columns = pd.MultiIndex.from_tuples([("Mincer--Zarnowitz", "Slope $b$"), ("Mincer--Zarnowitz", "$t(b=1)$"), ("Var. explained", "by category (%)"),
                                           ("Correlation with", "Category"), ("Correlation with", "Group $\\times$ dir.")])
    return s.rename_axis("Signal")


def a_spanning():
    short = {"Category mean (8-yr)": "Category mean", **{FORECAST_LABEL[m]: SHORT[m] for m in MODELS}}
    s = report.table_spanning(R).rename(index=short)
    t = s.unstack("factors")
    t = t.reorder_levels([1, 0], axis=1)[[(f, c) for f in ["none", "factors"] for c in ["beta", "alpha_bp_day", "t_alpha", "R2", "appraisal_ratio"]]]
    t = t.rename(columns={"none": "Benchmark only", "factors": "Benchmark + factors", "alpha_bp_day": "$\\alpha$ (bp)", "t_alpha": "$t$", "R2": "$R^2$",
                          "appraisal_ratio": "Appraisal", "beta": "$\\beta$"})
    return t.loc[s.droplevel("factors").index.unique()].rename_axis(["Weighting", "Test portfolio", "Benchmark"])


def a_drift():
    t = report.table_drift(R)
    t = t[t.index.get_level_values(0).isin(list(report.INFO_GROUPS) + ["decile long-short"])]
    headings = {"stories": ("", "Headlines"), "jump_bp": ("Jump", "bp"), "t_jump": ("Jump", "$t$"),
                "day1_bp": ("Day 1", "bp"), "t_day1": ("Day 1", "$t$"),
                "days_1_10_bp": ("Days 1--10", "bp"), "t_1_10": ("Days 1--10", "$t$")}
    return_columns = ["jump_bp", "day1_bp", "days_1_10_bp"]
    t[return_columns] = t[return_columns].map(lambda v: f"{v:.1f}")
    t.columns = pd.MultiIndex.from_tuples([headings[c] for c in t.columns])
    ls = t.xs("decile long-short", level=0).drop(columns=[headings["stories"]])
    return panels(**{"Event categories": t.drop(index="decile long-short", level=0).rename_axis(["Group", "Category"]),
                     "Decile long-short portfolios": ls.rename_axis("Signal")})


DECOMP_ROWS = {"Full decomposition": ["event category: level", "event category: deviation", "firm identity", "idiosyncratic: level",
                                             "idiosyncratic: deviation", "forecast"],
               f"The idiosyncratic component and the firm's own {HISTORY_DAYS}-day mean": [LABELS["yH"], LABELS["yUH"]]}
DECOMP_HEAD = {"A_j": ("Share (%)", "Cov"), "var_share": ("Share (%)", "Var"), "IC_%": ("Standalone", "Rank IC (%)"),
               "EW_Sharpe": ("Standalone", "EW SR"), "VW_Sharpe": ("Standalone", "VW SR")}


def a_decomposition():
    """the full decomposition and the 20-day split of the idiosyncratic component, the two models side by side"""
    t, rows = report.table_decomposition_full(R)[list(DECOMP_HEAD)], []
    t[["A_j", "var_share"]] *= 100
    labels = {**PARTS_TEX, "firm identity": "Firm identity $\\hat y^F$ (all level)"}
    for name, parts in DECOMP_ROWS.items():
        wide = pd.concat({m: t.loc[m].loc[parts].rename(index=labels) for m in t.index.unique(0)}, axis=1)
        rows += [pd.DataFrame(index=[Heading(name)], columns=wide.columns, dtype=float), wide]
    body = pd.concat(rows).rename_axis("Decomposition")
    body.columns = pd.MultiIndex.from_tuples([(m,) + DECOMP_HEAD[c] for m, c in body.columns])
    return body


PROBE_HEAD = {"IC_%": ("Rank IC (%)", "Mean"), "DM_t": ("Rank IC (%)", "DM $t$"), "EW_Sharpe": ("EW SR", "SR"), "JK_z": ("EW SR", "JK $z$"),
              "VW_Sharpe": ("VW SR", "SR"), "JK_z_VW": ("VW SR", "JK $z$")}
RESID_HEAD = {**PROBE_HEAD, "R2_bp": ("OOS $R^2$", "bp"), "t_mse": ("OOS $R^2$", "$t$")}


def a_probe():
    """each refitted signal with its tests against the headline embedding"""
    def one(m):
        s = report.table(R, "2_probe", "summary", m).set_index("signal").loc[list(report.PROBE_NAMES), ["IC_%", "EW_Sharpe", "VW_Sharpe"]]
        return s.join(report.table(R, "2_probe", "increments", m).set_index("A")[["DM_t", "JK_z", "JK_z_VW"]])[list(PROBE_HEAD)]
    t = report.per_model(R, "2_probe", one).rename(index=report.PROBE_NAMES)
    t.columns = pd.MultiIndex.from_tuples([PROBE_HEAD[c] for c in t.columns])
    return t.rename_axis(["Model", "Input"])


REWRITE_HEAD = {"n": ("", "Rewrites"), "d_dir": ("Forecast change (sd)", "Signed"), "abs_d_yhat_sd": ("Forecast change (sd)", "Absolute"),
                "correct_sign": ("Forecast change (sd)", "Correct sign (%)"), "abs_d_rank": ("Rank change", "Absolute"),
                "decile_change": ("Rank change", "Decile (%)"), "cosine": ("", "Cosine"), "in_category_subspace": ("In subspace (%)", "Category"),
                "in_firm_subspace": ("In subspace (%)", "Firm")}
REWRITE_EXAMPLES = {"Profit headline": "Microsoft 2Q EPS 14c >MSFT", "Loss headline": "Tesla 4Q Loss/Shr 69c >TSLA"}   # Panel B of the rewrite table
REWRITE_COLS = ["n", "d_dir", "abs_d_yhat_sd", "correct_sign", "abs_d_rank", "decile_change", "cosine", "in_category_subspace", "in_firm_subspace"]


def a_rewrites():
    """Panel A: the rewrite statistics, the models as row groups; Panel B: the rewrites of REWRITE_EXAMPLES (Data/clean/perturb_texts.parquet)"""
    t = report.table_rewrites(R).rename(index={"name swapped (same industry)": "name swapped"})
    t[["in_category_subspace", "in_firm_subspace"]] *= 100
    t["cosine"] = t.cosine.map("{:.3f}".format)                         # 2 decimals would print 1.00 throughout
    rows = []
    for m in t.index.unique(0):
        rows += [pd.DataFrame(index=[Heading(m)], columns=REWRITE_COLS, dtype=float), t.loc[m, REWRITE_COLS]]
    body = pd.concat(rows).rename_axis("Rewrite")
    body.columns = pd.MultiIndex.from_tuples([REWRITE_HEAD[c] for c in REWRITE_COLS])
    P = pd.read_parquet(ROOT / "Data/clean/perturb_texts.parquet", columns=["rp_story_id", "variant", "text"])   # 2_perturb --build: model-free
    cols = {}
    for name, head in REWRITE_EXAMPLES.items():
        sid = P.rp_story_id[(P.variant == "original") & (P.text == head)]
        assert len(sid) == 1, f"{head}: {len(sid)} stories"
        x = P[P.rp_story_id == sid.iat[0]].set_index("variant").text
        assert x.index.is_unique and set(report.REWRITES) <= set(x.index), f"{head}: rewrites missing"
        cols[name] = x.str.replace("$", "\\$", regex=False)
    ex = pd.DataFrame(cols).reindex(["original"] + report.REWRITES).rename(index={"name swapped (same industry)": "name swapped"})
    return panels(**{"Forecast and embedding changes": body, "Examples": ex.rename_axis("Rewrite")})


TOTAL, BAND_VAR, BAND_COV = "Share of total (%)", "Share of band\nvariance (%)", "Share of band\ncovariance (%)"
BAND = {"var_share": (TOTAL, "Variance"), "cov_share": (TOTAL, "Covariance"), "shrinkage": ("", "$\\bar\\phi$")}
SPECTRUM_A = {**BAND, **{f"{v}_{j}": (head, f"${j}$") for v, head in [("var", BAND_VAR), ("cov", BAND_COV)] for j in "CFI"}}
SPECTRUM_B = {**BAND, **{f"{v}_{j}": (head, name) for v, head in [("var", BAND_VAR), ("cov", BAND_COV)]
                         for j, name in [("G", "Group"), ("D", "Direction"), ("W", "Within cell")]}}


def a_spectrum():
    t = report.spectrum_bands(R).rename_axis(["Model", "PCs"])
    a, b = t[list(SPECTRUM_A)], t[list(SPECTRUM_B)]
    a.columns, b.columns = pd.MultiIndex.from_tuples(SPECTRUM_A.values()), pd.MultiIndex.from_tuples(SPECTRUM_B.values())
    return panels(**{"Content decomposition": a, "Group $\\times$ direction cells": b})


GROUP_HEAD = {"stories": ("", "Headlines"), "IC_category_mean_%": ("Rank IC (%)", "Category mean"), "IC_forecast_%": ("Rank IC (%)", "Forecast"),
              "A_yC": ("Share of the predictive covariance (%)", "Category"),
              "A_yF": ("Share of the predictive covariance (%)", "Firm"), "A_yU": ("Share of the predictive covariance (%)", "Idiosyncratic"),
              "A_lev": ("Share of the predictive covariance (%)", "Level"), "A_dev": ("Share of the predictive covariance (%)", "Deviation"),
              "contribution_bp_day": ("Contribution to", "CKX L--S (bp)")}


def a_groups():
    t = report.table_groups(R)
    t.columns = pd.MultiIndex.from_tuples([GROUP_HEAD[c] for c in t.columns])
    return t.rename(index=lambda g: g.replace("-", " ").capitalize(), level=1).rename_axis(["Model", "Event group"])


FACTOR_ROWS = [("Content", "event category"), ("Content", "firm identity"), ("Content", "idiosyncratic"), ("Timing", "historical level"),
               ("Timing", "deviation from the level"), ("", "forecast")]
FACTOR_HEAD = {"alpha_%/yr": "$\\alpha$", "t_alpha": "$t$", "R2": "$R^2$", "b_rmw": "RMW", "b_hml": "HML"}


def a_factors():
    """the content and the timing parts and the forecast, EW and VW side by side"""
    t = report.table_factors(R)
    t = t.loc[[(m, lab) for m in t.index.unique(0) for _, lab in FACTOR_ROWS]]
    t.index = pd.MultiIndex.from_tuples([(m, dec, PARTS_TEX[lab]) for m in t.index.unique(0) for dec, lab in FACTOR_ROWS],
                                        names=["Model", "Decomposition", ""])
    t.columns = pd.MultiIndex.from_tuples([(WEIGHTS[w], FACTOR_HEAD[c]) for w, c in t.columns])
    return t


def a_horizons():
    return horizon_panels(report.CELLS, False)


CONSTRUCTION_ROWS = report.FIG4_UPPER                                   # Figure 4's inputs; the template and event-group curves stay in the results


def a_reoptimized():
    return attribution("re-optimized")


def a_constructions(col="Sharpe", encoders=MODELS, industry=False):
    """the inputs of CONSTRUCTION_ROWS by model; industry: then those of Figure 4's lower panel with the industry-group indicators
    in the residualization (4_didisheim --industry)"""
    t = report.standalone(R, col, encoders)
    ind = report.standalone(R, col, encoders, "4_didisheim/industry") if industry else None
    def rows(m):
        out = t.loc[m].loc[CONSTRUCTION_ROWS, report.MONTHLY_H].rename(index=report.STANDALONE_NAMES)
        if industry:
            out = pd.concat([out, ind.loc[m].loc[report.FIG4_LOWER, report.MONTHLY_H].rename(index=lambda o: report.STANDALONE_NAMES[o] + report.INDUSTRY_CONTROL)])
        return out
    return pd.concat({m: rows(m) for m in t.index.unique(0)}, names=["Model", "Input"]).rename(columns=lambda c: f"$h={c}$")


def dims():
    """{model label: embedding dimension d}"""
    return {lab: int(report.table(R, "2_spectrum", "by_k", m).k.max()) for m, lab in report.models(R, "2_spectrum").items()}


def a_representation(first=FIRST_YEAR, encoders=MODELS):
    def panel(sample):
        t = report.representation(R, sample, first, encoders)
        t.columns = pd.MultiIndex.from_tuples([("$R^2$ (%)", c) for c in t.columns])
        return t.rename_axis(["Model", "Inputs"])
    return panels(**{"Split sample": panel("split"), "Fitted on the previous window": panel("previous window")})


def tab4():
    """the inputs (rows) by model and outcome (columns), the split sample and the fit on the previous window"""
    def panel(sample):
        t = report.representation(R, sample, inputs=report.VALUATION_INPUTS).unstack(0)
        t = t.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([list(MODELS.values()), list(report.REPRESENTATION_OUTCOMES.values())]))
        return t.reindex(index=list(report.VALUATION_INPUTS.values())).rename_axis("Inputs")
    return panels(**{"Split sample": panel("split"), "Fitted on the previous window": panel("previous window")})


def headed(parts):
    """one tabular of the network tables: each part's rows under an italic heading (a rule before every heading but the first)"""
    first = next(iter(parts.values()))
    rows = []
    for name, df in parts.items():
        rows += [pd.DataFrame(index=[Heading(name)], columns=first.columns, dtype=float), df]
    t = pd.concat(rows).rename_axis("Signal")
    return t


def tab5():
    """industry classifications and the rows of the main encoders, windows from FIRST_YEAR (one run, one sample)"""
    net = report.networks(R, MODELS, display=True)
    return headed({"Panel A: Industry classifications": net[""], **{f"Panel {chr(66 + i)}: {lab}": net[lab] for i, lab in enumerate(MODELS.values())}})


def c_cutoff_groups():
    """Table 11: per component a heading, each main encoder's mean ICs before and after the split, then the relative IC ratio with its
    interval below it; one column per group of report.CUTOFF_GROUPS"""
    t = report.table_cutoff_groups(R)
    RB, LM = MODELS
    periods = {"pre": f"{TEST_YEARS[0]}--{POST_YEAR[LM] - 1}", "post": f"{POST_YEAR[LM]}--{TEST_YEARS[-1]}"}
    rows, index = [], []
    for k in TIMING:
        d = t[t.part == k].set_index("group").reindex(list(report.CUTOFF_GROUPS))
        rows.append([np.nan] * len(d))
        index.append((Heading(PARTS_TEX[LABELS[k]]), ""))
        for m in (RB, LM):
            for p, lab in periods.items():
                rows.append(list(d[f"{m}_{p}"]))
                index.append((MODELS[m], lab))
        rows += [list(d.ratio), [f"[{fmt(a, '', 2)}, {fmt(b, '', 2)}]" if np.isfinite(a) else np.nan for a, b in zip(d.lo, d.hi)]]
        index += [("Relative IC ratio", ""), ("Relative IC ratio", "")]
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index),
                        columns=pd.MultiIndex.from_tuples(list(report.CUTOFF_GROUPS.values())), dtype=object)


def b_networks():
    """Table C7: each main encoder's rows on windows from the year after its pretraining cutoff, with the industry classifications on the
    same firm-years"""
    nets = {m: report.networks(R, {m: lab}, POST_YEAR[m], display=True) for m, lab in MODELS.items()}
    return headed({f"Panel {chr(65 + i)}: {lab}, windows from {POST_YEAR[m]}": pd.concat([nets[m][""], nets[m][lab]])
                   for i, (m, lab) in enumerate(MODELS.items())})


def c_networks():
    """Table B7: the robustness encoder's rows on windows from FIRST_YEAR and from the B tables' second segment"""
    lab = ROBUST_MODELS[ROBUST]
    nets = {first: report.networks(R, {ROBUST: lab}, first, display=True) for first in [FIRST_YEAR, SEGMENTS[ROBUST]["post"][0]]}
    return headed({f"Panel {chr(65 + i)}: windows from {first}": pd.concat([net[""], net[lab]]) for i, (first, net) in enumerate(nets.items())})


# ---- Table 10 and Tables B: the robustness encoder ------------------------------------------------------------------------------------------
def by_segment(build):
    """a B table: build(segment) for the robustness encoder, its segments (signal years) in place of the model as the outer rows"""
    parts = {report.segment_label(ROBUST, s): build(s) for s in SEGMENTS[ROBUST]}
    stack = lambda key: pd.concat({lab: (p[key] if key else p).droplevel(0) for lab, p in parts.items()}, names=["Signal years"])
    first = next(iter(parts.values()))
    return {k: stack(k) for k in first} if isinstance(first, dict) else stack(None)


def c1():
    """Table 2 for the robustness encoder: the FM panel's segments side by side (columns renumbered), the residual panel's stacked"""
    seg = {report.segment_label(ROBUST, s): s for s in SEGMENTS[ROBUST]}
    fm = pd.concat({lab: fm_panel(f"fm_{s}", ROBUST_MODELS).droplevel(0, axis=1) for lab, s in seg.items()}, axis=1)
    fm.columns = pd.MultiIndex.from_tuples([(s, f"({k + 1})") for k, (s, _) in enumerate(fm.columns)])
    r = pd.concat({lab: residual_panel(f"residual_{s}", ROBUST_MODELS) for lab, s in seg.items()}, names=["Signal years", "Forecast"])
    return panels(**{FM_PANEL: fm, RESID_PANEL: r})


def months_h1(m, name):
    """the signal months of an attribution table at h = 1, before the report's month filter (0 without the row)"""
    A = report.table(R, "4_didisheim/attribution", name, m)
    n = A[(A.weights == "fixed") & (A.tau == 0) & (A.part == "total")].months
    return int(n.iloc[0]) if len(n) else 0


def a_old_news():
    t = report.joint_fm(R).rename(index=report.JOINT_FM_TEX).rename(columns=lambda k: f"$k={k}$")
    return t.rename_axis(["Model", ""])


# ---- replication tables: the papers' layouts, our two models under the papers' names ----------------------------------------------
REP_MODELS = {"Llama-2-13b-hf": "LLaMA2", "roberta-large": "RoBERTa"}
CKX_PERIODS = {"paper": "2004--2019H1", "extended": f"2004--{TEST_YEARS[-1]}"}
DID_PERIODS = {"didisheim": "2000--2022", "didisheim_extended": f"2000--{TEST_YEARS[-1]}"}
ALPHA_ROWS = {"MktRF_DMRS": "MktRF_DMRS", "HML_DMRS": "HML_DMRS", "RMW_DMRS": "RMW_DMRS", "CMA_DMRS": "CMA_DMRS", "SMB_DMRS": "SMB_DMRS",
              "R_MKT": "R_MKT", "R_ME": "R_ME", "R_IA": "R_IA", "R_ROE": "R_ROE", "R_EG": "R_EG", "ltrev": "LT_Rev", "strev": "ST_Rev",
              "umd": "Mom"}                                                    # their Tables 20-21, in their order


def ckx(name, period, **kw):
    """{model: table} of Results/replication/ckx/<model>/<period>/<name>.csv"""
    return {m: report.read(REP / "ckx" / m / period / f"{name}.csv", **kw) for m in REP_MODELS}


def did(name, folder, **kw):
    """{model: table} of Results/replication/<folder>/<model>/<name>.csv"""
    return {m: report.read(REP / folder / m / f"{name}.csv", **kw) for m in REP_MODELS}


def r_ckx_corr():
    """their Table 3: mean, pooled and large-cap correlations, then the years; the extended panel adds the years after the paper's"""
    rows = {"Mean (Pearson) %": "Mean", "Pooled %": "Pooled", "Over NYSE median %": "Over NYSE Median"}
    out = {}
    for period, lab in CKX_PERIODS.items():
        t = pd.DataFrame({REP_MODELS[m]: f.set_index(f.columns[0])["value"] for m, f in ckx("T03_correlation", period).items()})
        years = [y for y in t.index if y.isdigit() and (period == "paper" or int(y) >= 2019)]
        out[lab] = t.loc[list(rows) + years].rename(index=rows).rename_axis("")
    return panels(**out)


def r_ckx_dm():
    """their Table 4 (RoBERTa row, LLaMA2 column; bold above the one-sided 5% critical value) and Figure 7 (strategy correlations)"""
    f = REP / "ckx" / "cross_model.csv"
    t = report.read(f, manifest=f.with_name("run_cross_model.json")).set_index("period").rename(index=CKX_PERIODS)
    bold = lambda v: f"$\\mathbf{{{v:.3f}}}$" if v > 1.645 else f"{v:.3f}"
    return pd.DataFrame({"Diebold--Mariano, RoBERTa vs. LLaMA2": t.DM_stat.map(bold), "Strategy correlation, EW": t.strategy_corr_EW.map("{:.2f}".format),
                         "Strategy correlation, VW": t.strategy_corr_VW.map("{:.2f}".format)}).T.rename_axis("")


def r_ckx_sorts():
    """their Table 5: the two models side by side, EW and VW long / short / long-short, rows Ret, Std, SR (annualized decimals)"""
    out = {}
    for period, lab in CKX_PERIODS.items():
        frames = {}
        for m, f in ckx("T05_quintile", period, header=[0, 1], index_col=0).items():
            f = f.T.rename(columns={"Ret_%/yr": "Ret", "Std_%/yr": "Std"})
            f[["Ret", "Std"]] /= 100
            frames[REP_MODELS[m]] = f.T
        t = pd.concat(frames, axis=1)
        out[lab] = t.loc[["Ret", "Std", "SR"]].map(lambda v: f"{v:.2f}").rename_axis("")      # their decimals, whatever the headers
    return panels(**out)


def r_ckx_costs():
    """their Table 9: net Sharpe ratio of the EW quintile long-short under EWCT, by gamma, models by period"""
    cols = {}
    for period, lab in CKX_PERIODS.items():
        for m, f in ckx("T08_ewct", period, index_col=0).items():
            cols[(lab, REP_MODELS[m])] = f.net_SR
    t = pd.DataFrame(cols)
    t = t.loc[[g for g in t.index if g < 1]]
    t.index = [f"{g:.2f}" for g in t.index]
    return t.rename_axis("$\\gamma$")


def r_ckx_alphas():
    """their Tables 20 (EW) and 21 (VW): coefficients with OLS standard errors below and stars, Constant and Average Ret annualized"""
    star = lambda b, se: "***" if abs(b / se) > 2.576 else "**" if abs(b / se) > 1.96 else "*" if abs(b / se) > 1.645 else ""
    num = lambda v, st="": f"${v:.3f}^{{{st}}}$" if st else f"{v:.3f}"
    out = {}
    for w, lab in [("EW", "Equal-weighted (Table 20)"), ("VW", "Value-weighted (Table 21)")]:
        cols = {}
        for period, plab in [("paper", CKX_PERIODS["paper"]), ("extended", f"2004--{dmrs_end()}")]:
            for m, f in ckx("T20_21_alphas", period, index_col=0).items():
                r, col = f.loc[w], []
                for k in ALPHA_ROWS:
                    col += [num(r[f"b_{k}"], star(r[f"b_{k}"], r[f"se_{k}"])), f"({r[f'se_{k}']:.3f})"]
                a, se = r["alpha_ann_%"] / 100, r["se_alpha_ann_%"] / 100
                col += [num(a, star(a, se)), f"({se:.3f})", num(r["avg_ret_ann_%"] / 100), num(r["adj_R2"])]
                cols[(REP_MODELS[m], plab)] = col
        index = [x for k in ALPHA_ROWS.values() for x in (k, "")] + ["Constant", "", "Average Ret", "Adj R-squared"]
        out[lab] = pd.DataFrame(cols, index=pd.Index(index, name=""))
    return panels(**out)


def r_did_msrr():
    """their Figures 4 and 7 in numbers: the MSRR Sharpe ratio of the news shock under each residualization, of the predicted news,
    of the control, and within the size groups, models by period"""
    cols = {}
    for folder, plab in DID_PERIODS.items():
        size = did("F07_size", folder)
        for m, f in did("F04_msrr", folder).items():
            v = f.set_index(["spec", "portfolio"]).Sharpe
            z = size[m].set_index(["spec", "size"]).Sharpe
            col = {"Raw": v[("Raw", "raw embedding")], **{k: v[(k, "news shock")] for k in ["Constant", "CAPM", "FF3", "FF6", "JKP", "IND", "JKP + IND"]},
                   "Predicted news (JKP)": v[("JKP", "predicted news")], "JKP characteristics (control)": v[("JKP characteristics (control)", "control")],
                   "Mega + Large (JKP)": z[("JKP", "Mega + Large")], "Small + Micro (JKP)": z[("JKP", "Small + Micro")]}
            cols[(REP_MODELS[m], plab)] = pd.Series(col)
    return pd.DataFrame(cols).rename_axis("")


R7_DECIMALS = {"t_alpha": 1, "SR": 2}                                    # their Table 7: returns and alphas 3 decimals


def r_did_sorts():
    """their Table 7: quintiles Low ... High and H-L, equal- and capped value-weighted; Panel A the MSE forecast, Panel B the MSRR
    weights; one panel per model and approach, replication period"""
    out = {}
    order = ["Low (L)", "2", "3", "4", "High (H)", "H-L"]
    for m, f in did("T07_sorts", "didisheim").items():
        for panel in ["MSE", "MSRR"]:
            d = f[f.panel == panel].set_index(["weighting", "portfolio"])[["avg", "std", "alpha", "t_alpha", "SR"]]
            t = d.unstack("weighting").swaplevel(axis=1).reindex(order)
            t = t[[(w, c) for w in ["EW", "capped VW"] for c in ["avg", "std", "alpha", "t_alpha", "SR"]]]
            t = t.apply(lambda s: s.map(lambda v: f"{v:.{R7_DECIMALS.get(s.name[1], 3)}f}"))
            t.columns = pd.MultiIndex.from_tuples([({"EW": "Equal-weighted", "capped VW": "Capped value-weighted"}[w],
                                                    {"avg": "Avg", "std": "Std", "alpha": "Alpha", "t_alpha": "t-stat", "SR": "SR"}[c]) for w, c in t.columns])
            out[f"{REP_MODELS[m]}, {panel}"] = t.rename_axis("")
    return panels(**out)


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def word(n):
    """a small count in words"""
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


TRAIN_SPAN = f"$T-{TRAIN_YEARS + VAL_YEARS}$ through $T-{VAL_YEARS + 1}$"                 # the ridge's training years
VAL_SPAN = " and ".join(f"$T-{k}$" for k in range(VAL_YEARS, 0, -1))                       # its validation years
PROJ_SPAN = f"the up to {word(PROJECTION_YEARS)} years before each test year"              # lib.ridge.firm_years
DEC_SPAN = f"the up to {word(PROJECTION_YEARS)} calendar years ending at each year end"         # lib.firms.december_windows


CUT_NAMES = " and ".join(f"{SHORT[m]} ({pd.Period(c, 'M').strftime('%B %Y')})" for m, c in PRETRAIN_CUTOFF.items())
CUT_LINES = "Dashed lines mark the pretraining cutoffs of " + CUT_NAMES + "."
THREE = "RoBERTa-large, Llama-2 (13B), and ChronoGPT (December 1999 vintage)"
SIZE_DAILY = "Micro, small, and big stocks are below the NYSE 20th capitalization percentile, between it and the NYSE median, and at or above the median, on the preceding trading day."
SIZE_JKP = "Size groups are JKP's at the signal month end: micro stocks are below the NYSE 20th capitalization percentile, small stocks between it and the NYSE median, and big stocks above the median."
JUMP = "from the preceding close or, for news published during the trading session (09:30--16:00), from that session's open"


def tests(benchmark):
    return "DM $t$ and JK $z$ test rank-IC and Sharpe-ratio differences from " + benchmark + " \\citep{diebold2002comparing,jobson1981performance,memmel2003performance}."


# ---- the list ---------------------------------------------------------------------------------------------------------------------


CAP_REPDATA = ("Headline-level ridge forecasts from RoBERTa-large and Llama-2 (13B) embeddings of Dow Jones RavenPack headlines with an event category are averaged within firm-days. Stocks are US common shares at or above the NYSE 5th capitalization percentile on the previous day.")
CKX_SPANS = " and ".join(CKX_PERIODS.values())
SEG_MIN = str(report.SEGMENT_MIN_MONTHS)


# ---- the notes' shared sentences ----------------------------------------------------------------------------------------------
def level_short(what):
    """the historical level of a forecast or an embedding, one sentence"""
    return "The historical level is the mean of the firm's " + what + " over its news months in the preceding " + str(LEVEL_WINDOW) + " months; the deviation is the remainder."


LEVEL_FORECAST, LEVEL_EMBEDDING = level_short("monthly average forecast"), level_short("monthly average embedding")
NOTE_IC = "Rank IC is the mean daily Spearman correlation with next open-to-open returns, in percent."
NOTE_SR = "SR is the annualized Sharpe ratio of equal-weighted (EW) and value-weighted (VW) decile long-short portfolios."
NOTE_RESID_BUILD = "Panel B adds to the eight-year category mean a ridge forecast of the return minus its category's training-window mean, estimated on headline embeddings minus their category's training-window mean embedding"
NOTE_HORIZON_PANELS = "Panel A uses the daily ridge forecast, Panel B its headline-weighted monthly average, and Panel C a ridge forecast fitted to monthly average embeddings."
NOTE_FIG_SPLIT = "Upper panels separate historical level and deviation; lower panels show the category, firm, and idiosyncratic components of the two-way projection, the first and last split into level and deviation. Diamonds show the total, to which the components sum."
POST_FOR = " and ".join(f"{y}--{TEST_YEARS[-1]} for {MODELS[m]}" for m, y in POST_YEAR.items())   # the C tables' evaluation years
POST_EVAL = "evaluation starts the January after each pretraining cutoff, covering " + POST_FOR
LEVEL_COMPONENTS = ("Historical levels average the category and idiosyncratic components' monthly means over the firm's news months in the preceding "
                    + str(LEVEL_WINDOW) + " months; firm identity lies entirely in the level, and deviations are the remainders.")
MSRR_FULL = "maximum Sharpe ratio regression (MSRR)"

TABLES = [
    ("paper", "tab1", "Return Forecasts from Event Categories and Headline Embeddings", tab1,
     "This table reports out-of-sample forecasts of next open-to-open returns from news classifications and headline embeddings over " + str(YEARS) + ". Category forecasts are mean returns over the one or eight years before test year $T$, shrunk toward their event group's mean with weight $n/(n+" + str(CATEGORY_MEAN_SHRINKAGE) + ")$ on the category mean, where $n$ is the category's headline count. Embedding forecasts use ridge regressions on RoBERTa-large and Llama-2 (13B) embeddings, trained on " + TRAIN_SPAN + " and validated on " + VAL_SPAN + ". Coarser forecasts are eight-year means by RavenPack event group or RavenPack type, headline direction from a fixed word rule, or group by direction. LMMD is the Loughran--McDonald positive-minus-negative word count divided by the headline's word count. ESS is RavenPack's event sentiment score. Rank IC is the mean daily Spearman correlation with returns, in percent, with Newey--West $t$-statistics (" + str(NW_LAGS_DAILY) + " lags). SR is the annualized Sharpe ratio of equal-weighted and value-weighted decile long-short portfolios, with value weights from previous-day capitalization. Net SR deducts 10 basis points per 100% turnover, or 20 below the NYSE 20th capitalization percentile, when $\\gamma$ of the gap between drifted holdings and the target portfolio is traded each day."),
    ("paper", "tab2", "Return Forecasts Combining Categories and Embeddings", tab2,
     "This table reports the contribution of embeddings beyond event-category forecasts of next open-to-open returns over " + str(YEARS) + ". Ridge estimation for test year $T$ uses " + TRAIN_SPAN + " and validation " + VAL_SPAN + ". Panel A reports daily Fama--MacBeth regressions of returns on signals standardized within day; coefficients are in basis points per standard deviation, with Newey--West $t$-statistics (" + str(NW_LAGS_DAILY) + " lags) in parentheses. The group $\\times$ direction mean is the eight-year mean by RavenPack event group and headline direction. " + NOTE_RESID_BUILD + ". No combination weight is estimated; the ridge penalty maximizes the validation rank IC of the combined forecast. Rank IC is the mean daily Spearman correlation with next open-to-open returns, in percent. DM $t$ and JK $z$ test rank-IC and Sharpe-ratio differences from the category benchmark. OOS $R^2$, in basis points, is measured against a zero forecast; its $t$-statistic tests the mean daily difference in summed squared errors from the category benchmark. " + NOTE_SR),
    ("paper", "tab3", "Explaining Earnings Surprises and Predicting Next-Day Returns", tab3,
     "This table reports how well earnings headlines explain earnings surprises and predict next-day returns, " + str(SURPRISE['eval'][0]) + "--" + str(SURPRISE['eval'][1]) + ". Headlines arrive between a timed IBES announcement and 09:00 New York time on its assigned trading day. SUE is actual earnings minus consensus, divided by the previous day's price, winsorized at the 1st and 99th percentiles and expressed as a percentile rank among the announcements of the preceding 12 months. $K$ denotes category dummies, $\\hat y$ the CKX forecast, and $E$ the headline embedding. $E_L$ is the firm's historical level, the mean of its monthly average embeddings over its news months in the preceding " + str(LEVEL_WINDOW) + " months, and $E_D=E-E_L$; in $K+E_D+E_L$ they have separate coefficients and ridge penalties. All specifications use the same headlines; coefficients use all earlier years, and ridge penalties are chosen on the preceding year. Panel A reports out-of-sample $R^2$ relative to the training mean and Spearman correlations, averaged over test years. Panel B's $R^2$ is relative to a zero forecast, and its rank IC is the mean daily Spearman correlation with next open-to-open returns, in percent."),
    ("paper", "tab4", "Valuations and Growth Forecasts Explained by Firm Embeddings", tab4,
     "This table reports $R^2$, in percent, on held-out firms from cross-sectional ridge regressions of year-end valuations and growth forecasts on firm-level news embeddings and firm characteristics, " + str(YEARS) + ". Outcomes are log price-to-book and log price-to-earnings, the latter for positive earnings, and IBES median long-term growth forecasts (LTG). News inputs use the headlines of up to six calendar years through the year end. The two-way category--firm projection of their embeddings gives firm identity $F$; $C^{\\mathrm{mix}}$ is the firm's mean category component and $E^{\\mathrm{avg}}$ its average embedding. Controls are Fama--French 49 industry indicators, 60-month CAPM beta, log book equity, asset growth, gross profits to assets, and dividends to assets. Each embedding input, the characteristics, and the industry indicators have their own ridge penalty, chosen by leave-one-out error. Each of three fixed company folds is held out in turn. Panel A fits on the other two folds; Panel B, which starts in " + str(TEST_YEARS[0] + 1) + ", uses slopes fitted on those folds one year earlier and refits only the intercept. $R^2$ is measured against the held-out fold mean and averaged over folds and years."),
    ("paper", "tab5", "Firm Networks and Industry Classifications", tab5,
     "This table reports adjusted $R^2$, in percent, from pooled regressions of year-end firm characteristics on their mean over each firm's peers, with year fixed effects, " + str(YEARS) + ". Industry peers share the firm's SIC-3, NAICS-4, or GICS-6 code. Embedding peers are the ten most similar firms under each signal, where similarity is cosine similarity minus the mean of the two firms' median cosine similarities with all other firms. Signals use the headlines of " + DEC_SPAN + ": the two-way category--firm projection gives firm identity $F$; $C^{\\mathrm{mix}}$ is the firm's mean category component and $E^{\\mathrm{avg}}$ its average embedding. Two-signal rows use two peer means: the ten most similar firms and the next ten for $E^{\\mathrm{avg}}$, and each signal's ten for $F$, $C^{\\mathrm{mix}}$. OI is operating income before depreciation; $\\Delta$Sales is one-year sales growth; market $\\beta$ is 60-month CAPM beta; asset $\\beta$ is market $\\beta$ times the ratio of market equity to market equity plus debt. For each characteristic, all rows use the same firm-years. Overlap is the percentage of a firm's peers in its own industry."),
    ("paper", "tab6", "Decomposition of the Embedding Forecast", tab6,
     "This table decomposes embedding forecasts of next open-to-open returns over " + str(YEARS) + ". A two-way least-squares projection of the headline embeddings of " + PROJ_SPAN + " on category and firm indicators gives category effects ($C$), firm effects ($F$), and the idiosyncratic component ($I$). $F$ is the firm effect for firms with at least " + word(MIN_STORIES) + " headlines in the projection window, spread over at least " + word(MIN_FIRM_YEARS) + " of its calendar years, and zero for other firms. Applying the forecast's ridge coefficients gives $\\hat y-\\bar y=\\hat y^C+\\hat y^F+\\hat y^I$, where $\\bar y$ is the ridge intercept. Covariance shares divide each component's covariance with returns by the forecast's; variance shares divide its variance by the forecast's. Moments pool firm-days after removing day means. Covariance shares sum to 100; variance shares need not. Standalone columns use each component as a signal. " + NOTE_IC + " " + NOTE_SR),
    ("paper", "tab7", "Forecast Covariance with Returns across Horizons", tab7,
     "This table reports forecast covariances with returns across horizons for signal years " + str(YEARS) + ". Panel A uses the daily ridge forecast, Panel B its headline-weighted monthly average, and Panel C a ridge forecast fitted to monthly average embeddings with the daily forecast's training and validation years and penalties selected by monthly rank IC. Daily $h=0$ is the nontradable return to the eligible open and $h\\ge1$ the $h$-th subsequent open-to-open return; monthly $h$ counts months after the signal month. The historical level is the mean of the firm's monthly average forecast over its news months in the preceding " + str(LEVEL_WINDOW) + " months, computed with the current coefficients; without such months it is the forecast from the firm effect of a two-way category--firm projection of headline embeddings, or zero when the firm has $F=0$. The deviation is the remainder. Cov is the cross-sectional covariance times $10^6$, pooled across firm-days or firm-months; Share is the percentage of the forecast's covariance, omitted when the latter has $|t|<2$. Newey--West $t$-statistics use " + str(NW_LAGS_DAILY) + " daily or " + str(NW_LAGS_MONTHLY) + " monthly lags."),
    ("paper", "tab8", "Decomposition of the Residualized-Embedding Strategy", tab8,
     "This table decomposes the returns of the residualized-embedding strategy for signal months " + str(YEARS) + ". As in DKPT, firm-month embeddings are recursively standardized and residualized each month on 132 JKP characteristics, and managed portfolios are combined out of sample by maximum Sharpe ratio regression (MSRR). News is assigned by arrival month, and $h$ denotes the $h$-th subsequent return month; every horizon uses the signal month's $h=1$ MSRR coefficients. A two-way projection of headline embeddings on category and firm indicators gives category effects ($C$), firm effects ($F$), and the idiosyncratic component ($I$). Historical levels average the category and idiosyncratic components' monthly means over the firm's news months in the preceding " + str(LEVEL_WINDOW) + " months; firm identity lies entirely in the level, and deviations are the remainders. Each component is residualized like the full embedding and uses the same coefficients, so component returns sum to the total. Ret is the annualized mean in percent after scaling the strategy to 10% annual volatility. Share is the percentage of the total mean return, omitted when the total has $|t|<2$; SR is annualized. Newey--West $t$-statistics use " + str(NW_LAGS_MONTHLY) + " lags."),
    ("paper", "tab9", "Return Regressions on Lagged Deviations and Current Levels", a_old_news,
     "This table reports monthly Fama--MacBeth regressions of next-month returns on the deviation of month $\\tau-k$ and the historical level of month $\\tau$, for firms with news in both months, " + str(YEARS) + ". " + LEVEL_EMBEDDING + " Each component is standardized and residualized on JKP characteristics, converted into a score by its inner product with the maximum Sharpe ratio regression (MSRR) coefficients of its own signal month, and standardized cross-sectionally. Coefficients are percentage returns per standard deviation of the score, with Newey--West $t$-statistics (" + str(NW_LAGS_MONTHLY) + " lags) in parentheses. News is assigned by arrival month, and the return month is the $(k+1)$-th month after the deviation's month."),
    ("appendix", "tabA_drift", "Returns around News by Event Category", a_drift,
     "This table reports returns around news for categories in the displayed event groups, " + str(YEARS) + ". Jump is the return to the first eligible open, " + JUMP + ". Day 1 is the subsequent open-to-open return; Days 1--10 sums ten such returns. Returns are in basis points, in excess of the mean return of all headlines on the same trading day, and averaged equally across trading days. The sample includes headlines with an observed Jump and a complete ten-day return path. The Headlines column counts each category's headlines, and the table shows categories with at least " + format(DRIFT_TABLE_MIN_STORIES, ',') + " of them. Panel B reports decile long-short portfolios sorted on category and embedding forecasts. Newey--West $t$-statistics use " + str(NW_LAGS_DAILY) + " lags for Jump and Day 1 and 10 for Days 1--10."),
    ("appendix", "tabA_subsamples", "Signal Performance by Subsample", a_subsamples,
     "This table reports category-mean, embedding, and RavenPack event sentiment score (ESS) forecasts of next open-to-open returns across subsamples, " + str(YEARS) + ". " + SIZE_DAILY + " Price-derived news comprises stock prices, technical analysis, order imbalances, indexes, and stock picks. Headline-only and full-article are source formats; all forecasts use headlines. Headlines (%) is the subsample's percentage of all headlines. Panel B reports full-sample differences from the eight-year category mean; DM $t$ and JK $z$ test the rank-IC and Sharpe-ratio differences. " + NOTE_IC + " " + NOTE_SR),
    ("appendix", "tabA_costs", "Portfolio Sharpe Ratios before and after Trading Costs", a_costs,
     "This table reports annualized Sharpe ratios before and after trading costs of daily equal- and value-weighted decile long-short portfolios held from open to open, " + str(YEARS) + ". Signals are the one- and eight-year category means, the embedding ridge forecasts, RavenPack's event sentiment score (ESS), Loughran--McDonald sentiment (LMMD), and eight-year means by RavenPack event group, RavenPack type, headline direction from a fixed word rule, or group by direction. Rebalancing follows $w_t=(1-\\gamma)\\tilde w_{t-1}+\\gamma w_t^*$, where $\\tilde w_{t-1}$ is the drifted holding and $w_t^*$ the target. Costs per 100% turnover are 10 basis points for stocks at or above the NYSE 20th capitalization percentile and 20 basis points below it. Turnover is half the sum of absolute changes in the long-short weights; holdings drift with each stock's return relative to the long-short return."),
    ("appendix", "tabA_windows", "Category Means over Alternative Estimation Windows", a_windows,
     "This table reports category forecasts of next open-to-open returns from category means over the one, three, or eight years preceding test year $T$, an expanding window from 2000 through $T-1$, or an expanding window through $T-3$, on the same firm-days over " + str(YEARS) + ". Category means shrink toward the mean return of their event group with weight $n/(n+" + str(CATEGORY_MEAN_SHRINKAGE) + ")$ on the category mean, where $n$ is the category's headline count in the window; a category absent from the window takes its group's mean. " + NOTE_IC + " L--S is the mean daily decile long-short return in basis points. " + NOTE_SR),
    ("appendix", "tabA_mz", "Mincer--Zarnowitz Slopes and Correlation with Category Benchmarks", a_mz,
     "This table reports Mincer--Zarnowitz slopes and category-related variation in forecasts, " + str(YEARS) + ". Slopes come from daily regressions of next open-to-open returns on forecasts in return units; $t(b=1)$ tests a unit slope. No slope is reported for RavenPack's event sentiment score, which is not a return forecast. Variance explained by category is the headline-level $R^2$ on category indicators, in percent. The final columns report firm-day correlations with the eight-year category mean and with the eight-year mean by RavenPack event group and headline direction."),
    ("appendix", "tabA_spanning", "Spanning Regressions", a_spanning,
     "This table regresses daily decile long-short returns on the indicated benchmark portfolio, alone or with the Fama--French five factors and the momentum, short-term reversal, and long-term reversal factors, " + str(YEARS) + ". Category mean is the eight-year category mean. Test and benchmark portfolios use the same weighting, equal (EW) or value (VW). Benchmark-only regressions use open-to-open returns; factor regressions follow CKX in valuing both portfolios at the close to match factor timing while keeping the trading rule. $\\beta$ is the benchmark loading, $\\alpha$ is in basis points per day, and Appraisal is annualized alpha divided by residual standard deviation. Newey--West $t$-statistics use 5 lags."),
    ("appendix", "tabA_rewrites", "Forecast and Embedding Changes under Headline Rewrites", a_rewrites, lambda: ("This table reports embedding and forecast changes under rewrites of earnings headlines, with the ridge coefficients fixed, " + str(YEARS) + ". Rewrites multiply or divide reported numbers by ten, reverse an unambiguous directional expression, swap the firm name for that of the firm closest in capitalization in the same six-digit GICS industry, or remove the name and ticker. Forecast changes are in standard deviations of the test year's headline-level forecasts. Signed changes are positive when the forecast moves in the direction the rewrite implies, and Correct sign is the percentage moving that way. Both use only magnitude rewrites and direction flips of headlines whose wording has an unambiguous direction; Absolute forecast changes use all rewrites of each variant. Rank changes are the mean percentile-point change of the firm-day forecast (Absolute) and the percentage switching deciles (Decile). Cosine compares original and rewritten embeddings. In subspace is the squared projection of the normalized mean embedding change onto the leading " + str(SUBSPACE) + " directions of the training-window category centroids or firm effects, the two magnitude rewrites sharing one pooled direction; random-direction benchmarks are " + " and ".join(f"{100 * SUBSPACE / d:.1f}% for {lab}" for lab, d in dims().items()) + ".")),
    ("appendix", "tabA_centroid", "Centroid Decomposition of the Embedding Forecast", a_centroid,
     "This table decomposes the forecast of next open-to-open returns as $\\hat y-\\bar y=\\beta'\\bar E_c+\\beta'(E-\\bar E_c)$ with the ridge coefficients fixed, " + str(YEARS) + ". $E$ is the centered headline embedding and $\\bar E_c$ its mean over the training-window headlines of the category, zero for categories without any. Covariance shares divide each component's covariance with returns by the forecast's, and variance shares divide its variance by the forecast's; moments pool firm-days after removing day means. Covariance shares sum to 100; variance shares need not. Standalone columns use each component as a signal. " + NOTE_IC + " " + NOTE_SR),
    ("appendix", "tabA_probe", "Forecasts Retrained on Templates and Centroids", a_probe,
     "This table reports ridge forecasts of next open-to-open returns fitted separately to headline embeddings $E$, category templates $E_c^{\\mathrm{tpl}}$, embeddings minus templates $E-E_c^{\\mathrm{tpl}}$, category centroids $\\bar E_c$, and within-category embeddings $E-\\bar E_c$, " + str(YEARS) + ". Templates are the sentences ``Company'' followed by the RavenPack category name, such as ``Company analyst ratings change positive,'' embedded like the headlines. A category's centroid is the mean embedding of its headlines in training years " + TRAIN_SPAN + "; unseen categories use the pooled mean. Each input uses the daily ridge's training and validation years and its own validation-selected penalty. DM $t$ and JK $z$ test rank-IC and Sharpe-ratio differences from the headline forecast. Rank IC is the mean daily Spearman correlation with next open-to-open returns, in percent. " + NOTE_SR),
    ("appendix", "tabA_decomposition", "Content and Timing Decomposition of the Embedding Forecast", a_decomposition,
     "This table reports the content and timing decomposition of embedding forecasts of next open-to-open returns, " + str(YEARS) + ". A two-way projection of the headline embeddings of " + PROJ_SPAN + " on category and firm indicators gives category ($C$), firm ($F$), and idiosyncratic ($I$) components, and the forecast's ridge coefficients give $\\hat y-\\bar y=\\hat y^C+\\hat y^F+\\hat y^I$, where $\\bar y$ is the ridge intercept. $F$ is the firm effect for firms with at least " + word(MIN_STORIES) + " headlines in the projection window, spread over at least " + word(MIN_FIRM_YEARS) + " of its calendar years, and zero for other firms. The first block splits the category and idiosyncratic components into historical levels, which average their monthly means over the firm's news months in the preceding " + str(LEVEL_WINDOW) + " months, and deviations; firm identity lies entirely in the level. The second block instead splits the idiosyncratic component into the firm's mean over its headlines of the preceding 20 calendar days, $\\hat y^I_H$, and the remainder. Shares divide each component's covariance with returns, or its variance, by the forecast's. Moments pool firm-days after removing day means. Standalone columns use each component as a signal. " + NOTE_IC + " " + NOTE_SR),
    ("appendix", "tabA_factors", "Factor Alphas of Forecast-Component Portfolios", a_factors,
     "This table regresses daily equal- and value-weighted decile long-short returns on the Fama--French five factors and the momentum, short-term reversal, and long-term reversal factors, " + str(YEARS) + ". Portfolios are sorted on the daily ridge forecast, on its category, firm, or idiosyncratic component from a two-way category--firm projection, or on its historical level $\\hat y_L$ or deviation $\\hat y_D$. The historical level is the mean of the firm's monthly average forecast over its news months in the preceding " + str(LEVEL_WINDOW) + " months, computed with the current coefficients. Without such months it is the forecast from the projection's firm effect, or zero when the firm has $F=0$. The deviation is the remainder. Following CKX, portfolios keep the open-to-open trading rule but are valued at the close to match factor timing. $\\alpha$ is annualized in percent; RMW and HML are profitability and value loadings. Newey--West $t$-statistics use 5 lags."),
    ("appendix", "tabA_spectrum", "Variance and Predictive Covariance by Principal-Component Band", a_spectrum,
     "This table decomposes embedding variance and the forecast's covariance with next open-to-open returns across principal-component bands, ordered by decreasing training eigenvalue, " + str(YEARS) + ". The first columns give each band's percentage of the total. $\\bar\\phi$ averages the ridge shrinkage factor $\\lambda_k/(\\lambda_k+\\lambda)$ within the band, where $\\lambda_k$ is the $k$th training eigenvalue and $\\lambda$ the ridge penalty. Band-share columns give each component's percentage of the band's variance or covariance; PC $k$ contributes its ridge coefficient $w_k$ times the covariance of the component's coordinate with returns. Panel A uses category ($C$), firm ($F$), and idiosyncratic ($I$) components from a two-way projection on the ridge training window, " + TRAIN_SPAN + ". Panel B separates cells of RavenPack event group by headline direction into group centroids (Group), deviations of cell centroids from them (Direction), and within-cell residuals (Within cell). Variance and shrinkage average test years; covariance pools firm-days after removing day means."),
    ("appendix", "tabA_groups", "Forecast Decomposition by Event Group", a_groups,
     "This table reports forecasts of next open-to-open returns within event groups with at least " + format(MIN_GROUP_STORIES, ',') + " headlines, " + str(YEARS) + ". Rank IC is the mean daily Spearman correlation across the group's firm-days, in percent. Content shares use the category, firm, and idiosyncratic components of a two-way category--firm projection; timing shares use the historical level and the deviation from it. The historical level is the mean of the firm's monthly average forecast over its news months in the preceding 12 months. The category mean is the eight-year mean. Shares divide component covariance by forecast covariance within the group and are omitted when the latter has $|t|<2$, with 5 Newey--West lags. Contribution to CKX L--S is the group's contribution to the equal-weighted decile long-short return, in basis points per day; contributions sum to the portfolio return across all groups, including those omitted."),
    ("appendix", "tabA_horizons", "Return Covariances of the Full Forecast Decomposition", a_horizons,
     "This table reports return covariances of the category level and deviation, firm identity, and the idiosyncratic level and deviation, for signal years " + str(YEARS) + ". " + NOTE_HORIZON_PANELS + " Category, firm, and idiosyncratic components come from a two-way category--firm projection of headline embeddings. Historical levels average the category and idiosyncratic components' monthly means over the firm's news months in the preceding " + str(LEVEL_WINDOW) + " months; deviations are the remainders, and firm identity lies entirely in the level. Daily $h=0$ is the nontradable return to the eligible open and $h\\ge1$ the $h$-th subsequent open-to-open return; monthly $h$ counts months after the signal month. Cov is the cross-sectional covariance times $10^6$; Share is the percentage of the forecast's covariance, omitted when the latter has $|t|<2$. Newey--West $t$-statistics use " + str(NW_LAGS_DAILY) + " daily or " + str(NW_LAGS_MONTHLY) + " monthly lags."),
    ("appendix", "tabA_reoptimized", "MSRR Strategy Decomposition with Re-estimated Weights", a_reoptimized,
     "This table decomposes monthly " + MSRR_FULL + " strategy returns with coefficients re-estimated at each horizon, for signal years " + str(YEARS) + ". Firm-month embeddings are recursively standardized, residualized each month on 132 JKP characteristics, and combined out of sample using MSRR estimated on signal months from " + str(FIRST_YEAR) + ". News is assigned by arrival month, and $h$ denotes the $h$-th subsequent return month. Panel A separates the historical level, the mean of the firm's monthly average embedding over its news months in the preceding " + str(LEVEL_WINDOW) + " months, from the deviation; Panel B also separates the category, firm, and idiosyncratic components of the two-way category--firm projection. Each component is residualized like the full embedding and uses the same coefficients, so component returns sum to the total. Ret is the annualized mean in percent after scaling the strategy to 10% annual volatility at each horizon. Share is the percentage of the total mean return, omitted when the total has $|t|<2$; SR is annualized. Newey--West $t$-statistics use " + str(NW_LAGS_MONTHLY) + " lags."),
    ("appendix", "tabA_constructions", "MSRR Strategy Sharpe Ratios under Alternative Inputs", lambda: a_constructions(industry=True), "This table reports annualized Sharpe ratios of monthly " + MSRR_FULL + " strategies fitted to alternative text inputs, for signal years " + str(YEARS) + ". Inputs are the embedding, its historical level and deviation, and the level and firm identity lagged " + str(LEVEL_LAG) + " months, where $\\tau$ is the signal month. " + LEVEL_EMBEDDING + " Lagged firm identity is the firm effect of the two-way category--firm projection of headline embeddings three test years earlier. Inputs are residualized each month on JKP characteristics; rows with an industry control also include GICS-4 industry indicators in the residualization. The level, the lagged level, and lagged firm identity are traded only on the firm-months where they exist. MSRR is fitted separately for each input and horizon on signal months from " + str(TEST_YEARS[0]) + ", the first " + str(MSRR_INIT) + " months initializing it. News is assigned by arrival month, and $h$ denotes the $h$-th subsequent return month."),
    ("robustness", "tabC_post_residual", "Category and Embedding Forecasts after the Pretraining Cutoffs", lambda: tab2(True),
     "This table reports the contribution of embeddings beyond category forecasts of next open-to-open returns after pretraining. " + "Construction as in Table~\\ref{tab:tab2}, except that " + POST_EVAL + ". Panel B's category row uses RoBERTa-large's period, and each model's comparisons use its own period."),
    ("robustness", "tabC_post_split", "Forecast Decomposition after the Pretraining Cutoffs", lambda: tab6("post"),
     "This table decomposes daily forecasts of next open-to-open returns after pretraining. " + "Construction as in Table~\\ref{tab:tab6}, except that " + POST_EVAL + "."),
    ("robustness", "tabC_post_horizons", "Forecast Return Covariances after the Pretraining Cutoffs",
     lambda: horizon_panels(report.TIMING_PARTS, True, "post"),
     "This table reports forecast covariances with returns after pretraining. " + "Construction as in Table~\\ref{tab:tab7}, except that " + POST_EVAL + ". Monthly horizons require at least " + SEG_MIN + " signal months."),
    ("robustness", "tabC_post_attribution", "MSRR Strategy Return Decomposition after the Pretraining Cutoffs",
     lambda: attribution(name="attribution_post"), lambda: ("This table decomposes monthly MSRR strategy returns after pretraining, with " + " and ".join(str(months_h1(m, "attribution_post")) for m in MODELS) + " signal months at $h=1$. Construction as in Table~\\ref{tab:tab8}, except that " + POST_EVAL + ", and the strategy is scaled to 10% annual volatility in each model's evaluation sample. Horizons with fewer than " + SEG_MIN + " signal months are omitted.")),
    ("robustness", "tabC_post_constructions", "Alternative MSRR Strategy Inputs after the Pretraining Cutoffs",
     lambda: a_constructions("Sharpe_post"),
     "This table reports annualized Sharpe ratios of monthly MSRR strategies fitted to alternative text inputs after pretraining. Construction as in the upper panels of Figure~\\ref{fig:fig4_inputs}, except that " + POST_EVAL + ". News is assigned by arrival month, and $h$ denotes the $h$-th subsequent return month. Horizons with fewer than " + SEG_MIN + " signal months are blank."),
    ("robustness", "tabC_post_representation", "Valuations and Growth Forecasts after the Pretraining Cutoffs",
     lambda: a_representation(POST_YEAR),
     "This table reports $R^2$ on held-out firms from cross-sectional regressions of year-end valuations and growth forecasts on firm-level embeddings of news published after each model's pretraining cutoff and firm characteristics. Construction as in Table~\\ref{tab:tab4}, except for the inputs shown and that year ends are " + POST_FOR + ", and news windows start no earlier than the January after the cutoff. Panel B starts one year later."),
    ("robustness", "tabC_post_networks", "Firm Networks after the Pretraining Cutoffs", b_networks,
     "This table reports adjusted $R^2$, in percent, from pooled regressions with year fixed effects of year-end firm characteristics on their mean over each firm's peers: the firms sharing its industry code, or the ten firms most similar in firm identity $F$, mean category component $C^{\\mathrm{mix}}$, or average embedding $E^{\\mathrm{avg}}$. Construction as in Table~\\ref{tab:tab5}, except that news windows start no earlier than the January after each model's pretraining cutoff: Panel A uses RoBERTa-large for year ends in 2020--2025, and Panel B uses Llama-2 (13B) for year ends in 2023--2025."),
    ("paper", "tab10", "Forecast, MSRR Strategy, and Valuation Results by Encoder",
     lambda: panels(**{f"Signal years {YEARS}": report.encoder_summary(R), f"Signal years {SEG}": report.encoder_summary(R, "post")}),
     "This table reports the main results for ChronoGPT, RoBERTa-large, and Llama-2 (13B), whose pretraining ends in December 1999, July 2019, and September 2022. The coefficient is the Fama--MacBeth coefficient on the forecast controlling for the eight-year category mean, in basis points per standard deviation, with its Newey--West $t$-statistic (" + str(NW_LAGS_DAILY) + " lags) in parentheses. " + NOTE_IC + " SR is the annualized Sharpe ratio. Daily-forecast portfolios are equal-weighted (EW) or value-weighted (VW) decile long-short portfolios. Covariance shares divide each component's covariance with returns by the forecast's. Maximum Sharpe ratio regression (MSRR) strategies use JKP-residualized embeddings and apply each signal month's $h=1$ coefficients at $h=12$, while the lagged-level strategy is fitted separately. The historical level is the mean of the firm's monthly average embedding over its news months in the preceding " + str(LEVEL_WINDOW) + " months, and the lagged level is " + str(LEVEL_LAG) + " months older. Level shares are percentages of the strategy's mean return. Entries are blank at horizons with fewer than " + SEG_MIN + " signal months, and level shares also when the strategy's mean return has $|t|<2$. Firm-identity $R^2$ is the $R^2$ of log price-to-book on held-out firms, with its increment over controls for Fama--French 49 industries, 60-month CAPM beta, log book equity, asset growth, gross profits to assets, and dividends to assets. The $F$-network rows report the adjusted $R^2$ of log price-to-book on its mean over the ten most similar firms under $F$, and its difference from SIC-3 on the same firm-years. Valuation and network windows cover the up to " + word(PROJECTION_YEARS) + " years through each December, no earlier than " + str(FIRST_YEAR) + " in Panel A and " + str(SEGMENTS[ROBUST]['post'][0]) + " in Panel B."),
    ("paper", "tab11", "Rank ICs across the Llama-2 Pretraining Cutoff by Firm Size and News Intensity", c_cutoff_groups,
     "This table reports mean daily cross-sectional Spearman correlations (rank ICs), in percent, of the historical level and the deviation of the daily ridge forecast with next open-to-open returns for RoBERTa-large and Llama-2 (13B), before and from January 2023, the first calendar year after Llama-2's September 2022 pretraining cutoff. Forecasts are averaged within firm-days with a return to the eligible open and ten subsequent open-to-open returns, on the trading days both encoders share. " + LEVEL_FORECAST + " Micro, small, and big stocks are below the NYSE 20th capitalization percentile, between it and the NYSE median, and at or above the median. News intensity is the firm's number of sample headlines in the 12 calendar months before the trading day's month; terciles are formed each day. The relative IC ratio divides Llama-2's ratio of its later to its earlier mean IC by the same ratio for RoBERTa-large. Brackets report 90% moving-block bootstrap intervals with " + str(report.CUTOFF_BLOCK) + "-day blocks drawn within each period. A relative IC ratio is omitted when Llama-2's earlier mean IC or RoBERTa-large's later one has a Newey--West $|t|<2$."),
    ("chrono", "tabB_residual", "Category and Embedding Forecasts with ChronoGPT$_{1999}$", c1,
     "This table reports the contribution of ChronoGPT embeddings beyond category forecasts of next open-to-open returns in signal years " + str(YEARS) + " and " + SEG + ". Construction as in Table~\\ref{tab:tab2}, except for the encoder. The " + SEG + " results evaluate the same forecasts. ChronoGPT is the December 1999 vintage of \\citet{he2026chronologically}; embeddings average final-layer representations over non-padding tokens."),
    ("chrono", "tabB_split", "Forecast Decomposition with ChronoGPT$_{1999}$", lambda: by_segment(lambda s: tab6(s, ROBUST_MODELS)),
     "This table decomposes daily forecasts of next open-to-open returns from the December 1999 ChronoGPT vintage in signal years " + str(YEARS) + " and " + SEG + ". Construction as in Table~\\ref{tab:tab6}, except for the encoder. The " + SEG + " results evaluate the same forecasts."),
    ("chrono", "tabB_horizons", "Forecast Return Covariances with ChronoGPT$_{1999}$",
     lambda: by_segment(lambda s: horizon_panels(report.TIMING_PARTS, True, s, ROBUST_MODELS)),
     "This table reports forecast covariances with returns across horizons for the December 1999 ChronoGPT vintage in signal years " + str(YEARS) + " and " + SEG + ". Construction as in Table~\\ref{tab:tab7}, except for the encoder. The " + SEG + " results evaluate the same forecasts. Monthly horizons require at least " + SEG_MIN + " signal months."),
    ("chrono", "tabB_attribution", "MSRR Strategy Return Decomposition with ChronoGPT$_{1999}$",
     lambda: by_segment(lambda s: attribution(name=f"attribution_{s}", encoders=ROBUST_MODELS)), lambda: ("This table decomposes monthly MSRR strategy returns for the December 1999 ChronoGPT vintage in signal years " + str(YEARS) + " and " + SEG + ", with " + " and ".join(str(months_h1(ROBUST, f"attribution_{s}")) for s in SEGMENTS[ROBUST]) + " signal months at $h=1$. Construction as in Table~\\ref{tab:tab8}, except for the encoder. The " + SEG + " results evaluate the same strategy, scaled to 10% annual volatility within that sample. Horizons with fewer than " + SEG_MIN + " signal months are omitted.")),
    ("chrono", "tabB_constructions", "Alternative MSRR Strategy Inputs with ChronoGPT$_{1999}$",
     lambda: by_segment(lambda s: a_constructions(f"Sharpe_{s}", ROBUST_MODELS)),
     "This table reports annualized Sharpe ratios of monthly MSRR strategies fitted to alternative text inputs from the December 1999 ChronoGPT vintage in signal years " + str(YEARS) + " and " + SEG + ". Construction as in the upper panels of Figure~\\ref{fig:fig4_inputs}, except for the encoder. The " + SEG + " results evaluate the same strategies. News is assigned by arrival month, and $h$ denotes the $h$-th subsequent return month. Horizons with fewer than " + SEG_MIN + " signal months are blank."),
    ("chrono", "tabB_representation", "Valuations and Growth Forecasts with ChronoGPT$_{1999}$", lambda: a_representation(encoders=ROBUST_MODELS),
     "This table reports $R^2$ on held-out firms from cross-sectional regressions of year-end valuations and growth forecasts on firm-level news embeddings from the December 1999 ChronoGPT vintage and firm characteristics, " + str(YEARS) + ". Construction as in Table~\\ref{tab:tab4}, except for the encoder and the inputs shown. News windows start no earlier than " + str(FIRST_YEAR) + "."),
    ("chrono", "tabB_networks", "Firm Networks with ChronoGPT$_{1999}$", c_networks,
     "This table reports adjusted $R^2$, in percent, from pooled regressions with year fixed effects of year-end firm characteristics on their mean over each firm's peers: the firms sharing its industry code, or the ten firms most similar in firm identity $F$, mean category component $C^{\\mathrm{mix}}$, or average embedding $E^{\\mathrm{avg}}$ from the December 1999 ChronoGPT vintage. Construction as in Table~\\ref{tab:tab5}, except for the encoder. Panel A covers year ends in " + str(YEARS) + " with news windows starting no earlier than " + str(FIRST_YEAR) + ", and Panel B covers year ends in " + SEG + " with news windows starting no earlier than " + SEG[:4] + "."),
    ("replication", "tabR1_ckx_corr", "CKX Replication: Return Prediction Correlations", r_ckx_corr,
     "This table reports forecast-return correlations in the CKX replication, multiplied by 100: the mean daily Pearson correlation, the pooled firm-day correlation, the mean daily correlation among stocks at or above the NYSE median capitalization, and the mean daily correlation by year. Panels cover " + CKX_SPANS + "; the latter lists the years from 2019. Headlines name one company with relevance 100. Data begin in " + str(FIRST_YEAR) + ", so 2004--2007 have two to five training years. " + CAP_REPDATA),
    ("replication", "tabR2_ckx_dm", "CKX Replication: Forecast Comparisons and Strategy Correlations", r_ckx_dm,
     "This table compares the CKX replication forecasts of the two models over " + CKX_SPANS + ". Diebold--Mariano statistics \\citep{diebold2002comparing}, with " + str(NW_LAGS_DAILY) + " Newey--West lags, compare squared errors on common firm-days in the top or bottom forecast quintile of either model. Positive values favor Llama-2; bold values exceed the one-sided 5% critical value. Strategy correlations use daily equal- and value-weighted quintile long-short returns. " + CAP_REPDATA),
    ("replication", "tabR3_ckx_sorts", "CKX Replication: Portfolios Sorted on Return Forecasts", r_ckx_sorts,
     "This table reports daily quintile portfolios in the CKX replication over " + CKX_SPANS + ". Portfolios use equal or previous-day capitalization weights and hold stocks from open to open. Long and Short are the returns of the top- and bottom-quintile stocks. Ret and Std are annualized means and standard deviations in decimals; SR is annualized. " + CAP_REPDATA),
    ("replication", "tabR4_ckx_costs", "CKX Replication: Sharpe Ratios after Trading Costs", r_ckx_costs,
     "This table reports net annualized Sharpe ratios of equal-weighted quintile long-short portfolios in the CKX replication, " + CKX_SPANS + ". Each day, $\\gamma$ of the gap between drifted holdings and the target portfolio is traded. Costs per 100% turnover are 10 basis points for stocks at or above the NYSE 20th capitalization percentile and 20 basis points below it. " + CAP_REPDATA),
    ("replication", "tabR5_ckx_alphas", "CKX Replication: Factor Regressions of Portfolio Returns", r_ckx_alphas,
     "This table regresses CKX replication quintile long-short returns, valued at the close, on hedged DMRS factors, $q^5$ factors, and the momentum (UMD), short-term reversal (STREV), and long-term reversal (LTREV) factors. Samples cover " + CKX_PERIODS["paper"] + " and 2004--" + str(dmrs_end()) + ", the end of the DMRS factors. Constant and Average Ret are annualized decimals; parentheses contain OLS standard errors. One, two, and three stars denote $p<0.10$, $p<0.05$, and $p<0.01$. " + CAP_REPDATA),
    ("replication", "tabR6_did_msrr", "DKPT Replication: News Portfolio Sharpe Ratios", r_did_msrr,
     "This table reports annualized MSRR Sharpe ratios in the DKPT replication on data for " + " and ".join(DID_PERIODS.values()) + "; the first " + str(MSRR_INIT) + " months initialize MSRR. Raw uses recursively standardized firm-month embeddings. News shocks residualize them each month on a constant; beta (CAPM); beta, size, and book-to-market (FF3); these plus profitability, investment, and momentum (FF6); the 132 JKP characteristics with the highest coverage over the sample period, rank-standardized each month across all JKP stocks, as in DKPT; GICS industry groups (IND); or JKP plus IND. Predicted news uses the fitted values under JKP; the control uses the JKP ranks, demeaned among the month's firms with news, in place of embeddings. The size rows use JKP news shocks within JKP's mega and large or small and micro groups. Each month's sample has the firm-months with news and a next-month JKP return. MSRR uses leave-one-out ridge. News is assigned by arrival month. The replication substitutes Dow Jones headlines and RoBERTa-large and Llama-2 (13B) for Reuters articles and E5-Mistral-7B."),
    ("replication", "tabR7_did_sorts", "DKPT Replication: News-Shock Portfolio Sorts", r_did_sorts,
     "This table reports monthly DKPT replication quintiles sorted on mean-squared-error (MSE) return forecasts or MSRR weights, 2003--2022. News shocks are recursively standardized firm-month embeddings residualized each month on JKP characteristics. The MSE forecast is an expanding-window ridge on unwinsorized excess returns, refitted annually after 36 months, with generalized cross-validation. Portfolios use equal weights or capitalization weights capped at the NYSE 80th percentile, in months with at least " + str(MIN_MONTH_STOCKS) + " stocks. Avg, Std, and CAPM Alpha of excess returns are annualized decimals; alpha $t$-statistics use " + str(NW_LAGS_MONTHLY) + " Newey--West lags, and SR is annualized. All panels use the MSE forecast months. News is assigned by arrival month."),
]


def main():
    shutil.rmtree(OUT, ignore_errors=True)                               # no table of an earlier run survives
    OUT.mkdir(parents=True)
    chrono = (ROOT / f"Data/clean/run_pred_{ROBUST}.json").exists()      # Table 10 and the B tables once the robustness encoder's forecasts exist
    written = []
    for kind, label, title, build, note in TABLES:
        if (kind == "chrono" or label == "tab10") and not chrono:
            continue
        note = note() if callable(note) else note                         # a note that quotes results
        (OUT / f"{label}.tex").write_text(environment(label, title, note, build()))
        written.append((kind, label))
    if not chrono:                                                       # one page in their place
        (OUT / "tabB_pending.tex").write_text("\\begin{center}\\textbf{Table 10 and Tables B1--B7 (ChronoGPT): not available in this run}\\\\[1ex]\n"
                                             f"The forecasts of {ROBUST} do not exist yet.\\end{{center}}\n")
        written.append(("chrono", "tabB_pending"))
    master = []
    for kind in ["paper", "appendix", "chrono", "robustness", "replication"]:
        master.append(f"% ---- {kind} ----")
        master += [f"\\input{{tex/{label}.tex}}" for k, label in written if k == kind]
    (OUT / "tables.tex").write_text("\n".join(master) + "\n")
    print(f"{len(written)} tables written to {OUT}")
    if not compile_exhibits(written):                                         # an acceptance run must not pass silently
        sys.exit(1)


CKX_FIG, DID_FIG = "../replication/ckx/figures/", "../replication/didisheim/figures/"   # the replication figures, relative to Results/paper
FIG_PANELS = "Left panels show mean daily Spearman correlations with next open-to-open returns, multiplied by 100; right panels show annualized equal-weighted decile long-short Sharpe ratios."


FIG_DKPT = "The DKPT replication uses Dow Jones headlines with RoBERTa-large and Llama-2 (13B) on data for " + DID_PERIODS["didisheim"] + "; the first " + str(MSRR_INIT) + " months initialize MSRR. Firm-month embeddings are recursively standardized, and portfolios use out-of-sample MSRR."
FIGURES = {   # exhibits.pdf: the title and note of every figure of REVIEW; a pattern [abc] has one title and one note per panel figure
    "figA_stability": ("Annual Performance of the Main Signals",
                       "This figure reports annual performance of one- and eight-year category means, RoBERTa-large and Llama-2 ridge forecasts, and RavenPack event sentiment, " + str(YEARS) + ". Forecasts are averaged within firm-days. " + FIG_PANELS),
    "figA_stability_coarse": ("Annual Performance of Coarser Classification Signals",
                              "This figure reports annual performance of forecasts from coarser RavenPack classifications and headline direction, and of Loughran--McDonald dictionary sentiment, " + str(YEARS) + ", with the eight-year category mean as benchmark. Direction is positive, negative, or neutral under a fixed word rule. Sentiment is the positive-minus-negative dictionary word count divided by the headline's word count. " + FIG_PANELS),
    "fig1_drift_*": ("Cumulative Returns after News by Event Category",
                     "This figure plots cumulative returns after news by event category, " + str(YEARS) + ", one panel per event group. The horizontal axis counts trading days after the first eligible open. Curves sum subsequent open-to-open returns, in basis points, in excess of the mean return of all headlines on the same trading day, averaged equally across trading days. The sample includes headlines with an observed initial return and a complete " + word(H) + "-day return path, and each plotted category has at least " + format(DRIFT_MIN_STORIES, ",") + " such headlines. Each legend entry gives the category's number of headlines, in thousands, and its jump, the initial excess return to the eligible open " + JUMP + ". Curves start at zero at the eligible open and exclude the jump. Shading shows 95% Newey--West confidence intervals with $\\max(" + str(NW_LAGS_DAILY) + ",h)$ lags at day $h$."),
    "fig2[abc]": (("Daily Forecast Covariance with Returns across Horizons", "Monthly Return Covariances of the Averaged Daily Forecast",
                   "Monthly Forecast Covariance with Returns across Horizons"),
                  ("This figure decomposes the covariance of the daily ridge forecast with returns across trading-day horizons, for signal years " + str(YEARS) + ". Horizon $h$ is the $h$-th open-to-open return after the news. The historical level is the mean of the firm's monthly average forecast over its news months in the preceding " + str(LEVEL_WINDOW) + " months, computed with the current coefficients; without such months it is the forecast from the firm effect $F$ of a two-way category--firm projection of headline embeddings, or zero when $F=0$. The deviation is the remainder. Upper panels separate level and deviation; lower panels show the category, firm, and idiosyncratic components of this projection, the first and last split into level and deviation, with firm identity entirely in the level. Diamonds show the total, to which the components sum. Covariances are cross-sectional, pooled across firm-days, and multiplied by $10^6$. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_DAILY) + " lags.",
"This figure decomposes the covariance of the headline-weighted monthly average of daily forecasts with returns across monthly horizons, for signal years " + str(YEARS) + ". News is assigned by trading month; horizons count months after the signal month, $h=1,\\ldots," + str(len(TAUS_FIG)) + "$. " + LEVEL_FORECAST + " " + NOTE_FIG_SPLIT + " Covariances are cross-sectional, pooled across firm-months, and multiplied by $10^6$. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags.",
"This figure decomposes the covariance of the ridge forecast fitted to monthly average embeddings with returns across monthly horizons, for signal years " + str(YEARS) + ". News is assigned by trading month; horizons count months after the signal month, $h=1,\\ldots," + str(len(TAUS_FIG)) + "$. " + level_short("monthly forecast") + " " + NOTE_FIG_SPLIT + " Covariances are cross-sectional, pooled across firm-months, and multiplied by $10^6$. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags.")),
    "figA_persistence_[abc]": (("Daily Forecast Level and Deviation across Horizons",
                                "Averaged Daily Forecast Level and Deviation across Horizons",
                                "Monthly Forecast Level and Deviation across Horizons"),
                               ("This figure reports the historical level and the deviation of the daily ridge forecast as predictors of returns across trading-day horizons, for signal years " + str(YEARS) + ". Horizon $h$ is the $h$-th open-to-open return after the news. " + LEVEL_FORECAST + " Upper panels report each component's rank IC, the mean cross-sectional Spearman correlation with the horizon-$h$ return, in percent. Lower panels report the two components' coefficients in joint Fama--MacBeth regressions after both are residualized on JKP characteristics, in basis points per standard deviation; the residualization is estimated on the preceding month's firm-days with preceding month-end characteristics and applied to the current month. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_DAILY) + " lags.",
"This figure reports the historical level and the deviation of the headline-weighted monthly average of daily forecasts as predictors of returns across monthly horizons, for signal years " + str(YEARS) + ". News is assigned by trading month; horizons count months after the signal month. " + LEVEL_FORECAST + " Upper panels report each component's rank IC, the mean cross-sectional Spearman correlation with the horizon-$h$ return, in percent. Lower panels report the two components' coefficients in joint Fama--MacBeth regressions after both are residualized cross-sectionally on signal-month-end JKP characteristics, in percent per standard deviation. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags.",
"This figure reports the historical level and the deviation of the ridge forecast fitted to monthly average embeddings as predictors of returns across monthly horizons, for signal years " + str(YEARS) + ". News is assigned by trading month; horizons count months after the signal month. " + level_short("monthly forecast") + " Upper panels report each component's rank IC, the mean cross-sectional Spearman correlation with the horizon-$h$ return, in percent. Lower panels report the two components' coefficients in joint Fama--MacBeth regressions after both are residualized cross-sectionally on signal-month-end JKP characteristics, in percent per standard deviation. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags.")),
    "figA_drift_size": ("Cumulative Returns after News by Firm Size",
                        "This figure plots cumulative returns of decile long-short portfolios sorted on the eight-year category mean within micro, small, and big stocks, " + str(YEARS) + ". " + SIZE_DAILY + " Deciles are formed each day among the group's firm-days. The horizontal axis counts trading days after the first eligible open, and curves sum subsequent open-to-open returns, in basis points, averaged equally within deciles and across trading days. The sample includes headlines with an observed initial return and a complete " + str(SIZE_H) + "-day return path. Legends report the jump, the initial return to the eligible open " + JUMP + ". Curves start at zero at the eligible open and exclude the jump. Shading shows 95% Newey--West confidence intervals with $\\max(5,h)$ lags at day $h$."),
    "figA_bridge_size": ("Averaged Daily Forecast Decomposition by Firm Size",
                         "This figure reports the covariance of each component of the headline-weighted monthly average of daily forecasts with returns $h$ months after the signal month, $h=1,\\ldots," + str(len(TAUS_FIG)) + "$, within micro, small, and big stocks, for signal years " + str(YEARS) + ". Components are the category level and deviation, firm identity, and the idiosyncratic level and deviation of the two-way category--firm projection. " + LEVEL_COMPONENTS + " " + SIZE_JKP + " News is assigned by trading month. Covariances are cross-sectional within each month and group, pooled across firm-months, and multiplied by $10^6$. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags."),
    "figA_attribution_size": ("MSRR Strategy Return Decomposition by Firm Size",
                              "This figure splits each component's contribution to monthly " + MSRR_FULL + " strategy returns by the size group of each firm in the signal month, for signal years " + str(YEARS) + ". Components are the category level and deviation, firm identity, and the idiosyncratic level and deviation of the two-way category--firm projection, each residualized each month on JKP characteristics. " + LEVEL_COMPONENTS + " " + SIZE_JKP + " Each group's contribution sums its firms' returns under the same portfolio coefficients, so the three groups add up to the component's full-sample contribution. News is assigned by arrival month; horizon $h$ is the $h$-th subsequent return month, $h=1,\\ldots," + str(len(TAUS_FIG)) + "$, and every horizon uses the signal month's $h=1$ MSRR coefficients. Returns are annualized percentages after scaling the strategy to 10% annual volatility at each horizon. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags."),
    "fig3_attribution": ("MSRR Strategy Return Decomposition across Horizons",
                         "This figure decomposes monthly " + MSRR_FULL + " strategy returns for signal years " + str(YEARS) + ". Firm-month embeddings are residualized each month on JKP characteristics and combined out of sample using MSRR. News is assigned by arrival month; horizon $h$ is the $h$-th subsequent return month, $h=1,\\ldots," + str(len(TAUS_FIG)) + "$, and every horizon uses the signal month's $h=1$ MSRR coefficients. Components come from the two-way category--firm projection of headline embeddings and are residualized like the embedding. " + LEVEL_COMPONENTS + " Upper panels separate historical level and deviation; lower panels show the category, firm, and idiosyncratic components, the first and last split into level and deviation. Diamonds show the total, to which the components sum. Returns are annualized percentages after scaling the strategy to 10% annual volatility at each horizon, with the same scale for every component. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags."),
    "figA_persistence_strategy": ("MSRR Signal Level and Deviation across Horizons",
                                  "This figure reports the historical level and the deviation of the monthly maximum Sharpe ratio regression (MSRR) signal as predictors of returns across monthly horizons, for signal years " + str(YEARS) + ". Firm-month embeddings are residualized on JKP characteristics and combined out of sample using MSRR. " + LEVEL_EMBEDDING + " Each is residualized like the embedding and projected onto its signal month's $h=1$ MSRR coefficients. News is assigned by arrival month; horizon $h$ is the $h$-th subsequent return month. Upper panels report each component's rank IC, the mean cross-sectional Spearman correlation with the horizon-$h$ return, in percent. Lower panels report the two components' coefficients in joint Fama--MacBeth regressions, in percent per standard deviation. Both panels use the JKP returns of firms with news in the signal month, with or without news in the return month. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags."),
    "figA_old_news": ("Return Regressions on Lagged Deviations across Horizons",
                      "This figure reports the coefficient on the deviation of month $\\tau-k$ in monthly Fama--MacBeth regressions of next-month returns on this deviation and the historical level of month $\\tau$, for firms with news in both months and signal years " + str(YEARS) + ". " + LEVEL_EMBEDDING + " News is assigned by arrival month. Each component is standardized and residualized on JKP characteristics, converted into a score by its inner product with the " + MSRR_FULL + " coefficients of its own signal month, and standardized cross-sectionally; coefficients are percentage returns per standard deviation of the score. The horizontal axis is the return month's distance from the deviation month, $h=k+1$. Filled markers are horizons that are multiples of three months. Shading shows 95% Newey--West confidence intervals with " + str(NW_LAGS_MONTHLY) + " lags."),
    "figA_cutoff_monthly": ("Rolling Rank IC of Monthly MSRR Levels and Deviations",
                            "This figure reports the rank IC of the historical level (upper panel) and deviation (lower panel) of the monthly maximum Sharpe ratio regression (MSRR) signal for " + THREE + ", for signal years " + str(YEARS) + ". News is assigned by arrival month. " + LEVEL_EMBEDDING + " Both are residualized on JKP characteristics and projected onto the signal month's $h=1$ MSRR coefficients, estimated out of sample. Rank IC is the cross-sectional Spearman correlation with the next month's JKP excess return, in percent, averaged over the 36 signal months ending at each date, among the signal months shared by the three encoders. Shading shows 95% Newey--West confidence intervals for RoBERTa-large and Llama-2 with " + str(NW_LAGS_MONTHLY) + " lags within each window. " + CUT_LINES),
    "fig4_inputs": ("MSRR Strategy Sharpe Ratios by Input and Horizon",
                    "This figure reports annualized maximum Sharpe ratio regression (MSRR) Sharpe ratios across inputs and subsequent return months, for signal years " + str(YEARS) + ". Upper panels report strategies on the embedding, its historical level and deviation, the level lagged " + str(LEVEL_LAG) + " months, and firm identity lagged " + str(LEVEL_LAG) + " months, each residualized each month on JKP characteristics. Lower panels report MSRR strategies on the embedding, the historical level, and lagged firm identity, each residualized on JKP characteristics alone (solid lines) or also on GICS-4 industry indicators (dashed lines). " + LEVEL_EMBEDDING + " Lagged firm identity is the firm effect of the two-way category--firm projection of headline embeddings three test years earlier. The level, the lagged level, and lagged firm identity are traded only on the firm-months where they exist. MSRR is fitted separately for each input and horizon on signal months from " + str(TEST_YEARS[0]) + ", the first " + str(MSRR_INIT) + " months initializing it. Shading divides 95% Newey--West confidence intervals of mean returns by sample volatility and is shown for the solid lines."),
    "fig5_cutoff_returns": ("Cumulative MSRR Strategy Returns by Component and Encoder",
                            "This figure plots cumulative returns of the monthly maximum Sharpe ratio regression (MSRR) strategy and of its level and deviation components for " + THREE + ", for signal years " + str(YEARS) + ". News is assigned by arrival month. Firm-month embeddings are residualized each month on JKP characteristics and combined out of sample using MSRR; returns are those of the month after the signal month. " + LEVEL_EMBEDDING + " Each component is residualized like the embedding and uses the same coefficients, so component returns sum to the strategy return. Each encoder's strategy is scaled to 10% annual volatility over the plotted months, with the same scale for its components. Cumulative returns sum monthly percentage returns, dated at the end of the signal month, over the signal months shared by the three encoders. " + CUT_LINES),
    "fig6_cutoff": ("Rolling Rank IC of Daily Forecast Levels and Deviations",
                    "This figure reports the rank IC of the level (upper panel) and deviation (lower panel) of the daily ridge forecast for " + THREE + ", for signal years " + str(YEARS) + ". Forecasts are averaged within firm-days with a return to the eligible open and ten subsequent open-to-open returns, on the trading days shared by the three encoders. Rank IC is the cross-sectional Spearman correlation with next open-to-open returns, in percent, averaged over the preceding 252 trading days. " + LEVEL_FORECAST + " Shading shows 95% Newey--West confidence intervals for RoBERTa-large and Llama-2 with 40 lags within each window. " + CUT_LINES),
    "figA_cutoff_returns_daily": ("Cumulative Returns of Daily Forecast Portfolios by Component and Encoder",
                                  "This figure plots cumulative returns of a daily long-short portfolio formed on the ridge forecast and of its historical level and deviation components for " + THREE + ", for signal years " + str(YEARS) + ". Portfolio weights are the forecast minus its cross-sectional mean on the day, divided by the sample's average number of firms per day, so the next open-to-open return splits into level and deviation components. " + LEVEL_FORECAST + " Each encoder's portfolio is scaled to 10% annual volatility over the plotted days, with the same scale for its components. Cumulative returns sum daily percentage returns over the trading days shared by the three encoders. " + CUT_LINES),
    "figA_spectrum": ("Cumulative Variance and Predictive Covariance by Principal Direction",
                      "This figure plots cumulative embedding variance and forecast-return covariance across principal directions, ordered by decreasing training variance on a logarithmic axis, " + str(YEARS) + ". Components are category effects ($C$), firm effects ($F$), and idiosyncratic components ($I$) from a two-way projection on the ridge training window, and All is the full embedding (upper panels) or the forecast (lower panels). Upper panels normalize each component and the full embedding by its own total variance and average test years. Lower panels normalize all contributions by the forecast's covariance with next open-to-open returns, pooling firm-days after removing day means. Dotted lines separate principal-component bands."),
    CKX_FIG + "fig5_cumulative": ("CKX Replication: Cumulative Portfolio Returns",
                                  "This figure plots cumulative log returns in the CKX replication, 2004--" + str(TEST_YEARS[-1]) + ". Stocks are sorted on RoBERTa-large or Llama-2 ridge forecasts into equal- or value-weighted quintiles and held from open to open; long-short portfolios buy the top quintile and sell the bottom. The market is shown for comparison. The dashed line marks June 2019, the end of CKX's sample."),
    CKX_FIG + "fig10_assimilation": ("CKX Replication: Returns by Firm Size and Trading Day",
                                     "This figure reports average open-to-open returns on trading days 1--6 after portfolio formation in the CKX replication, " + CKX_PERIODS["paper"] + ". Within each size group, stocks are sorted on embedding forecasts into equal-weighted quintiles; long-short composition is fixed across horizons. Day 1 is the first open-to-open holding period. Large stocks are at or above the NYSE median capitalization; small stocks are below it. Returns are in basis points per day. Shading shows 95% Newey--West confidence intervals."),
    DID_FIG + "fig4_msrr": ("DKPT Replication: News Portfolio Performance",
                            "This figure reports annualized Sharpe ratios (left), annualized CAPM alphas with 95% confidence intervals (middle), and cumulative returns of the raw embedding and of predicted news and news shocks under JKP residualization (right). Embeddings are projected each month on the indicated firm characteristics; fitted values are predicted news and residuals are news shocks. Bars distinguish predicted news from news shocks; dashed lines denote raw embeddings. Cumulative returns sum monthly returns of portfolios scaled to 10% annual volatility. " + FIG_DKPT),
    DID_FIG + "fig7_size": ("DKPT Replication: News Portfolios by Firm Size",
                            "This figure reports annualized news-shock Sharpe ratios by residualization method (left) and cumulative returns under JKP residualization (right). News shocks are the residuals of monthly projections of embeddings on the indicated firm characteristics. Size groups are JKP's: mega and large stocks, above the NYSE median capitalization, and small and micro stocks, below it and without nano caps. Cumulative-return portfolios are scaled to 10% annual volatility. " + FIG_DKPT),
    DID_FIG + "fig10_delay": ("DKPT Replication: Returns and Alphas by Trading Delay",
                              "This figure reports annualized mean returns and CAPM alphas of JKP-residualized news-shock portfolios delayed by 1--36 months. MSRR coefficients are those of the undelayed strategy in the signal month. Each portfolio is scaled to 10% annual volatility; returns and alphas are in decimals. Shading shows 95% Newey--West confidence intervals. " + FIG_DKPT),
}

REVIEW = ["tab[1-9]", "tab10", "tab11", "fig1_drift_*", "fig2a", "fig3_attribution", "fig4_inputs", "fig5_cutoff_returns", "fig6_cutoff",
          "tabA_*", "figA_stability", "figA_stability_coarse", "figA_drift_size", "figA_spectrum", "fig2b", "fig2c",
          "figA_persistence_a", "figA_persistence_b", "figA_persistence_c", "figA_bridge_size", "figA_attribution_size", "figA_persistence_strategy",
          "figA_old_news", "figA_cutoff_monthly", "figA_cutoff_returns_daily",
          "tabB*", "tabC*", "tabR1*", "tabR2*", "tabR3*",
          CKX_FIG + "fig5_cumulative", "tabR4*", "tabR5*", CKX_FIG + "fig10_assimilation",
          "tabR6*", DID_FIG + "fig4_msrr", DID_FIG + "fig7_size", "tabR7*", DID_FIG + "fig10_delay"]
APPENDIX_FIGS = {"fig2b", "fig2c"}                                       # paper figures the paper shows in its appendix
PANEL_OF = {"figA_stability_coarse": "figA_stability"}                   # a second panel under the previous figure's number (the paper's Figure A.1)
# the order of exhibits.pdf: the paper's tables (in number order) and figures, the appendix's in the paper's order, Tables B and C, then the replication
FIG_KEY = r"_?(\[abc\]|\*)$"                                              # a figure's label: its REVIEW pattern without the panel part


def figure_entry(item):
    """FIGURES' (title, note) of a REVIEW item; a single panel (fig2b) takes its letter's entry of the pattern (fig2[abc])"""
    if item in FIGURES:
        return FIGURES[item]
    for key, (titles, notes) in FIGURES.items():
        if "[abc]" in key and fnmatch.fnmatch(item, key):
            k = "abc".index(item[-1])
            return titles[k], notes[k]
    raise KeyError(item)


def review_body(written):
    """exhibits.tex's body: the REVIEW items one page each (tables by label, numbered as in the paper: 1, A1, R1; figures by file name
    pattern, several figures two to a row); a written table missing from REVIEW comes last. `written`: (kind, label) in TABLES order"""
    count = {"paper": 0, "appendix": 0, "robustness": 0, "chrono": 0}
    number = {}
    for kind, label in written:
        if kind == "replication":
            number[label] = "R" + re.search(r"R(\d+)", label).group(1)
        else:
            count[kind] += 1
            number[label] = {"paper": "", "appendix": "A", "chrono": "B", "robustness": "C"}[kind] + str(count[kind])
    fig_count = {"": 0, "A": 0, "R": 0}
    table = lambda label: f"\\renewcommand{{\\thetable}}{{{number[label]}}}\n\\input{{tex/{label}.tex}}\n\\clearpage"
    written = [label for _, label in written]
    out, used = [], set()
    for pos, item in enumerate(REVIEW):
        if item.startswith("tab"):
            for label in [w for w in written if fnmatch.fnmatch(w, item) and w not in used]:
                out.append(table(label))
                used.add(label)
            continue
        if item in ("fig5_cutoff_returns", "fig6_cutoff", "figA_cutoff_monthly", "figA_cutoff_returns_daily") and "tabB_pending" in written:   # no robustness encoder in this run
            continue
        figs = [f for f in [R / f"{item}.pdf"] if f.exists()] if "/" in item else sorted((R / "figures").glob(f"{item}.pdf"))
        expected = len(report.INFO_GROUPS) if item == "fig1_drift_*" else 3 if "[abc]" in item else 1
        assert len(figs) == expected, f"figure {item}.pdf of exhibits.pdf: {len(figs)} files, {expected} expected"
        kind = "R" if "/" in item else "A" if item.startswith("figA") or item in APPENDIX_FIGS else ""
        if item in PANEL_OF:                                               # panel b of the previous figure
            fig_no = f"{kind}{fig_count[kind]}b"
        else:
            fig_count[kind] += 1
            nxt = REVIEW[pos + 1] if pos + 1 < len(REVIEW) else None
            fig_no = f"{kind}{fig_count[kind]}" + ("a" if PANEL_OF.get(nxt) == item else "")
        titles, notes = (x if isinstance(x, tuple) else (x,) * len(figs) for x in figure_entry(item))
        notes = [note_names(re.sub(r"(?<!\\)%", r"\\%", n)) for n in notes]
        if "/" not in item:                                                # the label of the figure's number, for the notes' references
            out.append(f"\\renewcommand{{\\thefigure}}{{{fig_no}}}\\refstepcounter{{figure}}\\label{{fig:{re.sub(FIG_KEY, '', item)}}}")
        path = lambda f: item + ".pdf" if "/" in item else "figures/" + f.name
        page = lambda number, files, width, height, title, note: (
            f"\\begin{{center}}\\textbf{{Figure {number}: {title}}}\\end{{center}}\n"
            f"\\begin{{minipage}}{{\\linewidth}}\\footnotesize {note}\\end{{minipage}}\n\\medskip\n\n\\begin{{center}}\n"
            + "\n".join(f"\\includegraphics[width={width}\\linewidth,height={height:.2f}\\textheight,keepaspectratio]{{{path(f)}}}" for f in files)
            + "\n\\end{center}\n\\clearpage")
        if len(figs) > 3:                                                  # many small figures two to a row
            out.append(page(fig_no, figs, 0.48, 0.75 / -(-len(figs) // 2), titles[0], notes[0]))
        else:                                                              # one figure, or its panels a, b, c one per page
            out += [page(fig_no + ("abc"[i] if len(figs) > 1 else ""), [f], 0.85, 0.75, titles[i], notes[i]) for i, f in enumerate(figs)]
    return "\n".join(out + [table(w) for w in written if w not in used])


# the works cited in table notes, entries as in the paper's bib
BIB = r'''@article{hoberg2010product,
  author = {Hoberg, Gerard and Phillips, Gordon},
  title = {Product Market Synergies and Competition in Mergers and Acquisitions: A Text-Based Analysis},
  journal = {Review of Financial Studies},
  volume = {23},
  number = {10},
  pages = {3773--3811},
  year = {2010}
}

@article{hoberg2016text,
  author = {Hoberg, Gerard and Phillips, Gordon},
  title = {Text-Based Network Industries and Endogenous Product Differentiation},
  journal = {Journal of Political Economy},
  volume = {124},
  number = {5},
  pages = {1423--1465},
  year = {2016}
}

@article{sarkar2026economic,
  title={Economic representations},
  author={Sarkar, Suproteem K},
  journal={Available at SSRN 6999938},
  year={2026}
}

@article{he2026chronologically,
  title={Chronologically consistent large language models},
  author={He, Songrun and Lv, Linying and Manela, Asaf and Wu, Jimmy},
  journal={Journal of Financial Economics},
  volume={185},
  pages={104374},
  year={2026},
  publisher={Elsevier}
}

@incollection{mincer1969evaluation,
  title={The Evaluation of Economic Forecasts},
  author={Mincer, Jacob A and Zarnowitz, Victor},
  booktitle={Economic Forecasts and Expectations: Analysis of Forecasting Behavior and Performance},
  editor={Mincer, Jacob A},
  pages={3--46},
  year={1969},
  publisher={NBER}
}

@article{diebold2002comparing,
  title={Comparing predictive accuracy},
  author={Diebold, Francis X and Mariano, Robert S},
  journal={Journal of Business \& economic statistics},
  volume={20},
  number={1},
  pages={134--144},
  year={2002},
  publisher={Taylor \& Francis}
}

@article{loughran2011liability,
  title={When is a liability not a liability? {T}extual analysis, dictionaries, and 10-{K}s},
  author={Loughran, Tim and McDonald, Bill},
  journal={Journal of Finance},
  volume={66},
  number={1},
  pages={35--65},
  year={2011}
}

@article{memmel2003performance,
  title={Performance hypothesis testing with the {Sharpe} ratio},
  author={Memmel, Christoph},
  journal={Available at SSRN 412588},
  year={2003}
}

@article{jobson1981performance,
  title={Performance hypothesis testing with the {Sharpe} and {Treynor} measures},
  author={Jobson, J Dave and Korkie, Bob M},
  journal={Journal of Finance},
  pages={889--908},
  year={1981},
  publisher={JSTOR}
}
'''


def compile_exhibits(written):
    """compile tex/exhibits.tex (article on letter paper with 1in margins, booktabs, amsmath, adjustbox, graphicx, rotating, natbib with
    tex/refs.bib), the tables and figures in REVIEW order, from Results/paper"""
    (OUT / "refs.bib").write_text(BIB)
    (OUT / "exhibits.tex").write_text("\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{booktabs,amsmath,amssymb,adjustbox,graphicx,rotating,natbib}\n"
                                   "\\begin{document}\n" + review_body(written) + "\n\\bibliographystyle{plainnat}\n\\bibliography{refs}\n\\end{document}\n")
    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        print("pdflatex or bibtex not found: tex/exhibits.tex not compiled")
        return False
    for i in range(3):                                                   # bibtex after the first pass, then twice so that citations and references resolve
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", "tex", "tex/exhibits.tex"], cwd=R, capture_output=True, text=True)
        if p.returncode != 0:                                            # a failed pass is reported below from exhibits.log
            break
        if i == 0:
            b = subprocess.run(["bibtex", "exhibits"], cwd=OUT, capture_output=True, text=True)
            if b.returncode != 0:
                print("bibtex FAILED: " + str(OUT / "exhibits.blg"))
                for l in (b.stdout + b.stderr).splitlines()[-20:]:
                    print("  ", l)
                return False
    log = (OUT / "exhibits.log").read_text(errors="ignore") if (OUT / "exhibits.log").exists() else p.stdout
    problems = [l for l in log.splitlines() if l.startswith("!") or "undefined" in l.lower() or "Float too large" in l]
    problems += [l for l in log.splitlines() if (m := re.match(r"Overfull \\hbox \((\d+\.?\d*)pt too wide", l)) and float(m.group(1)) > OVERFULL_PT]
    ok = p.returncode == 0 and not problems
    print(f"pdflatex {'OK' if ok else 'FAILED'}: {OUT / 'exhibits.pdf'}")
    for l in problems[:20]:
        print("  ", l)
    return ok


if __name__ == "__main__":
    main()
