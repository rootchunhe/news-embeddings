"""Out-of-sample category means: the event type's average next-day return over a window of past years, shrunk towards the event
group's mean with weight n / (n + CATEGORY_MEAN_SHRINKAGE). Only stories whose return was realized before the test year enter
(day1_yr <= T - 1, the replication's rule). The 8-year window is the CKX information window (years T-8..T-1)."""
import re

import numpy as np
import pandas as pd

from lib import ROOT
from lib.constants import HISTORY_YEARS
from lib.panel import sample_rows

from common.config import CATEGORY_MEAN_SHRINKAGE

WINDOWS = {"m1": lambda yrs, T: yrs == T - 1, "m3": lambda yrs, T: (yrs >= T - 3) & (yrs <= T - 1), "m8": lambda yrs, T: (yrs >= T - 8) & (yrs <= T - 1),
           "mexp": lambda yrs, T: yrs <= T - 1, "mold": lambda yrs, T: yrs <= T - 3}
LABELS = {"m8": "Category mean (8-yr)", "m1": "Category mean (1-yr)", "m3": "Category mean (3-yr)", "mexp": "Category mean (expanding)",
          "mold": "Category mean (through T-3)"}
COLS = ["category", "group", "yr", "day1_yr", "y_fit"]
PAIRS = [("up", "down"), ("increase", "decrease"), ("above", "below"), ("upgrade", "downgrade"), ("positive", "negative"), ("buy", "sell"),
         ("bullish", "bearish"), ("raise", "cut"), ("gain", "loss"), ("overbought", "oversold")]   # the antonym modifiers of RavenPack's SUB_TYPE / PROPERTY
POSITIVE, NEGATIVE = {p for p, _ in PAIRS}, {n for _, n in PAIRS}
LM_FILE = "Data/Loughran-McDonald_MasterDictionary_1993-2023.csv"   # the master dictionary of February 2024


def forms(w):
    """the word and its regular inflections: s, es, d, ed, ing; a final e dropped before ing; a short final consonant doubled (cutting)"""
    out = {w, w + "s", w + "es", w + "d", w + "ed", w + "ing"}
    if w.endswith("e"):
        out.add(w[:-1] + "ing")
    if len(w) <= 3 and w[-1] not in "aeiouwy" and w[-2] in "aeiou":
        out |= {w + w[-1] + "ed", w + w[-1] + "ing"}
    return out


FORM = {f: w for w in POSITIVE | NEGATIVE for f in forms(w)}
WORD = re.compile(r"\b(" + "|".join(sorted(FORM, key=len, reverse=True)) + r")\b")


def direction(category):
    """+, - or 0 from the directional modifier of the category label (RavenPack builds the label as TYPE-SUB_TYPE-PROPERTY; the
    modifier words are the antonym PAIRS), 0 for a category without one"""
    toks = set(str(category).split("-"))
    if toks & NEGATIVE:
        return "-"
    return "+" if toks & POSITIVE else "0"


def headline_direction(headline):
    """+, - or 0 from the same antonym words in the headline itself (lower case, with their regular inflections, forms()); no
    RavenPack field enters. A negative word wins, as in direction()"""
    return headline.str.lower().str.findall(WORD).map(lambda ws: "-" if {FORM[w] for w in ws} & NEGATIVE else "+" if ws else "0")


def lm_tone(headline):
    """CKX's dictionary benchmark on the headline: Loughran-McDonald positive minus negative words over all words (a word belongs to a
    list when its column holds the year it was added; a negative year marks a removal)"""
    d = pd.read_csv(ROOT / LM_FILE, usecols=["Word", "Positive", "Negative"]).dropna(subset=["Word"])
    pos, neg = set(d.Word[d.Positive > 0].str.lower()), set(d.Word[d.Negative > 0].str.lower())
    return headline.str.lower().str.findall(r"[a-z]+").map(lambda ws: (sum(w in pos for w in ws) - sum(w in neg for w in ws)) / max(len(ws), 1))


def bucket(group, headline):
    """the coarse event type: RavenPack group x direction word of the headline"""
    return group.astype(str) + "/" + headline_direction(headline)


def history_rows(years=HISTORY_YEARS):
    """the sample stories of the history years (COLS only), to extend the training windows"""
    return sample_rows(years)[COLS]


def category_means(S, name, window, years, history=None):
    """adds column `name`: for every test year T in `years`, the category's mean training target y_fit over the window's years (stories
    realized by T - 1), shrunk towards its event group's mean with weight n_c / (n_c + CATEGORY_MEAN_SHRINKAGE); categories absent from the window
    get their group's mean. `history`: rows (COLS) of earlier years that extend the window."""
    H = pd.concat([S[COLS], history[COLS]], ignore_index=True) if history is not None else S[COLS]
    cat, grp, yrs, realized, y = H.category.values, H.group.values, H.yr.values, H.day1_yr.values, H.y_fit.values
    out = np.full(len(S), np.nan)
    for T in years:
        tr = window(yrs, T) & (realized <= T - 1)
        if tr.sum() == 0:
            continue
        d = pd.DataFrame({"c": cat[tr], "g": grp[tr], "y": y[tr]})
        gm = d.groupby("g").y.mean()
        c2g = d.drop_duplicates("c").set_index("c").g
        g = d.groupby("c").y.agg(["mean", "size"])
        w = g["size"] / (g["size"] + CATEGORY_MEAN_SHRINKAGE)
        m = w * g["mean"] + (1 - w) * c2g.reindex(g.index).map(gm).fillna(d.y.mean())
        te = S.yr.values == T
        out[te] = pd.Series(S.category.values[te]).map(m).fillna(pd.Series(S.group.values[te]).map(gm).fillna(d.y.mean())).values
    S[name] = out
    return S
