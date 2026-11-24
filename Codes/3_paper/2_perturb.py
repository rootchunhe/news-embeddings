"""Rule-based rewrites of structured earnings headlines, scored with the replication's own ridge. Run from Codes/3_paper/:
    ../../env/bin/python 2_perturb.py --build; 2_embed.py --model <model> --texts ../../Data/clean/perturb_texts.parquet (GPU); OMP_NUM_THREADS=8 ../../env/bin/python 2_perturb.py --model roberta-large
Output Data/clean/perturb_texts.parquet + run_perturb_texts.json (--build); Results/paper/2_perturb/<model>/summary.csv + run.json"""
from lib import ROOT
from lib.constants import TEST_YEARS
from lib.run import start, save, finish
from lib.ridge import WindowFit, category_index, replication_fit, rolling_windows
from lib.evaluate import bucket
from lib.panel import forecasts, industry, rewrite_pairs, rewrites, sample_rows

import re

import numpy as np
import pandas as pd

from common.config import MIN_NAMES
from common.emb import load_emb
from common.runinfo import write_manifest

NUMERIC = re.compile(r"\b(EPS|Net|Loss|Loss/Shr|Rev|Revenue|Sales|Profit|Beats|Misses|Tops|Sees)\b", re.I)
# a number with or without thousands separators; not a year (2023), a quarter / half (3Q, 1H), a form number (10-Q, 10-K, 8-K) or a bare fraction ($.50)
NUMBER = re.compile(r"(?<![A-Za-z\d.])(?!\d{4}\b)(?!\d[QH]\b)(?!\d+-[KQ]\b)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\.?\d)(?=[A-Za-z%]*\b|\s|$)")
FLIPS = [("Loss/Shr", "EPS"), ("EPS", "Loss/Shr"), ("Net Loss", "Net"), ("Loss", "Net"), ("Net", "Loss"), ("Profit", "Loss"), ("Rises", "Falls"), ("Falls", "Rises"),
         ("Rise", "Fall"), ("Fall", "Rise"), ("Beats", "Misses"), ("Misses", "Beats"), ("Tops", "Misses"), ("Above", "Below"), ("Below", "Above"), ("Up", "Down"),
         ("Down", "Up"), ("Raises", "Cuts"), ("Cuts", "Raises"), ("Higher", "Lower"), ("Lower", "Higher"), ("Gain", "Loss"), ("Jumps", "Plunges"), ("Plunges", "Jumps")]
NEGATIVE = {"Loss/Shr", "Loss", "Net Loss", "Falls", "Fall", "Misses", "Below", "Down", "Cuts", "Lower", "Plunges"}
POSITIVE = {a for a, _ in FLIPS} - NEGATIVE
DIRECTION = {w.lower() for pair in FLIPS for w in pair}                      # the flipped words: never part of a firm name
AMOUNT = {"Net", "Loss", "Net Loss", "Profit"}      # accounting lines: flipped only before an amount ("Net $5M", not "Net Interest Income")
SCALE = 10                                                             # magnitude rewrites: every number times and divided by SCALE
SIGNED = re.compile(r"(?<![^\s($])-\$?\d|\((?!\d{4}\))\$?\d[\d,.]*[cMBK]?\)")      # a number written with a minus sign or in parentheses (a loss)
COMPARE = re.compile(r"\b(?:vs|versus|ests?|estimates?|consensus|(?:yr|year)-(?:ago|earlier)|prior)\b|\bfrom\b.+\bto\b|\bto\b.+\bfrom\b", re.I)   # a comparison with another figure
NET_PROFIT = re.compile(r"\bNet(?=\s+(?:Income|Profit|Earnings)\b|\s*[-$(]*\d)")   # "Net $5M", "Net Income", not "Net Sales"
UNSIGNED = re.compile(r"\b(?:Drops?|Off|Slips?|Declines?|Narrows?|Widens?|Shrinks?)\b")       # a move whose direction depends on the line
TAG = re.compile(r"\s*>\S+\s*$")
SUFFIX = re.compile(r"(?:,?\s+(?:inc|corp|corporation|co|company|ltd|plc|llc|lp|holdings|group)(?![\w-])\.?)+", re.I)   # legal-form words after the name
PATH_TEXTS = ROOT / "Data/clean/perturb_texts.parquet"


def scale(text, k):
    """every number times k, written back in the number's own style (thousands separators kept)"""
    def rep(m):
        s = m.group(1)
        v = float(s.replace(",", "")) * k
        dec = len(s.partition(".")[2]) + (k < 1)                         # the number's own decimals, one more when divided ("$37.7M" /10 -> "$3.77M")
        r = f"{v:.{dec}f}"
        if k < 1 and r.endswith("0"):                                    # the extra decimal only when needed ("30c" -> "3c", "$1.50" -> "$0.15")
            r = r[:-1].rstrip(".")
        if "," in s:
            ip, _, fp = r.partition(".")
            r = f"{int(ip):,}" + (f".{fp}" if fp else "")
        return r
    return NUMBER.sub(rep, text)


def word(w):
    """the pattern of a direction word to flip"""
    return rf"\b{re.escape(w)}\b" + (r"(?=\s*[-$(]*\d)" if w in AMOUNT else "")


def flip(text):
    """(text with the first direction word found replaced in all its occurrences, whether the replacement is negative), or None when
    the rest of the headline carries another direction word or a number is written as a loss ("Net Loss $5M" -> "Net $5M";
    "Loss/Shr 31c" -> "EPS 31c"); Net, Loss, Net Loss and Profit are flipped only before an amount"""
    if SIGNED.search(text):
        return None
    found = [(a, b) for a, b in FLIPS if re.search(word(a), text)]
    if not found:
        return None
    a, b = found[0]
    rest = re.sub(word(a), "", text)
    if any(re.search(rf"\b{re.escape(x)}\b", rest) for x, _ in FLIPS if x != a):
        return None
    return re.sub(word(a), b, text), b in NEGATIVE


def abbrev(h, e):
    """headline word h is entity word e or Dow Jones' abbreviation of it: same first letter, h's letters in order inside e
    ("Fincl": Financial, "Labs": Laboratories); never a template or direction word ("Net" is no abbreviation of Networks, "Cuts" none of Computers)"""
    h, e = h.lower().strip(".()"), e.lower().strip(".()")
    if h == e:
        return True
    if NUMERIC.fullmatch(h) or h in DIRECTION or len(h) < 2 or len(h) >= len(e) or h[0] != e[0]:
        return False
    it = iter(e)
    return all(c in it for c in h)


def name_parts(text, entity):
    """the firm's name prefix of the headline (longest run of the entity's leading words, then the following words that abbreviate
    the next entity words, then the legal-form words: "Spire Inc 2Q Net" -> "Spire Inc", "South Fincl Grp 1Q Loss" -> "South Fincl Grp")
    and the ticker tag"""
    ws = [w for w in re.split(r"[\s,]+", str(entity)) if w]
    pos = [i for i, w in enumerate(ws) if i == 0 or SUFFIX.fullmatch(" " + w) is None]          # the first word is always the name ("Group 1 Automotive")
    words = [ws[i] for i in pos]
    prefix = ""
    for k in range(min(4, len(words)), 0, -1):
        cand = " ".join(words[:k])
        if text.lower().startswith(cand.lower()) and not text[len(cand):len(cand) + 1].isalnum():     # whole words ("Ameri" is not "American")
            end, j = len(cand), pos[k - 1] + 1
            while j < len(ws):
                m = re.match(r"\s+([^\s,:;]+?)(?=['’]s\b|[\s,:;]|$)", text[end:], re.I)
                if m is None or not abbrev(m.group(1), ws[j]):
                    break
                end, j = end + m.end(), j + 1
            m = SUFFIX.match(text, end)
            prefix = text[:m.end() if m else end]
            break
    tag = TAG.search(text)
    return prefix, (tag.group(0) if tag else "")


def direction_sign(text):
    """+1 for a headline with positive direction words only, -1 with negative ones only, 0 with both or none (a revenue alone),
    with a number written with a minus sign or in parentheses, with a comparison (vs, an estimate, a prior figure, from ... to) or with a
    move such as Drops or Narrows (UNSIGNED); "Net" counts as profit wording only before income, profit, earnings or an amount:
    the direction a rewrite should move the forecast is then unclear"""
    if SIGNED.search(text) or COMPARE.search(text) or UNSIGNED.search(text):
        return 0
    def has(words, t):
        return any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)
    t = re.sub(r"\bNet Loss\b", "", text)
    neg, pos = has(NEGATIVE, text), has(POSITIVE - {"Net"}, t) or bool(NET_PROFIT.search(t))
    return 0 if neg == pos else -1 if neg else 1


def expected_sign(variant, sign):
    """the direction the forecast should move: +1 up, -1 down, 0 none (the name rewrites, a headline without a direction: direction_sign 0)"""
    if variant.startswith("magnitude x"):
        return sign
    if variant.startswith("magnitude /") or variant == "direction flipped":
        return -sign
    return 0


def day_ranks(S, R):
    """ranks (1..n) of the firm-day forecast of each rewritten story among the day's n firm-days before (r0) and after (r1) the rewrite,
    the day's other firm-days unchanged, and n; S: the sample stories with the replication forecast yhat, R: rewrites with permno,
    day0 and the forecast change d_raw. Days with fewer than MIN_NAMES firm-days are left out (NaN)"""
    fd = S.groupby(["day0", "permno"]).yhat.agg(["sum", "count"]).reset_index()
    fd = fd[fd.groupby("day0").permno.transform("size") >= MIN_NAMES].assign(m=lambda d: d["sum"] / d["count"])
    R = R.merge(fd[["day0", "permno", "m", "count"]], on=["day0", "permno"], how="left")
    new = (R.m + R.d_raw / R["count"]).values
    r0, r1, n = np.full(len(R), np.nan), np.full(len(R), np.nan), np.full(len(R), np.nan)
    days = {d: np.sort(m.values) for d, m in fd.groupby("day0").m}
    ok = np.flatnonzero(R.m.notna().values)
    for d, j in R.iloc[ok].groupby("day0").indices.items():
        idx = ok[j]
        x, m = days[d], R.m.values[idx]
        r0[idx] = 1 + np.searchsorted(x, m)                                       # 1 + the other firm-days below the original
        r1[idx] = 1 + np.searchsorted(x, new[idx]) - (m < new[idx])               # the firm's own original is no other
        n[idx] = len(x)
    return r0, r1, n


def nearest_peer(k, rows, lc, permno):
    """the row among `rows` (one industry-year, sorted by log market cap lc) of another firm whose log cap is closest to row k's"""
    j = np.searchsorted(lc[rows], lc[k])
    lo, hi = j - 1, j
    while lo >= 0 and permno[rows[lo]] == permno[k]:
        lo -= 1
    while hi < len(rows) and permno[rows[hi]] == permno[k]:
        hi += 1
    near = [rows[i] for i in (lo, hi) if 0 <= i < len(rows)]
    return min(near, key=lambda r: abs(lc[r] - lc[k])) if near else None


def build(a, log):
    S = sample_rows(TEST_YEARS, ["headline", "entity_name"])
    S = S[S.group == "earnings"].reset_index(drop=True)
    S["headline"] = S.headline.fillna("").astype(str)
    S = S[S.headline.str.contains(NUMERIC) & S.headline.str.contains(NUMBER)]
    parts = S.apply(lambda r: name_parts(r.headline, r.entity_name), axis=1)
    S["prefix"], S["tag"] = [p[0] for p in parts], [p[1] for p in parts]
    S = S[S.prefix.str.len() > 0].reset_index(drop=True)
    S = industry(S)
    S = S[S.ind != "NA"].reset_index(drop=True)                      # every templated earnings headline with a (point-in-time) industry
    lc, permno = np.log(S.cap.values), S.permno.values                 # cap: the market cap of the day before the story
    by_ind = {k: v[np.argsort(lc[v], kind="stable")] for k, v in S.groupby(["ind", "yr"]).indices.items()}   # peers of the same industry and year
    # (a peer's name and cap may come from later in the year: the rewrite is a counterfactual text, no return is involved)
    rows, dropped = [], {}
    for i, r in S.iterrows():
        rest = r.headline[len(r.prefix):]                                  # the rewrites never touch the firm-name prefix ("3M", "7-Eleven", "Net ...")
        base = {"rp_story_id": r.rp_story_id, "permno": int(r.permno), "hid": int(r.hid), "yr": int(r.yr), "category": r.category, "ind": r.ind,
                "sign": direction_sign(rest)}
        variants = [("original", r.headline, False), (f"magnitude x{SCALE}", r.prefix + scale(rest, SCALE), False),
                    (f"magnitude /{SCALE}", r.prefix + scale(rest, 1 / SCALE), False)]
        f = flip(rest)
        if f:
            variants.append(("direction flipped", r.prefix + f[0], f[1]))
        peer = nearest_peer(i, by_ind[(r.ind, r.yr)], lc, permno)
        if peer is not None:
            o = S.iloc[peer]
            variants.append(("name swapped (same industry)", o.prefix + (rest.replace(r.tag, o.tag) if r.tag else rest), False))
        variants.append(("name removed", re.sub(r"^['’]s\b", "", rest.replace(r.tag, "").strip(" ,:-"), flags=re.I).strip(" ,:-"), False))   # "Fluor's 4Q Net" -> "4Q Net"
        for variant, text, neg in variants:
            if variant != "original" and text == r.headline:              # a rewrite that changed nothing is no rewrite
                dropped[variant] = dropped.get(variant, 0) + 1
                continue
            rows.append({**base, "variant": variant, "text": text, "to_negative": neg})
    P = pd.DataFrame(rows)
    fl = P[(P.variant == "direction flipped") & (P.sign != 0)]
    assert (fl.to_negative == (fl.sign == 1)).all(), "a flip whose direction disagrees with the headline's sign"
    P.to_parquet(PATH_TEXTS, index=False)
    write_manifest(PATH_TEXTS.with_name("run_perturb_texts.json"), {k: v for k, v in vars(a).items() if k != "model"}, spec=("paper",),
                   headlines=len(S), texts=len(P), dropped=dropped)                 # model-free: the texts are the same for every model
    log(f"{len(S):,} headlines, {len(P):,} texts -> {PATH_TEXTS}; rewrites equal to the original dropped: {dropped or 'none'}")
    print(P.groupby("variant").text.first().to_string())


def score(a, log, out):
    P, Z = rewrites(a.model)
    _, sd_T, fits = replication_fit(a.model)
    cats = category_index()
    orig = P[P.variant == "original"]
    E0 = load_emb(a.model, orig.hid.values, np.float32)
    Zo = Z[orig.index.values].astype(np.float32)
    cos0 = (E0 * Zo).sum(1) / (np.linalg.norm(E0, axis=1) * np.linalg.norm(Zo, axis=1))
    assert cos0.min() > 0.99, f"re-embedded originals differ from the stored embeddings (min cosine {cos0.min():.4f})"
    log(f"stored embeddings of the originals reproduced by the re-embedding (cosine min {cos0.min():.5f}, mean {cos0.mean():.6f})")
    rows = []
    for T, _, st, _ in rolling_windows(a.model, cats, sorted(P.yr.unique())):
        f = WindowFit(st)
        beta = f.replication_beta(fits[T]).astype(np.float32)
        v, Zv, Zo = rewrite_pairs(P, Z, T)
        d_raw = (f.z(Zv) - f.z(Zo)) @ beta
        rows.append(pd.DataFrame({"rp_story_id": v.rp_story_id.values, "permno": v.permno.values, "variant": v.variant.values, "yr": T,
                                  "e": [expected_sign(k, sg) for k, sg in zip(v.variant, v.sign)], "d_raw": d_raw, "d_yhat_sd": d_raw / float(sd_T[T]),
                                  "cosine": (Zv * Zo).sum(1) / (np.linalg.norm(Zv, axis=1) * np.linalg.norm(Zo, axis=1))}))
        log(f"{T}: {len(v):,} rewrites scored")
    R = pd.concat(rows, ignore_index=True)
    S = forecasts(sample_rows(TEST_YEARS), {"yhat": a.model})
    R = R.merge(S[["rp_story_id", "permno", "day0"]], on=["rp_story_id", "permno"], how="left")
    r0, r1, n = day_ranks(S, R)
    has = R.e != 0
    R = R.assign(d_dir=R.d_yhat_sd * R.e.where(has), correct_sign=(R.d_yhat_sd * R.e > 0).astype(float).where(has), abs_d_yhat_sd=R.d_yhat_sd.abs(),
                 abs_d_rank=100 * np.abs(r1 - r0) / (n - 1),                                  # percentile points
                 decile_change=pd.Series(bucket(r1, n) != bucket(r0, n), index=R.index).astype(float).where(~np.isnan(r0)))   # the legs' deciles (continuous forecasts: no ties)
    summ = R.groupby("variant").agg(n=("d_yhat_sd", "size"), d_yhat_sd=("d_yhat_sd", "mean"), d_dir=("d_dir", "mean"), abs_d_yhat_sd=("abs_d_yhat_sd", "mean"),
                                    correct_sign=("correct_sign", "mean"), abs_d_rank=("abs_d_rank", "mean"), decile_change=("decile_change", "mean"), cosine=("cosine", "mean"))
    summ[["correct_sign", "decile_change"]] *= 100
    save(out, "summary", summ.reset_index(), "rewrites of earnings headlines: forecast change (sd units; d_dir signed by the expected direction, "
         "correct_sign: % in the expected direction), change of the firm-day's percentile in its day (abs_d_rank, decile_change %) and cosine")
    finish(out, a, log, headlines=len(orig), texts=len(P), min_cosine_reembedded=float(cos0.min()))


def main():
    a, log, out = start(lambda a: None if a.build else "2_perturb", build=False)     # the build step writes no results folder
    if a.build:
        build(a, log)
    else:
        score(a, log, out)


if __name__ == "__main__":
    main()
