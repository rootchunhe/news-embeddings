"""Do news-embedding firm networks group firms with similar characteristics, as industry classifications do (Hoberg and Phillips 2010,
2016)? In the window of the December of every year T from the first year `--first` (the news of the PROJECTION_YEARS years up to T;
lib.firms.december_windows), the firms of Table 4 (a firm effect, positive book equity, complete controls) with SIC, NAICS (at least
four digits) and GICS codes and a firm effect in every encoder's window (lib.ridge.identified). A pair's similarity is the cosine of the two firms' vectors
minus the mean of the two firms' median cosine with all other firms (Hoberg and Phillips 2016), for each encoder of `--encoders` and each
firm-level input of the window's two-way projection: the average centered embedding E^avg, the firm component F and the category mix
C^mix (the firm's mean category component). A firm's peers under a signal are its PEERS firms of highest similarity (Hoberg and Phillips
2010: the ten most similar rivals); under an industry classification, the firms sharing its SIC-3, NAICS-4 or GICS-6 code. An industry
control is the equal-weighted mean of a characteristic over a set of peers, the firm excluded. Rows (ROWS): each classification and each
signal with one control; E^avg with two controls (its PEERS most similar firms and the next PEERS); F and C^mix with two controls (each
signal's PEERS most similar firms). A firm with a zero vector under any signal (no cosine) leaves every comparison of its December
(counted in run.json). Characteristics in that December, winsorized 1/99 within the December: Hoberg and Phillips' OIBDP / sales,
OIBDP / assets, sales growth, market beta and asset beta (market beta x ME / (ME + debt), debt beta zero) and Table 4's outcomes. Pooled
over the Decembers: the adjusted R2 of the characteristic on the row's controls with December fixed effects, on the firm-Decembers with
at least one peer under every industry classification (per characteristic, the same firm-Decembers for every row: the characteristic and
all controls observed). Also each row's overlap with each industry classification: the share of a firm's peers (under two controls, of the
union of the two sets) in its SIC-3, NAICS-4 and GICS-6 industry, averaged over firms and Decembers. Run from Codes/3_paper/:
    OMP_NUM_THREADS=8 ../../env/bin/python 2_networks.py --encoders roberta-large,Llama-2-13b-hf --first 2000
Output Results/paper/2_networks/<encoders joined by +>/<first>/r2.csv, overlap.csv + run.json"""
import numpy as np
import pandas as pd

from lib.constants import ENCODERS
from lib.firms import OUTCOMES, decembers, december_windows, firm_inputs
from lib.ridge import WindowFit
from lib.run import finish, save, start, years

from common.config import FIRST_YEAR

CODES = {"SIC-3": lambda c: c.sic.astype("int64").values // 10, "NAICS-4": lambda c: c.naics.astype("int64").astype(str).str[:4].values,
         "GICS-6": lambda c: c.gics.astype("int64").values // 100}
PEERS = 10                                                                  # a firm's peers under a signal: its most similar firms
SIGNALS = ["average", "F", "category mix"]                                  # lib.firms.firm_inputs
ROWS = {"average": ["average"], "F": ["F"], "category mix": ["category mix"], "average, average": ["average", "average next"],
        "F, category mix": ["F", "category mix"]}                           # an encoder's rows: their peer sets, one control each
CHARACTERISTICS = ["ebitda_sale", "ebitda_at", "sale_gr1", "beta_60m", "asset_beta"] + OUTCOMES
JKP = ["gics", "sic", "naics", "ebitda_sale", "ebitda_at", "sale_gr1", "debt_me"]


def section(a):
    return f"2_networks/{'+'.join(encoders(a))}/{a.first}"


def encoders(a):
    e = a.encoders.split(",")
    assert set(e) <= set(ENCODERS), f"unknown encoder in {a.encoders}"
    return e


def adjusted(S):
    """Hoberg and Phillips' median adjustment: each pair's similarity minus the mean of the two firms' median similarity with all
    other firms"""
    M = S.copy()
    np.fill_diagonal(M, np.nan)
    med = np.nanmedian(M, axis=1)
    return S - (med[:, None] + med[None, :]) / 2


def ranked(S, lo, hi):
    """0/1 matrix of each row's other firms ranked lo+1 to hi by similarity (the diagonal excluded)"""
    assert len(S) > hi and np.isfinite(S).all(), "ranked: too few firms or a non-finite similarity"
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1, kind="stable")[:, lo:hi]
    A = np.zeros(S.shape, dtype=bool)
    np.put_along_axis(A, order, True, axis=1)
    return A


def peer_sets(X):
    """{peer set: 0/1 matrix} of one encoder's firm inputs X: each signal's PEERS most similar firms, and E^avg's next PEERS"""
    cos = lambda V: (lambda Z: Z @ Z.T)(V / np.linalg.norm(V, axis=1, keepdims=True))
    S = {k: adjusted(cos(X[k])) for k in SIGNALS}
    out = {k: ranked(S[k], 0, PEERS) for k in SIGNALS}
    out["average next"] = ranked(S["average"], PEERS, 2 * PEERS)
    return out


def peer_mean(A, v):
    """the equal-weighted mean of v over each row's peers (A: 0/1, zero diagonal) with v observed; NaN without any"""
    ok = np.isfinite(v)
    num, den = A @ np.where(ok, v, 0.0), A @ ok.astype(float)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def adj_r2(d, y, xs):
    """adjusted R2 of d[y] on the columns xs with December fixed effects"""
    X = np.column_stack([np.ones(len(d)), pd.get_dummies(d.december, drop_first=True).values.astype(float)] + [d[x].values for x in xs])
    e = d[y].values - X @ np.linalg.lstsq(X, d[y].values, rcond=None)[0]
    n, k = X.shape
    return 1 - (e ** 2).sum() / (n - k) / (((d[y] - d[y].mean()) ** 2).sum() / (n - 1))


def main():
    a, log, out = start(section, model=False, encoders="roberta-large,Llama-2-13b-hf", first=FIRST_YEAR, years="")
    enc = encoders(a)
    D = decembers(JKP)
    D["asset_beta"] = D.beta_60m / (1 + D.debt_me.where(D.debt_me >= 0))
    D["naics"] = pd.to_numeric(D.naics).where(lambda x: x >= 1000)            # JKP codes a missing NAICS as 0; NAICS-4 needs four digits
    D = D.dropna(subset=["gics", "sic", "naics"])
    streams = [december_windows(m, years(a.years), firsts={a.first}) for m in enc]
    rows_of = {("", k): [("", k)] for k in CODES} | {(m, r): [(m, s) for s in sets] for m in enc for r, sets in ROWS.items()}
    controls = list(dict.fromkeys(s for sets in rows_of.values() for s in sets))   # every peer set, one control column each
    panel, overlap, diag, zero = [], [], {}, {}
    for wins in zip(*streams):
        dec = wins[0][1]
        assert all(w[1] == dec for w in wins), "the encoders' windows are not aligned"
        fits = [WindowFit(st, fst) for _, _, st, fst in wins]
        diag |= {f"{m}-{dec}": f.diagnostics(fst, firm_mask=f.has_F) for m, f, (_, _, _, fst) in zip(enc, fits, wins)}
        c = D[D.year == dec]
        for m, f in zip(enc, fits):                                      # the firms with a firm effect in every encoder's window
            keep = f.has_F
            c = c.merge(pd.DataFrame({"permno": f.permno[keep], f"i_{m}": np.where(keep)[0]}), on="permno")
        assert c.permno.is_unique, "a firm twice in a December"
        c = c.sort_values("permno").reset_index(drop=True)                # ties (ranked) in permno order
        X = {m: firm_inputs(f, fst, c[f"i_{m}"].values) for m, (_, _, _, fst), f in zip(enc, wins, fits)}
        ok = np.logical_and.reduce([np.linalg.norm(X[m][k], axis=1) > 0 for m in enc for k in SIGNALS])
        zero[str(dec)] = int((~ok).sum())                                # a zero vector has no cosine: the firm leaves every comparison
        c, X = c[ok].reset_index(drop=True), {m: {k: v[ok] for k, v in x.items()} for m, x in X.items()}
        off = ~np.eye(len(c), dtype=bool)
        A = {("", k): (lambda v: (v[:, None] == v[None, :]) & off)(code(c)) for k, code in CODES.items()}
        for m in enc:
            A |= {(m, k): M for k, M in peer_sets(X[m]).items()}
        common = np.logical_and.reduce([A[("", k)].any(1) for k in CODES])     # the firms with a peer under every industry classification
        for (m, r), sets in rows_of.items():
            U = np.logical_or.reduce([A[s] for s in sets])
            peers = U.sum(1)
            share = {k: np.where(peers > 0, (U & A[("", k)]).sum(1) / np.maximum(peers, 1), np.nan) for k in CODES}
            overlap.append({"december": dec, "encoder": m, "row": r, "peers": peers[common].mean(),
                            **{"overlap_with_" + k: np.nanmean(v[common]) for k, v in share.items()}})
        rows = c.loc[common, ["permno"]].assign(december=dec)
        for x in CHARACTERISTICS:
            v = c[x].astype(float).values
            v = np.clip(v, *np.nanquantile(v, [0.01, 0.99]))
            rows[x] = v[common]
            for (m, k), M in A.items():
                rows[f"{x}|{m}|{k}"] = peer_mean(M.astype(float), v)[common]
        panel.append(rows)
        log(f"December {dec}: {len(c):,} firms, {common.mean():.1%} with a peer under every industry classification")
    P = pd.concat(panel, ignore_index=True)
    res = []
    for x in CHARACTERISTICS:
        d = P.dropna(subset=[x] + [f"{x}|{m}|{k}" for m, k in controls])
        for (m, r), sets in rows_of.items():
            res.append({"characteristic": x, "encoder": m, "row": r, "adj_R2": adj_r2(d, x, [f"{x}|{s[0]}|{s[1]}" for s in sets]),
                        "firm_decembers": len(d)})
    save(out, "r2", pd.DataFrame(res), "adjusted R2 of each characteristic on the row's industry controls")
    save(out, "overlap", pd.DataFrame(overlap))
    print(pd.DataFrame(overlap).groupby(["encoder", "row"], sort=False)[["peers"] + ["overlap_with_" + k for k in CODES]].mean().round(3).to_string())
    finish(out, a, log, decembers=int(P.december.nunique()), firm_decembers=len(P), two_way=diag, zero_vector_firms=zero)


if __name__ == "__main__":
    main()
