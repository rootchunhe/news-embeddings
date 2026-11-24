"""Calibration of the news-embedding framework of main.tex Appendix D ("A Calibrated Framework with Look-Ahead Bias").
Run from FinLLM/Framework/ after cutoff_inputs.py (run from Codes/3_paper/):
    ../env/bin/python calibration.py             # baseline: tables, figures, calibration.json, calibration_fit.csv
    ../env/bin/python calibration.py variants    # Table D5: the joint search under alternative specifications
Reads Results/paper (read only), cutoff_inputs.json and cutoff_daily.csv.gz. Writes next to this file (the paper's tables are
pasted from these by hand):
    cal_params.tex       Table D2, parameters             cal_shares.tex     Table D3, coverage shares
    cal_targets.tex      Table D4, targeted moments       cal_checks.tex     Table 13, statistics outside the targets
    cal_cutoff.tex       Table 12, relative IC ratios     cal_variants.tex   Table D5 (with cal_variants.json)
    cal_fig2a/b/c.pdf, cal_fig3.pdf                       model against data by component and horizon (calibration_figures.py)
    cal_cutoff_ic.pdf    rank ICs around January 2023: level ICs, the level's IC ratio by horizon, placebo splits
    cal_cutoff_placebo.pdf   the same without the level ICs (Appendix D)
    calibration.json     parameters, fit, table entries, the numbers the text quotes (text_numbers), input md5s (provenance)
    calibration_fit.csv  every moment: role (step1, target, check), block, encoder, horizons lo..hi or component, data, se, model
Search units: daily covariances x 1e6; monthly MSRR component returns in %/yr before volatility scaling. Equation numbers refer
to Appendix D."""
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "Results"
CUT_FILE, DAILY_FILE = HERE / "cutoff_inputs.json", HERE / "cutoff_daily.csv.gz"
CUT = json.load(open(CUT_FILE))

# ---- encoders, calendar, constants ---------------------------------------------------------------------------------------
CG, RB, LM = "chrono-gpt-v1-19991231", "roberta-large", "Llama-2-13b-hf"
NAME = {CG: "ChronoGPT", RB: "RoBERTa-large", LM: "Llama-2 (13B)"}         # table column order
CODE = {RB: "R", CG: "C", LM: "L"}                                         # parameter suffixes, in the search's order
MODEL = {v: k for k, v in CODE.items()}
PANELS = (RB, LM)                                                          # the two forecast encoders
FORECAST = {LM: "Llama-2 forecast", RB: "RoBERTa forecast"}                # labels in Results/paper/1_category
CAT = "Category mean (8-yr)"
TD = 21                                                                    # trading days per month
mi = lambda y, m: 12 * y + m - 1                                           # integer month
month_date = lambda t: f"{t // 12}-{t % 12 + 1:02d}-01"
SAMPLE = (mi(2004, 1), mi(2025, 12))                                       # signal months
STANDALONE_START = mi(2005, 1)      # first signal month of the standalone strategies (after MSRR's initial 12 months)
PROJ_MONTHS = 72                    # projection and valuation windows: six years (the first December window has 60 months; we use 72)
LEVEL_MONTHS, LEVEL_LAG = 12, 36    # historical level: the preceding 12 months; lagged level: 36 months earlier
MO_H, DAY_H = 36, 10                # monthly and daily horizons; the 36-month return variance of eq. (D23) uses the longest
NW_DAILY, NW_MONTHLY = 5, 6         # Newey-West lags (common/config.py)
N_TRAIN = CUT["n_train"]            # news items in the ridge's training window (Proposition D1)
N_F = 1 / CUT["firm_valuation"]["inv_n"]                                   # nbar^F, harmonic mean over the valuation windows
SHARES = {f: tuple(CUT["availability"][w][k] for k in ("F", "IL", "ID", "H")) for f, w in (("day", "firm_days"), ("mo", "firm_months"))}

# memory and the pretraining cutoffs (Section D.2)
T_STAR, T_STAR_R, MEM_START = mi(2022, 9), mi(2019, 7), mi(2000, 1)       # Llama-2's and RoBERTa's (CKX) cutoffs; first training month
N_THETA = 360                                                              # memory window, months (Crane, Karra and Soto 2025)
MEM = {"L": (MEM_START, T_STAR), "R": (MEM_START, T_STAR_R)}               # remembered months the forecast uses
SPLITS = {"L": {"treat": "L", "control": "R", "last": SAMPLE[1], "split": mi(2023, 1)},    # eq. (D28), first January after the cutoff
          "R": {"treat": "R", "control": "L", "last": T_STAR, "split": mi(2020, 1)}}       # eq. (D29), through Llama-2's cutoff
CUTOFF_SEED, CUTOFF_BLOCK = 20261109, 40                                   # the bootstrap of Codes/3_paper/report.py
PLACEBO_YEARS, PLACEBO_POST = range(2008, 2017), 3                         # placebo splits each January, three years after each
COMPONENTS = (("yF", "F"), ("lev", "lev"), ("uL", "IL"), ("dev", "dev"), ("cD", "CD"))   # cutoff_daily.csv.gz columns, keys
CUTOFF_TARGETED = ("F", "lev")

# targets (Section D.3): band means; the first horizon only where the model gives it event drift
PARTS = ["cL", "cD", "yF", "uL", "uD"]
DAY_BANDS = {"cD": {1: (1, 1), 2: (2, 10)}, "uD": {1: (1, 1), 2: (2, 10)}, "yF": {1: (2, 10)}, "uL": {1: (2, 10)}}
MO_TARGETS = ["yF", "uL", "uD"]
MO_BANDS = {2: (2, 12), 3: (13, 24), 4: (25, 36)}
SOBOL_N, SOBOL_SEED = 64, 20261115
S2W_TOL = 1e-3                                                             # tolerance of the sigma_theta^2 fixed point
NAMES = ["rnu", "Bz", "Bx", "rho", "a_R", "a_C", "a_L", "q_R", "q_C", "g_R", "g_C", "M_R", "M_C", "M_L", "eta", "etaR"]
X0 = np.array([4.4, 0.03, 0.10, 0.975, 0.89, 0.86, 0.97, 0.15, 0.12, 0.40, 0.35, 5.0, 2.0, 25.0, 1.2, 0.5])
LB = np.r_[[1e-2, 1e-5, 1e-5, 0.5], np.full(7, 1e-3), np.full(3, 1e-3), [0.0, 0.0]]
UB = np.r_[[200, 10, 10, 0.9995], np.full(7, 20.0), np.full(3, 1e5), [200.0, 50.0]]
SOBOL_BOX = {"rnu": (0.3, 30, 1), "Bz": (1e-3, 0.5, 1), "Bx": (5e-3, 1.0, 1), "rho": (0.5, 0.9995, 0), "a_R": (0.2, 4, 1),
             "a_C": (0.2, 4, 1), "a_L": (0.2, 4, 1), "q_R": (0.02, 1.5, 1), "q_C": (0.02, 1.5, 1), "g_R": (0.05, 3, 1),
             "g_C": (0.05, 3, 1), "M_R": (0.3, 60, 1), "M_C": (0.1, 30, 1), "M_L": (0.3, 150, 1), "eta": (0.0, 5.0, 0),
             "etaR": (0.0, 2.0, 0)}                                         # (low, high, log-uniform)
SOBOL_BOX.update({f"{k}_{e}": SOBOL_BOX["M_" + e] for e in "RCL" for k in ("UF", "UL")})   # variant ups_FL
# parameter names: kappa, vt = kappa, vartheta and S the long-short scale (step 1); sigK2 = delta_1^2 Var(m_K) (x 1e6); rnu =
# Var(nu) / Var(m_K); Bz, Bx = pi^2 Var(zbar), pi^2 Var(x) (x 1e6); rho = rho_m; a_e, q_e, g_e the loadings; M_e the strategy scale
# Upsilon of eq. (D17); eta, etaR the memory loadings of Llama-2 and RoBERTa; s2w = sigma_theta^2; nbar; sueK = R2(s|K) of eq. (D7).
# Variant switches carried in the same dict: mem_scale, r2zx, no_shares, UF_e and UL_e (Upsilon_F and Upsilon_IL)


def train_months(T):
    """training months of test year T (lib/ridge.py: years max(T - 8, 2000) .. T - 3)"""
    return np.arange(mi(max(T - 8, 2000), 1), mi(T - 3, 12) + 1)


def lam(m, e):
    """1 for months inside encoder e's memory"""
    m = np.asarray(m)
    lo, hi = MEM[e]
    return ((m >= lo) & (m <= hi)).astype(float)


# consistency of the inputs with the constants
assert all(hi - N_THETA + 1 <= lo for lo, hi in MEM.values())            # both memory windows begin before the first training month
assert mi(PLACEBO_YEARS[-1] + PLACEBO_POST, 1) <= T_STAR_R                 # every placebo sample ends inside RoBERTa's memory
assert sorted(set(CUT["firm_valuation"]["window_months"])) == [60, PROJ_MONTHS]
assert CUT["lagged_breadth"]["months"] == [month_date(STANDALONE_START)[:7], month_date(SAMPLE[1] - 1)[:7]]
assert CUT["return_variance"]["months"] == MO_H
assert all(abs(sum(s[:3]) - 1) < 1e-9 for s in SHARES.values())
# the training months of a test year lie inside memory whenever a remembered return does, so the coefficient's coverage f_e(T)
# (Section D.2) is one wherever it matters and drops out of every memory term
assert all(lam(train_months(m // 12), e).mean() == 1 for e in MEM for m in range(SAMPLE[0], SAMPLE[1] + 1) for h in range(MO_H + 1)
           if lam(m + h, e))


# ---- inputs --------------------------------------------------------------------------------------------------------------
INPUTS = set()                                                             # files read, for the provenance block


def read(path, **kw):
    INPUTS.add(Path(path))
    return pd.read_csv(path, **kw)


@lru_cache(maxsize=None)
def _paper(part, model, name):
    return read(RES / "paper" / part / model / f"{name}.csv")


def paper(part, model, name):
    return _paper(part, model, name).copy()


@lru_cache(maxsize=None)
def attribution(m):
    """DKPT's fixed-weight attribution of encoder m by component, indexed by h months after the signal month"""
    a = paper("4_didisheim/attribution", m, "attribution")
    return {k: g.assign(h=g.tau + 1).set_index("h").sort_index() for k, g in a[a.weights == "fixed"].groupby("part")}


def load():
    """category levels and, for the two forecast encoders, the daily term structure by component (covariances x 1e6)"""
    d = {"levels": read(RES / "paper/1_category/levels.csv").set_index("signal"), "ts": {}}
    for m in PANELS:
        d["ts"][m] = {k: g.set_index("h").sort_index() for k, g in paper("4_horizon", m, "term_structure").groupby("part")}
    return d


def r2_log_pb(m):
    """firm identity's out-of-sample R2 of log P/B over Table 4's December windows (2_representation)"""
    r2 = read(RES / f"paper/2_representation/{m}/r2.csv")
    return float(r2[(r2["sample"] == "split") & (r2.outcome == "log_pb") & (r2.first_year == r2.first_year.min())].groupby("inputs").R2.mean()["F"])


def nw_se(x, lags):
    """Newey-West (Bartlett) standard error of the mean, as common/portfolio.py nw_tstat"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    e = x - x.mean()
    v = e @ e / len(x) + sum(2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / len(x) for l in range(1, lags + 1))
    return float(np.sqrt(v / len(x)))


# ---- the model: pure functions of the parameter dict p (search units; a variant's switches ride in p) ---------------------
def phi(c):
    return c ** 2 / (1 + c ** 2)


def rhobar(rho, n=LEVEL_MONTHS):
    """rho-bar_L: mean of rho_m^l over the level's months l = 1..n"""
    return rho * (1 - rho ** n) / (n * (1 - rho))


@lru_cache(maxsize=4096)
def rhobar_F(rho):
    """rho-bar_F: for news in month j of the test year, the mean of rho_m^l over the projection window's lags j..71+j, averaged over j"""
    return float(np.mean([np.mean(rho ** np.arange(j, PROJ_MONTHS + j)) for j in range(1, 13)]))


def avg_corr(rho, n):
    """mean of rho^|i - j| over i, j < n: the share of x's variance an n-month average keeps"""
    i = np.arange(n)
    return float(np.mean(rho ** np.abs(i[:, None] - i[None, :])))


def drift_month(vt, h):
    """delta-bar_h / delta_1: drift in the h-th month after the signal month relative to day-one drift, geometric decay"""
    return vt ** (TD * (np.asarray(h, float) - 1)) * (1 - vt ** TD) ** 2 / (TD * (1 - vt) ** 2)


def month_reliability(g, n):
    """21^2 phi(sqrt(nbar) g): the monthly state reliability of an encoder (a firm-month averages nbar items), times a month of
    the premium on a month of returns"""
    return TD ** 2 * g ** 2 / (g ** 2 + 1 / n)


def parts(p, e, freq, G=None):
    """state terms by component (daily covariances x 1e6, times G, phi(g) unless given) with the coverage shares of freq (Table D3):
    F keeps varpi_F of the firm-constant terms (Fz permanent, Fx the projection window's mean of x), I_L varpi_IL of them (Lz, Lx)
    and H (rho-bar_L - rho-bar_F) of x (Ls), I_D the rest (Dz, Dx, Ds); eqs. (D32)-(D34). Variant no_shares: varpi_F = 1"""
    kf, kl, kD, H = SHARES[freq]
    if p.get("no_shares"):
        kf, kl, kD = 1.0, 0.0, 0.0
    G = phi(p["g_" + e]) if G is None else G
    cf, rb, Bz, Bx = rhobar_F(p["rho"]), rhobar(p["rho"]), p["Bz"], p["Bx"]
    return {"Fz": kf * G * Bz, "Fx": kf * G * cf * Bx, "Lz": kl * G * Bz, "Lx": kl * G * cf * Bx, "Ls": H * G * (rb - cf) * Bx,
            "Dz": kD * G * Bz, "Dx": kD * G * cf * Bx, "Ds": G * ((1 - H) * (rb - cf) + 1 - rb) * Bx, "k": (kf, kl, kD)}


def eta(p, e):
    return float({"L": p["eta"], "R": p["etaR"]}[e])


def mem_var(p):
    """variance of the remembered monthly news: sigma_theta^2, times mem_scale in the variant that remembers cash-flow news only"""
    return p["s2w"] * p.get("mem_scale", 1.0)


@lru_cache(maxsize=None)
def remembered_share(e, h):
    """share of signal months whose return month h months later lies inside e's memory"""
    return float(lam(np.arange(SAMPLE[0], SAMPLE[1] - h + 1) + h, e).mean())


def k_daily(p, e):
    """daily memory covariance (x 1e6) for a remembered return day, eq. (D25): phi(eta) sigma_theta^2 / (21^2 N_theta)"""
    E2 = eta(p, e) ** 2
    return 1e6 * E2 * mem_var(p) / (TD ** 2 * N_THETA * (E2 + 1))


def model_cells(p, e):
    """encoder e's model values by component: daily covariances x 1e6, h = 1..10 (eqs. D2, D32-D34), and MSRR component returns,
    months 1..36 (eqs. D10, D35-D37 times Upsilon = M_e; with R^2(z|X) > 0, eq. D18). The category level is zero"""
    vt, rho, n, sK = p["vt"], p["rho"], p["nbar"], p["sigK2"]
    a, det = p["a_" + e], phi(p["q_" + e]) * p["rnu"] * sK                 # the detail's day-one covariance
    hd, hm = np.arange(1, DAY_H + 1.0), np.arange(1, MO_H + 1.0)
    kern, rd = vt ** (hd - 1), rho ** ((hd - 1) / TD)
    q = parts(p, e, "day")
    kf, kl, kD = q["k"]
    mem = k_daily(p, e) * remembered_share(e, 0) if e in MEM else 0.0
    day = {"cL": 0 * hd, "cD": phi(a) * sK * kern, "yF": q["Fz"] + q["Fx"] * rd + kf * mem,
           "uL": q["Lz"] + (q["Lx"] + q["Ls"]) * rd + kl * mem, "uD": det * kern + q["Dz"] + (q["Dx"] + q["Ds"]) * rd + kD * mem}
    ups, r2 = p["M_" + e], p.get("r2zx", 0.0)
    q = parts(p, e, "mo", ups * month_reliability(p["g_" + e] * np.sqrt(1 - r2), n))
    if r2:                          # minus Cov(z_X^j, zbar + rho^(h-1) x), z_X^j = c_j z_X, X measuring the current state with noise
        for z, xs in (("Fz", ("Fx",)), ("Lz", ("Lx", "Ls")), ("Dz", ("Dx", "Ds"))):
            cj = (q[z] + sum(q[k] for k in xs)) / (p["Bz"] + p["Bx"])            # G c_j, c_j = Cov(z^j, z)
            q[z] -= cj * r2 * p["Bz"]
            q[xs[0]] -= cj * r2 * p["Bx"]
    kf, kl, kD = q["k"]
    d1, dh, r = drift_month(vt, 1), drift_month(vt, hm), rho ** (hm - 1)
    if e in MEM:
        E2 = eta(p, e) ** 2
        mem = ups * 1e6 * E2 * mem_var(p) / (N_THETA * (E2 + 1 / n)) * np.array([remembered_share(e, int(h)) for h in hm])
    else:
        mem = 0.0
    mo = {"cL": 0 * hm, "cD": ups * n * phi(a) * sK * d1 * dh, "yF": q["Fz"] + q["Fx"] * r + kf * mem,
          "uL": q["Lz"] + (q["Lx"] + q["Ls"]) * r + kl * mem, "uD": ups * n * det * d1 * dh + q["Dz"] + (q["Dx"] + q["Ds"]) * r + kD * mem}
    if "UF_" + e in p:                                                     # variant ups_FL: Upsilon_F and Upsilon_IL replace M_e
        mo["yF"], mo["uL"] = mo["yF"] * p["UF_" + e] / ups, mo["uL"] * p["UL_" + e] / ups
    return {"day": day, "mo": mo}


def comp_cov(p, e, part, m):
    """encoder e's next-day covariance of a component (x 1e6) in signal months m: stationary part plus memory on remembered
    return months, in the component's coverage share; returns (total, memory)"""
    q = parts(p, e, "day")
    kf, kl, kD = q["k"]
    base = {"F": q["Fz"] + q["Fx"], "IL": q["Lz"] + q["Lx"] + q["Ls"], "CD": phi(p["a_" + e]) * p["sigK2"],
            "dev": (phi(p["a_" + e]) + phi(p["q_" + e]) * p["rnu"]) * p["sigK2"] + q["Dz"] + q["Dx"] + q["Ds"]}
    base["lev"] = base["F"] + base["IL"]
    share = {"F": kf, "IL": kl, "lev": kf + kl, "dev": kD, "CD": 0.0}[part]
    la = share * k_daily(p, e) * lam(m, e) if e in MEM else np.zeros(len(m))
    return base[part] + la, la


def cutoff_ratio(p, part, enc):
    """the model's relative IC ratio across encoder enc's split, eq. (D28) for Llama-2 and (D29) for RoBERTa: the split encoder's
    after-to-before ratio of mean next-day covariance over the control's (both with memory; the forecasts' dispersion is
    stationary in the model); and the memory share of the split encoder's covariance before the split, 1 - omega^o"""
    s = SPLITS[enc]
    m = np.arange(SAMPLE[0], s["last"] + 1)
    post = m >= s["split"]
    (A, la), (B, _) = comp_cov(p, s["treat"], part, m), comp_cov(p, s["control"], part, m)
    return float((A[post].mean() / A[~post].mean()) / (B[post].mean() / B[~post].mean())), float(la[~post].mean() / A[~post].mean())


def premium_var(p, k):
    """variance of the sum of k monthly state premiums (Var(z) = 1), the bracket of eq. (D23)"""
    szb = p["Bz"] / (p["Bz"] + p["Bx"])
    return TD ** 2 * 1e-6 * (p["Bz"] + p["Bx"]) * k ** 2 * (szb + (1 - szb) * avg_corr(p["rho"], k))


def sigma_theta2(p):
    """sigma_theta^2 from eq. (D23): the variance of 36-month return sums net of the premia's, over 36"""
    v = CUT["return_variance"]
    s = v["var_window"] - premium_var(p, v["months"])
    if s <= 0:
        raise RuntimeError("the state premia's variance exceeds the 36-month return variance")
    return s / v["months"]


def dr_share(p):
    """state-related discount-rate news over sigma_theta^2: pi_m^2 Var(x) (1 - rho_m^2) (chi / (1 - chi rho_m))^2,
    chi = 0.967^(1/12) (Vuolteenaho 2002)"""
    chi = 0.967 ** (1 / 12)
    return TD ** 2 * 1e-6 * p["Bx"] * (1 - p["rho"] ** 2) * (chi / (1 - chi * p["rho"])) ** 2 / p["s2w"]


def valuation_terms(p):
    """eq. (D20) on a December window of PROJ_MONTHS months: Var(z^F), Corr(z^F, z)^2 and, by encoder, the third term
    g^2 Var(z^F) / (g^2 Var(z^F) + 1/nbar^F)"""
    rho, Vz = p["rho"], p["Bz"] / (p["Bz"] + p["Bx"])
    B = Vz + (1 - Vz) * avg_corr(rho, PROJ_MONTHS)
    cov = Vz + (1 - Vz) * float(np.mean(rho ** np.arange(PROJ_MONTHS)))      # window ending at T
    third = {m: p["g_" + CODE[m]] ** 2 * B / (p["g_" + CODE[m]] ** 2 * B + 1 / N_F) for m in PANELS}
    return {"var_zF": B, "corr2": cov ** 2 / B, "third": third}


def cells(p, m, kind, hs, data=None):
    """model values by component and horizon: daily (covariance x 1e6), bridge (the month's average of the daily forecast,
    eq. D14 with the firm-month coverage shares), strategy (MSRR, %/yr before scaling) and ridge (the monthly projection
    forecast, DKPT's components up to a common factor, scaled to the data's total at h = 1); plus lev, dev and total"""
    e, hs = CODE[m], np.asarray(hs, float)
    z = np.zeros(len(hs))
    if kind in ("daily", "strategy", "ridge"):
        v = model_cells(p, e)["day" if kind == "daily" else "mo"]
        c = {k: v[k][hs.astype(int) - 1] for k in PARTS}
    else:
        q = parts(p, e, "mo")
        kf, kl, kD = q["k"]
        dh, dec = drift_month(p["vt"], hs), p["rho"] ** (hs - 1)
        mem = (np.array([TD * k_daily(p, e) * remembered_share(e, int(h)) for h in hs]) if e in MEM else z)
        GC1, det1 = phi(p["a_" + e]) * p["sigK2"], phi(p["q_" + e]) * p["rnu"] * p["sigK2"]
        c = {"cL": z, "cD": GC1 * dh, "yF": TD * (q["Fz"] + q["Fx"] * dec) + kf * mem,
             "uL": TD * (q["Lz"] + (q["Lx"] + q["Ls"]) * dec) + kl * mem,
             "uD": det1 * dh + TD * (q["Dz"] + (q["Dx"] + q["Ds"]) * dec) + kD * mem}
    if kind == "ridge":
        tot = sum(v[0] for v in c.values())
        c = {k: v * data / tot for k, v in c.items()}
    c["lev"], c["dev"] = c["cL"] + c["yF"] + c["uL"], c["cD"] + c["uD"]
    c["total"] = c["lev"] + c["dev"]
    return c


# ---- data moments ----------------------------------------------------------------------------------------------------------
def ic_ratio(A, B, post):
    """relative IC ratio: A's mean IC over B's after the split, relative to the same ratio before"""
    return (np.nanmean(A[post]) / np.nanmean(B[post])) / (np.nanmean(A[~post]) / np.nanmean(B[~post]))


def block_draws(A, B, post, L, rng, n=2000):
    """moving-block bootstrap of ic_ratio, blocks of L trading days drawn separately before and after the split"""
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


def split_targets(X, a, b, idx, split):
    """a over b across the split on trading days idx, by component: relative next-day rank-IC ratio with its 90% interval
    (40-day blocks), se = (hi - lo) / 3.29, and the same ratio of cross-sectional standard deviations"""
    post = np.asarray(idx >= pd.Timestamp(split))
    out = {"days": {"pre": int((~post).sum()), "post": int(post.sum())}}
    for k, name in COMPONENTS:
        A, B = X[a].loc[idx, f"ic_{k}"].values, X[b].loc[idx, f"ic_{k}"].values
        o = block_draws(A, B, post, CUTOFF_BLOCK, np.random.default_rng(CUTOFF_SEED))
        lo, hi = float(np.nanpercentile(o, 5)), float(np.nanpercentile(o, 95))
        disp = ic_ratio(X[a].loc[idx, f"sd_{k}"].values, X[b].loc[idx, f"sd_{k}"].values, post)
        out[name] = {"ic": float(ic_ratio(A, B, post)), "lo": lo, "hi": hi, "se": (hi - lo) / 3.29, "disp": float(disp)}
    return out


def placebo_ratios(X, idx):
    """Llama-2 over RoBERTa's relative IC ratio at each placebo split, shared days from the sample start through PLACEBO_POST years
    after the split (I_L omitted: its mean ICs are near zero); Figure D1, in no table"""
    out = {name: {} for _, name in COMPONENTS if name != "IL"}
    for y in PLACEBO_YEARS:
        ii = idx[idx < pd.Timestamp(f"{y + PLACEBO_POST}-01-01")]
        post = np.asarray(ii >= pd.Timestamp(f"{y}-01-01"))
        for k, name in COMPONENTS:
            if name in out:
                out[name][y] = float(ic_ratio(X[LM].loc[ii, f"ic_{k}"].values, X[RB].loc[ii, f"ic_{k}"].values, post))
    return out


IC_HS = (1, 2, 3, 5, 10)                                                   # horizons of the historical level's IC ratio (figure)


def level_ic_by_horizon():
    """the historical level's rank IC around January 2023 (framework figure): mean next-day IC before and after the split by encoder,
    and Llama-2 over RoBERTa's relative IC ratio by horizon with its 90% interval (40-day blocks), on the days the three encoders
    share (4_horizon ic_daily; at h = 1 the same series as cutoff_daily.csv.gz)"""
    rng, s = np.random.default_rng(CUTOFF_SEED), SPLITS["L"]
    out = {"means": {}, "ratio": {}}
    for h in IC_HS:
        S = {}
        for m in NAME:
            x = read(RES / "paper/4_horizon" / m / "ic_daily.csv.gz", parse_dates=["day0"])
            S[m] = x[x.h == h].set_index("day0")["lev"]
        idx = S[LM].index.intersection(S[RB].index).intersection(S[CG].index)
        S = {m: v.loc[idx].values for m, v in S.items()}
        post = np.asarray(idx >= pd.Timestamp(month_date(s["split"])))
        if h == 1:
            out["means"] = {m: (100 * float(np.nanmean(S[m][~post])), 100 * float(np.nanmean(S[m][post]))) for m in NAME}
        o = block_draws(S[LM], S[RB], post, CUTOFF_BLOCK, rng)
        out["ratio"][h] = {"split": float(ic_ratio(S[LM], S[RB], post)), "lo": float(np.nanpercentile(o, 5)), "hi": float(np.nanpercentile(o, 95))}
    return out


@lru_cache(maxsize=1)
def cutoff_targets():
    """the data's cutoff targets on the trading days all three encoders share, by split encoder (SPLITS), and the placebo splits.
    Every Llama-2 component becomes more dispersed relative to RoBERTa's by about the same factor, including C_D, which carries no
    memory, while C_D's IC ratio stays near one: the scale-free IC ratio is the target"""
    D = read(DAILY_FILE, parse_dates=["day0"])
    X = {m: g.set_index("day0").sort_index() for m, g in D.groupby("enc")}
    idx = X[LM].dropna().index.intersection(X[RB].dropna().index).intersection(X[CG].dropna().index).sort_values()
    out = {}
    for enc, s in SPLITS.items():
        end = (pd.Period(month_date(s["last"])[:7], "M") + 1).to_timestamp() - pd.Timedelta(days=1)   # last day of the last month
        out[enc] = split_targets(X, MODEL[s["treat"]], MODEL[s["control"]], idx[idx <= end], month_date(s["split"]))
    out["placebo"] = placebo_ratios(X, idx)
    return out


def search_moments(mo_h1=()):
    """every moment: role (step1: category long-short on days 0, 1, 10; target: band means and the relative IC ratios of F and
    E_L; check: single horizons, the SUE gain and the other components' IC ratios), block (weight one each in the search),
    encoder, horizons lo..hi or the cutoff component, data and se. A band's se: Newey-West on the band-averaged series by signal
    date, NW lags plus the band width minus one. mo_h1: components whose month 1 is also a target (variant)"""
    rows = []
    add = lambda role, blk, e, lo, hi, comp, v, se: rows.append((role, blk, e, lo, hi, comp, float(v), float(abs(se))))
    c = read(RES / "paper/1_drift/drift_ls.csv")
    c = c[c.signal == CAT].set_index("h")
    for h in range(11):
        add("step1" if h in (0, 1, 10) else "check", "ls", "-", h, h, "", c.ls_bp[h], c.ls_bp[h] / c.t[h])
    for m, e in CODE.items():
        t = paper("4_horizon", m, "term_structure")
        g = {k: t[t.part == k].set_index("h") for k in PARTS}
        cv = read(RES / "paper/4_horizon" / m / "cov_daily.csv.gz")
        mon = paper("4_didisheim/attribution", m, "monthly")
        mon = mon[mon.weights == "fixed"].assign(signal=lambda x: pd.PeriodIndex(x.month, freq="M") - x.tau)
        mo = attribution(m)
        for k in PARTS:
            for h in range(1, 11):
                add("check", f"day_{k}", e, h, h, k, g[k]["mean"][h], g[k]["mean"][h] / g[k]["t"][h])
            for lo, hi in DAY_BANDS.get(k, {}).values():
                mean = np.mean([g[k]["mean"][h] for h in range(lo, hi + 1)])
                ser = cv[cv.h.between(lo, hi)].groupby("day0")[k].mean() * 1e6
                assert abs(ser.mean() - mean) <= 1e-9 * max(1.0, abs(mean)), f"{m} {k}: cov_daily and term_structure disagree"
                add("target", f"dayband_{k}", e, lo, hi, k, mean, nw_se(ser.values, NW_DAILY + hi - lo))
            if k in MO_TARGETS:
                for lo, hi in ([(1, 1)] if k in mo_h1 else []) + list(MO_BANDS.values()):
                    mean = np.mean([mo[k]["mean_%/yr"][h] for h in range(lo, hi + 1)])
                    q = mon[(mon.part == k) & mon.tau.between(lo - 1, hi - 1)].groupby("signal").ret
                    ser = 1200 * q.mean()[q.count() == hi - lo + 1]             # signal months with every horizon of the band
                    add("target", f"moband_{k}", e, lo, hi, k, mean, nw_se(ser.values, NW_MONTHLY + hi - lo))
            for h in mo[k].index:
                add("check", f"mo_{k}", e, h, h, k, mo[k]["mean_%/yr"][h], mo[k]["mean_%/yr"][h] / mo[k]["t"][h])
        if m in FORECAST:
            pa = paper("3_surprise", m, "panel_a").set_index("spec")
            # se of a small R2 on n observations, about 2 sqrt(R2 / n)
            add("check", "sue", e, np.nan, np.nan, "", pa.R2_oos["C + E (ridge)"] - pa.R2_oos["C"],
                2 * np.sqrt(pa.R2_oos["C + E (ridge)"] / pa.n_test["C + E (ridge)"]))
    CT = cutoff_targets()
    for enc in SPLITS:
        for part in ("F", "lev", "IL", "dev", "CD"):
            add("target" if part in CUTOFF_TARGETED else "check", f"cutic_{enc}", enc, 1, 1, part, CT[enc][part]["ic"], CT[enc][part]["se"])
    return pd.DataFrame(rows, columns=["role", "block", "enc", "lo", "hi", "comp", "data", "se"])


def sue_categories():
    """R2(s|K): the out-of-sample SUE R2 of the categories alone (the same in both encoders' samples)"""
    r = [float(paper("3_surprise", m, "panel_a").set_index("spec").R2_oos["C"]) for m in PANELS]
    assert abs(r[0] - r[1]) < 1e-12
    return r[0]


def normalizations(d):
    """the category signal's scale (each forecast's next-day covariance times the squared ratio of the category mean's long-short
    return to the forecast's and the ratio of their Fama-MacBeth slopes, mean over the two forecast encoders) and Llama-2's SUE
    gain over its categories' R2, which sets its phi(q) by eq. (D7): gain = phi(q) (1 - R2(s|K))"""
    lv, fm = d["levels"], read(RES / "paper/1_category/fm_raw.csv").set_index("spec")
    sig = []
    for m, f in FORECAST.items():
        yh = d["ts"][m]["yhat"]["mean"][1]
        sig.append((lv.EW_LS_bp[CAT] / lv.EW_LS_bp[f]) ** 2 * fm.coef[f] / fm.coef[CAT] * yh)
    pa = paper("3_surprise", LM, "panel_a").set_index("spec").R2_oos
    return float(np.mean(sig)), float((pa["C + E (ridge)"] - pa["C"]) / (1 - sue_categories()))


# ---- the search ------------------------------------------------------------------------------------------------------------
def search_model(p, mom):
    """the model counterpart of each row of the moments mom"""
    cache, out = {}, np.empty(len(mom))
    for i, (b, e, lo, hi, comp) in enumerate(zip(mom.block, mom.enc, mom.lo, mom.hi, mom.comp)):
        if b == "ls":
            out[i] = p["S"] * (p["kappa"] if lo == 0 else (1 - p["kappa"]) * (1 - p["vt"] ** lo))
        elif b == "sue":                                                  # eq. (D7)
            out[i] = phi(p["q_" + e]) * (1 - p["sueK"])
        elif b.startswith("cutic"):
            out[i] = cutoff_ratio(p, comp, e)[0]
        else:
            if e not in cache:
                cache[e] = model_cells(p, e)
            v = cache[e]["day" if b.startswith("day") else "mo"][b.split("_", 1)[1]]
            out[i] = v[int(lo) - 1:int(hi)].mean()
    return out


def search_fit(mom, names, x0, lb, ub, fixed, starts=None):
    """minimize the sum over blocks of (1/n_b) t_b' t_b, t = (data - model) / se; from x0 scaled by 1, 0.7 and 1.4 unless starts are
    given; returns the best parameters, its cost (least_squares' half sum) and every start's (cost, x)"""
    w = np.sqrt(1 / mom.groupby("block").block.transform("size").values)
    def f(x):
        return w * (mom.data.values - search_model({**fixed, **dict(zip(names, x))}, mom)) / mom.se.values
    best, runs = None, []
    for x in (starts if starts is not None else [np.array(x0) * s_ for s_ in (1.0, 0.7, 1.4)]):
        r = least_squares(f, np.clip(np.array(x), lb + 1e-9, ub - 1e-9), bounds=(lb, ub), x_scale="jac", max_nfev=20000)
        if r.status <= 0:
            raise RuntimeError(f"least_squares did not converge: {r.message}")
        runs.append((float(r.cost), r.x))
        best = r if best is None or r.cost < best.cost else best
    return dict(zip(names, best.x)), float(best.cost), runs


def sobol_starts(names):
    """SOBOL_N scrambled Sobol points over SOBOL_BOX"""
    out = []
    for row in qmc.Sobol(len(names), scramble=True, seed=SOBOL_SEED).random(SOBOL_N):
        x = []
        for v, k in zip(row, names):
            lo, hi, lg = SOBOL_BOX[k]
            x.append(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))) if lg else lo + v * (hi - lo))
        out.append(np.array(x))
    return out


def search(d, spec=None):
    """step 1: kappa, vartheta (and the long-short scale S) from the category long-short; step 2: the rest jointly, with
    sigma_theta^2 solved as a damped fixed point of eq. (D23), converged when |s_new - s2w| <= S2W_TOL (s_new + s2_max); first
    round from a grid over the memory loadings and SOBOL_N Sobol points, later rounds from the previous solution; at the final
    variance the Sobol starts again (how many reach the minimum). Normalizations: the category signal's scale, Llama-2's phi(q)
    and g = 1. spec: a variant (VARIANTS). Returns parameters, fit diagnostics and the moments with the model's values"""
    spec = dict(spec or {})
    mom = search_moments(spec.pop("mo_h1", ()))
    sig, qL = normalizations(d)
    p, _, _ = search_fit(mom[mom.role == "step1"].reset_index(drop=True), ["kappa", "vt", "S"], [0.68, 0.26, 150.0],
                         np.array([0.05, 0.01, 1.0]), np.array([0.99, 0.99, 1e4]), {})
    names, x0, lb, ub = list(NAMES), X0.copy(), LB.copy(), UB.copy()
    fixed = {**p, "sigK2": sig, "g_L": 1.0, "q_L": float(np.sqrt(qL / (1 - qL))), "nbar": 1 / CUT["firm_month_inv_n"],
             "sueK": sue_categories()}
    if spec.pop("no_perm", False):
        keep = [k != "Bz" for k in names]
        names, x0, lb, ub, fixed["Bz"] = [k for k in names if k != "Bz"], x0[keep], lb[keep], ub[keep], 0.0
    if spec.pop("ups_FL", False):
        extra = [f"{k}_{e}" for e in "RCL" for k in ("UF", "UL")]
        x0e = [X0[NAMES.index("M_" + k.split("_")[1])] for k in extra]
        names, x0, lb, ub = names + extra, np.r_[x0, x0e], np.r_[lb, np.full(6, 1e-3)], np.r_[ub, np.full(6, 1e5)]
    assert set(spec) <= {"no_shares", "mem_scale", "r2zx"}, f"unknown variant switches {spec}"
    fixed.update(spec)                                                     # switches the model functions read
    start0 = []
    for e0 in (0.3, 1.2, 2.5):
        for r0 in (0.0, 0.5, 1.5):
            x_ = x0.copy()
            x_[names.index("eta")], x_[names.index("etaR")] = e0, r0
            start0 += [x_ * s_ for s_ in (1.0, 0.7, 1.4)]
    start0 += sobol_starts(names)
    T = mom[mom.role == "target"].reset_index(drop=True)
    s2w, x, rounds = None, x0, 0
    s2_max = CUT["return_variance"]["var_window"] / CUT["return_variance"]["months"]   # sigma_theta^2 without a premium
    for it in range(13):
        s_new = sigma_theta2({**fixed, **dict(zip(names, x))})
        if s2w is not None and abs(s_new - s2w) <= S2W_TOL * (s_new + s2_max):
            break
        if it == 12:
            raise RuntimeError("the sigma_theta^2 fixed point did not converge in 12 rounds")
        s2w = s_new if s2w is None else 0.5 * (s2w + s_new)
        fixed["s2w"] = s2w
        q, cost, _ = search_fit(T, names, x, lb, ub, fixed, start0 if it == 0 else [x])
        x, rounds = np.array([q[k] for k in names]), it + 1
    _, c_ms, runs = search_fit(T, names, x, lb, ub, fixed, sobol_starts(names))
    if c_ms < cost - 1e-6:
        raise RuntimeError(f"a Sobol start beats the search at the final variance: {c_ms} < {cost}")
    near = [r for r in runs if r[0] < cost + 1e-3]
    other = min((r for r in runs if r[0] >= cost + 1e-3), key=lambda r: r[0], default=None)   # the best other local minimum
    p = {**fixed, **dict(zip(names, x))}
    fit = {"criterion": 2 * cost, "rounds": rounds, "s2w_implied": s_new, "s2w_gap_rel": abs(s_new - s2w) / s_new,
           "starts": len(runs), "starts_at_min": len(near),
           "at_bounds": [k for k, v, l, u in zip(names, x, lb, ub) if v - l <= 1e-9 * max(1.0, abs(l)) or u - v <= 1e-9 * max(1.0, abs(u))],
           "next_min": None if other is None else {"criterion": 2 * other[0], "starts": sum(r[0] < other[0] + 1e-3 for r in runs) - len(near),
                                                   **{k: float(v) for k, v in zip(names, other[1])}}}
    mom["model"] = search_model(p, mom)
    return p, fit, mom


# ---- statistics outside the search -----------------------------------------------------------------------------------------
def finite_ic(c, b, sr, dof, extra=0.0):
    """Proposition D1: the population IC of the forecast and its finite-sample value with dof effective parameters on N_TRAIN
    items, IC^2 / (1 + dof sr^2 / (n S)), S = sum C^2 / V + extra; extra: the memory covariance, whose forecast variance along
    the memory direction is approximated by that covariance"""
    V, C = c ** 2 + 1, c * b
    S = np.sum(C ** 2 / V) + extra
    return np.sqrt(S) / sr, np.sqrt(S / sr ** 2 / (1 + dof * sr ** 2 / (N_TRAIN * S)))


def encoders(p):
    """Proposition D1 for the two forecast encoders: sigma_r set so that RoBERTa's finite-sample IC equals the data's; the model's
    IC and Mincer-Zarnowitz slope (fin / pop)^2 for both"""
    b = np.array([np.sqrt(1e-6 * p["sigK2"]), np.sqrt(1e-6 * p["sigK2"] * p["rnu"]), np.sqrt(1e-6 * (p["Bz"] + p["Bx"]))])
    out = {"b": b.tolist()}
    for m in PANELS:
        e = CODE[m]
        t = paper("4_horizon", m, "term_structure")
        out[m] = {"c": np.array([p["a_" + e], p["q_" + e], p["g_" + e]]), "dof": float(paper("2_spectrum", m, "dof").dof.mean()),
                  "memory": 1e-6 * k_daily(p, e) * remembered_share(e, 0), "IC_data": float(t[(t.part == "yhat") & (t.h == 1)]["IC_%"].iat[0])}
    ic = lambda m, sr: finite_ic(out[m]["c"], b, sr, out[m]["dof"], out[m]["memory"])
    sr = brentq(lambda v: 100 * ic(RB, v)[1] - out[RB]["IC_data"], 1e-3, 0.5)
    out["sigma_r"] = sr
    for m in PANELS:
        pop, fin = ic(m, sr)
        out[m].update({"IC_model": 100 * fin, "mz_model": (fin / pop) ** 2})
    fm = read(RES / "paper/1_category/fm_raw.csv").set_index("spec").coef
    for m, f in FORECAST.items():
        out[m]["mz_data"] = float(fm[f])
    return out


def lagged_level(p):
    """eqs. (D16), (D31): standalone month-1 MSRR strategies on the current historical level and on the level lagged 36 months.
    Weights are the population projection of next-month returns on the input (state and memory directions; the memory weight
    scales with the MSRR window's coverage f); the level averages twelve monthly embeddings (state variance g^2 [perm + avg_corr
    (1 - perm)], noise 1/(12 nbar)); the lagged level keeps the permanent state, the memory and rho_m^36 of x's covariance. With S
    the state's term, K the memory's, a and b the means of f lambda and f^2 over the signal months and r the lagged-to-current
    state covariance ratio, SR is proportional to (r^2 S + a K) / sqrt(r^2 S + b K); the lagged SR is multiplied by sqrt of its
    breadth (lagged_breadth)"""
    out = {}
    v, rho = 1 / (LEVEL_MONTHS * p["nbar"]), p["rho"]
    rb, perm = rhobar(rho), p["Bz"] / (p["Bz"] + p["Bx"])
    c, c_old = perm + rb * (1 - perm), perm + rb * rho ** LEVEL_LAG * (1 - perm)
    r = c_old / c
    sig = np.arange(STANDALONE_START, SAMPLE[1])
    s_lag, keep = CUT["lagged_breadth"]["lagged_over_current"], avg_corr(rho, LEVEL_MONTHS)
    for m, e in CODE.items():
        g = p["g_" + e]
        S = TD ** 2 * (p["Bz"] + p["Bx"]) * g ** 2 * c ** 2 / (g ** 2 * (perm + keep * (1 - perm)) + v)
        K = a = b = 0.0
        if e in MEM:
            E2 = eta(p, e) ** 2
            K = 1e6 * E2 * mem_var(p) / (N_THETA * (E2 + v))
            f = np.array([lam(np.arange(SAMPLE[0], x) + 1, e).mean() for x in sig])   # the expanding MSRR window's coverage
            a, b = float(np.mean(f * lam(sig + 1, e))), float(np.mean(f ** 2))
        sr_model = lambda q: (q * S + a * K) / np.sqrt(q * S + b * K)
        t = paper("4_didisheim/attribution", m, "standalone").assign(h=lambda x: x.tau + 1)
        sr = lambda obj: t[(t.object == obj) & (t.h == 1)].Sharpe.iat[0]
        out[m] = {"model": float(np.sqrt(s_lag) * sr_model(r ** 2) / sr_model(1.0)),
                  "data": sr("historical level 36 months earlier") / sr("historical level"),
                  "memory_share": float(a * K / (S + a * K)), "memory_share_lagged": float(a * K / (r ** 2 * S + a * K))}
    return out


def check_stats(d, p, enc, lag, mom):
    """the entries of Table 13 (model, data) for RoBERTa and Llama-2: firm identity's day-one covariance, the monthly-averaged
    forecast's F + I_L at month 1 and month 12 over month 1, the MSRR strategy's month-1 historical-level share, the
    lagged level's Sharpe ratio over the current level's, the forecast IC and Mincer-Zarnowitz slope, the SUE gain; and the
    Llama-2 over RoBERTa ratio of the valuation R2's third terms (eq. D20)"""
    st = {}
    for m in PANELS:
        con = paper("4_monthly_ridge", m, "contributions")
        br = con[(con.predictor == "bridge") & con.part.isin(["yF", "uL"])].groupby("tau")["mean"].sum()   # F + I_L
        brm = cells(p, m, "bridge", [1, 12])
        brm = brm["yF"] + brm["uL"]
        sue = mom[(mom.block == "sue") & (mom.enc == CODE[m])].iloc[0]
        sm, att = cells(p, m, "strategy", [1]), attribution(m)
        st[m] = {"f_day1": (cells(p, m, "daily", [1])["yF"][0], d["ts"][m]["yF"]["mean"][1]),
                 "br_lev": (brm[0], br[0]), "br_lev12": (brm[1] / brm[0], br[11] / br[0]),
                 "lagged_level": (lag[m]["model"], lag[m]["data"]),
                 "ic": (enc[m]["IC_model"], enc[m]["IC_data"]), "mz": (enc[m]["mz_model"], enc[m]["mz_data"]),
                 "lev_share": (100 * sm["lev"][0] / sm["total"][0], 100 * att["lev"]["mean_%/yr"][1] / att["total"]["mean_%/yr"][1]),
                 "sue": (100 * sue.model, 100 * sue.data)}
    third = valuation_terms(p)["third"]
    st["valuation"] = (third[LM] / third[RB], r2_log_pb(LM) / r2_log_pb(RB))
    return st


# ---- tables --------------------------------------------------------------------------------------------------------------
def num(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


def write_tab(name, align, head, lines):
    (HERE / name).write_text("\n".join([f"\\begin{{tabular}}{{{align}}}", "\\toprule", *head, "\\midrule", *lines,
                                        "\\bottomrule", "\\end{tabular}"]) + "\n")


def group(title, ncol, first=False):
    return ("" if first else "\\midrule\n") + f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"


def params_table(p, enc, lag):
    """Table D2"""
    one = lambda v: [v, "", "", ""]                                         # Common, then the three encoders blank
    by = lambda vals: ["", *[v if isinstance(v, str) else num(v) for v in vals]]
    c = {m: (p["a_" + e], p["q_" + e], p["g_" + e]) for m, e in CODE.items()}
    b, d1 = enc["b"], (1 - p["kappa"]) * (1 - p["vt"])
    rows = [
        group("Panel A: Set before the search", 7, first=True),
        ["$\\kappa$", "Announcement response", *one(num(p["kappa"])), "Category long--short return, \\eqref{eq:D1}"],
        ["$\\vartheta$", "Daily drift decay", *one(num(p["vt"])), "Category long--short return, \\eqref{eq:D1}"],
        ["$\\mathrm{sd}(m_K)$", "Category signal", *one(num(100 * b[0] / d1)), "Category-mean long--short return and slope"],
        ["$q$", "Detail loading", *by(["", "", c[LM][1]]), "SUE $R^2$ gain, \\eqref{eq:D7}"],
        ["$g$", "State loading", *by(["", "", "1"]), "Normalization"],
        ["$N_\\theta$", "Memory window, months", *by(["", str(N_THETA), str(N_THETA)]), "\\citet{crane2025total}"],
        ["$\\bar n$", "News items per firm-month", *one(num(p["nbar"])), "Harmonic mean in the data"],
        ["$\\bar n^F$", "News items per firm, valuation windows", *one(num(N_F, 0)), "Data, \\eqref{eq:D20}"],
        group("Panel B: Joint search", 7),
        ["$\\mathrm{sd}(\\nu)$", "Detail", *one(num(100 * b[1] / d1)), "Daily $I_D$, \\eqref{eq:D34}"],
        ["$\\pi$", "State premium", *one(num(1e4 * b[2], 1)), "Daily $F$ net of memory, \\eqref{eq:D32}"],
        ["$\\Var(\\bar z)$", "Permanent share", *one(num(p["Bz"] / (p["Bz"] + p["Bx"]))), "$F$, $I_L$, \\eqref{eq:D32}--\\eqref{eq:D36}"],
        ["$\\rho_m$", "Monthly state persistence", *one(num(p["rho"], 3)), "Monthly components, \\eqref{eq:D35}--\\eqref{eq:D37}"],
        ["$a$", "Category-signal loading", *by([c[m][0] for m in NAME]), "Daily $C_D$, \\eqref{eq:D2}"],
        ["$q$", "Detail loading", *by([c[CG][1], c[RB][1], ""]), "Daily $I_D$, \\eqref{eq:D34}"],
        ["$g$", "State loading", *by([c[CG][2], c[RB][2], ""]), "Daily $F$, $I_L$, monthly components"],
        ["$\\eta$", "Memory loading", *by(["", p["etaR"], p["eta"]]), "IC ratios (Llama-2), monthly $F$, $I_L$ (RoBERTa)"],
        ["$\\Upsilon$", "Monthly strategy scale", *by([p["M_" + CODE[m]] for m in NAME]), "Monthly components, \\eqref{eq:D17}"],
        ["$\\sigma_\\theta^2$", "Cash-flow-news variance, monthly", *one(num(p["s2w"], 3)), "\\eqref{eq:D23}, jointly with $\\pi$"],
        group("Panel C: Implied", 7),
        ["", "Half-life of $x$, months", *one(num(np.log(2) / -np.log(p["rho"]), 0)), "$\\ln 2/(-\\ln\\rho_m)$"],
        ["$\\phi(g)$", "Reliability of the state measure", *by([phi(c[m][2]) for m in NAME]), "$g^2/(1+g^2)$"],
        ["$\\phi(\\eta)$", "Reliability of the memory measure", *by(["", phi(p["etaR"]), phi(p["eta"])]), "$\\eta^2/(1+\\eta^2)$"],
        ["", "Memory's share of the first-month mean return, MSRR on $E_L$ (\\%)", *by([num(100 * lag[m]["memory_share"], 0) for m in NAME]),
         "\\eqref{eq:D16}, \\eqref{eq:D26}"],
        ["", "Memory's share of the first-month mean return, MSRR on $E_{L,\\tau-36}$ (\\%)",
         *by([num(100 * lag[m]["memory_share_lagged"], 0) for m in NAME]), "\\eqref{eq:D16}, \\eqref{eq:D31}"],
    ]
    write_tab("cal_params.tex", "llccccl",
              [" & ".join(["Parameter", "Description", "Common", "ChronoGPT$_{1999}$", "RoBERTa", "Llama-2", "Source"]) + " \\\\"],
              [r if isinstance(r, str) else " & ".join(r) + " \\\\" for r in rows])


def shares_table():
    """Table D3"""
    labs = (("Firm effect estimated", "$\\varpi_F$", "$F$", "F"), ("$F=0$, historical level available", "$\\varpi_{I_L}$", "$I_L$", "IL"),
            ("$F=0$, no historical level", "$\\varpi_{I_D}$", "$I_D$", "ID"), ("News available in the preceding 12 months", "$H$", "", "H"))
    av = CUT["availability"]
    write_tab("cal_shares.tex", "lcccc", [" & Share & Component & Firm-days & Firm-months \\\\"],
              [" & ".join((a, s, c, num(av["firm_days"][k], 3), num(av["firm_months"][k], 3))).replace("&  &", "& &") + " \\\\" for a, s, c, k in labs])


MONTH_BANDS = "Months 2--12, 13--24, 25--36"
TARGET_ROWS = [("ls", "Cumulative category long--short return", "D1", "Days 0, 1, 10"),
               "Daily covariance with returns",
               ("dayband_cD", "$C_D$", "D2", "Day 1; days 2--10"), ("dayband_uD", "$I_D$", "D34", "Day 1; days 2--10"),
               ("dayband_yF", "$F$", "D32", "Days 2--10"), ("dayband_uL", "$I_L$", "D33", "Days 2--10"),
               "MSRR strategy return",
               ("moband_yF", "$F$", "D35", MONTH_BANDS), ("moband_uL", "$I_L$", "D36", MONTH_BANDS), ("moband_uD", "$I_D$", "D37", MONTH_BANDS),
               "Relative IC ratios of $F$ and $E_L$",
               ("cutic_L", "Llama-2, January 2023", "D28", "Next day"), ("cutic_R", "RoBERTa, January 2020", "D29", "Next day")]


def targets_table(mom, fit):
    """Table D4: targeted moments by block, their equation, count and the fit's RMS t = (data - model) / se; the Sobol starts
    reaching the joint search's minimum"""
    lines = []
    for r in TARGET_ROWS:
        if isinstance(r, str):
            lines.append(group(r, 5, first=True))
            continue
        b, lab, eq, hor = r
        x = mom[(mom.block == b) & mom.role.isin(["step1", "target"])]
        t = (x.data - x.model) / x.se
        lines.append(f"{lab} & \\eqref{{eq:{eq}}} & {hor} & {len(x)} & {num(float(np.sqrt((t ** 2).mean())))} \\\\")
    lines += ["\\midrule", f"Sobol starts reaching the minimum & & & {fit['starts_at_min']} of {fit['starts']} & \\\\"]
    write_tab("cal_targets.tex", "lcccc", ["Moment & Equation & Horizons & Count & RMS $t$ \\\\"], lines)


def checks_table(st):
    """Table 13; the cells of the two normalizations (RoBERTa's IC, Llama-2's SUE gain) read "norm." """
    rows = [("Levels outside the targets", None),
            ("Monthly-averaged daily forecast, $F+I_L$, month 1", "br_lev", 2, "\\eqref{eq:D14}, \\eqref{eq:D26}"),
            ("Lagged historical level, month-1 Sharpe ratio over the current level's", "lagged_level", 2, "\\eqref{eq:D16}, \\eqref{eq:D31}"),
            ("SUE $R^2$ gain from the embedding (pp)", "sue", 1, "\\eqref{eq:D7}"),
            ("Firm-identity valuation $R^2$ from the state, Llama-2 over RoBERTa", "valuation", 2, "\\eqref{eq:D20}"),
            ("Model restrictions", None),
            ("Firm identity, day one", "f_day1", 3, "\\eqref{eq:D32}"),
            ("Monthly-averaged daily forecast, $F+I_L$, month 12 over month 1", "br_lev12", 2, "\\eqref{eq:D14}, \\eqref{eq:D26}"),
            ("MSRR strategy, historical-level share, month 1 (\\%)", "lev_share", 1, "\\eqref{eq:D10}, \\eqref{eq:D35}--\\eqref{eq:D37}"),
            ("Mincer--Zarnowitz slope", "mz", 2, "\\eqref{eq:D9}"),
            ("Embedding forecast IC (\\%)", "ic", 2, "\\eqref{eq:D8}")]
    norm = {("ic", RB), ("sue", LM)}
    lines = []
    for r in rows:
        if r[1] is None:
            lines.append(group(r[0], 6, first=not lines))
            continue
        lab, key, k, ref = r
        if key == "valuation":
            v = st["valuation"]
            lines.append(f"{lab} & \\multicolumn{{2}}{{c}}{{{num(v[0], k)}}} & \\multicolumn{{2}}{{c}}{{{num(v[1], k)}}} & {ref} \\\\")
            continue
        mv = {m: "norm." if (key, m) in norm else num(st[m][key][0], k) for m in PANELS}
        dv = {m: num(st[m][key][1], k) for m in PANELS}
        lines.append(f"{lab} & {mv[RB]} & {mv[LM]} & {dv[RB]} & {dv[LM]} & {ref} \\\\")
    write_tab("cal_checks.tex", "lccccl", [" & \\multicolumn{2}{c}{Model} & \\multicolumn{2}{c}{Data} & \\\\",
                                                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                                                "Statistic & RoBERTa & Llama-2 & RoBERTa & Llama-2 & Equation \\\\"], lines)


def cutoff_table(p):
    """Table 12 and its entries: by split encoder and component, the data's relative IC ratio with its 90% interval and dispersion
    ratio, the model's ratio and the look-ahead share 1 - omega^o; and the placebo summaries"""
    CT = cutoff_targets()
    labs = (("F", "Firm identity $F$"), ("lev", "Historical level $E_L$"), ("IL", "Idiosyncratic level $I_L$"),
            ("dev", "Deviation $E_D$"), ("CD", "Category deviation $C_D$"))
    out, lines = {}, []
    for i, (enc, title) in enumerate((("L", "Llama-2 across January 2023, relative to RoBERTa"),
                                      ("R", "RoBERTa across January 2020, relative to Llama-2"))):
        lines.append(("\\midrule\n" if i else "") + f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        out[enc] = {}
        for part, lab in labs:
            c = CT[enc][part]
            r, own = cutoff_ratio(p, part, enc)
            out[enc][part] = {**c, "model": r, "lookahead": own}
            lines.append(f"{lab} & {'Yes' if part in CUTOFF_TARGETED else 'No'} & {num(c['ic'])} & [{num(c['lo'])}, {num(c['hi'])}] & "
                         f"{num(c['disp'])} & {num(r)} & {num(100 * own, 0)} \\\\")
    out["placebo"] = {part: {"ratios": v, **dict(zip(("q25", "median", "q75"), map(float, np.percentile(list(v.values()), [25, 50, 75]))))}
                      for part, v in CT["placebo"].items()}
    write_tab("cal_cutoff.tex", "lcccccc", [" & & \\multicolumn{3}{c}{Data} & \\multicolumn{2}{c}{Model} \\\\",
                                            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
                                            "Component & Targeted & IC ratio & 90\\% interval & Dispersion ratio & IC ratio & Memory share (\\%) \\\\"], lines)
    return out


# ---- numbers the text quotes, provenance -----------------------------------------------------------------------------------
def text_numbers(d, p, fit, lag):
    """numbers Appendix D and Section 7.3 quote that no table shows (Table D5's ranges are in cal_variants.json)"""
    v36 = CUT["return_variance"]
    out = {"premium_share_36m": premium_var(p, MO_H) / v36["var_window"], "dr_share": dr_share(p), "criterion": fit["criterion"],
           "sue_R2_categories": p["sueK"], "var_36m": v36["var_window"], "windows_36m": v36["windows"],
           "drift_month1_share": float(drift_month(p["vt"], 1) * (1 - p["vt"])),     # delta-bar_1 / (1 - kappa): month 1's share of the drift
           "lagged_breadth": CUT["lagged_breadth"]["lagged_over_current"], "half_life": float(np.log(2) / -np.log(p["rho"])),
           **{k: {} for k in ("valuation_r2_state", "eq_D6", "cL_share_day1", "cL_share_month1",
                              "memory_share_level_month1", "memory_share_lagged_level", "IL_over_F", "month2_refit",
                              "f_day1_over_days2_10", "f_ic_day1", "f_ic_days2_10")}}
    vt = valuation_terms(p)
    for m in PANELS:
        e, ts, a = CODE[m], d["ts"][m], attribution(m)
        out["eq_D6"][m] = {"lhs": (1 - phi(p["a_" + e])) * p["sigK2"],
                           "rhs": phi(p["q_" + e]) * p["rnu"] * p["sigK2"] + phi(p["g_" + e]) * (p["Bz"] + p["Bx"])}
        out["valuation_r2_state"][m] = r2_log_pb(m) / (vt["corr2"] * vt["third"][m])
        out["cL_share_day1"][m] = float(ts["cL"]["mean"][1] / ts["yhat"]["mean"][1])
        out["cL_share_month1"][m] = float(a["cL"]["mean_%/yr"][1] / a["total"]["mean_%/yr"][1])
        out["memory_share_level_month1"][m] = lag[m]["memory_share"]
        out["memory_share_lagged_level"][m] = lag[m]["memory_share_lagged"]
        r = paper("4_didisheim/attribution", m, "attribution").query("tau == 1").set_index(["weights", "part"])["mean_%/yr"]
        out["month2_refit"][m] = {k: {"fixed": float(r["fixed", k]), "refit": float(r["re-optimized", k])} for k in ("yF", "uD")}
    for m in NAME:
        t = paper("4_horizon", m, "term_structure").set_index(["part", "h"])["mean"]
        mo = attribution(m)
        out["IL_over_F"][m] = {"daily_2_10": float(t["uL"].loc[2:DAY_H].mean() / t["yF"].loc[2:DAY_H].mean()),
                               "monthly_2_36": float(mo["uL"]["mean_%/yr"].loc[2:MO_H].mean() / mo["yF"]["mean_%/yr"].loc[2:MO_H].mean())}
        f = paper("4_horizon", m, "term_structure").query("part == 'yF'").set_index("h")
        out["f_day1_over_days2_10"][m] = float(f["mean"][1] / f["mean"].loc[2:DAY_H].mean())
        out["f_ic_day1"][m], out["f_ic_days2_10"][m] = float(f["IC_%"][1]), float(f["IC_%"].loc[2:DAY_H].mean())
    return out


def provenance():
    md5 = lambda f: hashlib.md5(Path(f).read_bytes()).hexdigest()
    files = [HERE / "calibration.py", HERE / "calibration_figures.py", CUT_FILE, *sorted(INPUTS)]
    return {str(Path(f).relative_to(HERE.parent)): md5(f) for f in dict.fromkeys(files)}


def dump(obj, name):
    json.dump(obj, open(HERE / name, "w"), indent=1, allow_nan=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def stored(p):
    """the parameter dict without the variant's boolean switch"""
    return {k: v for k, v in p.items() if not isinstance(v, bool)}


# ---- alternative specifications (Table D5) ------------------------------------------------------------------------------
VARIANTS = (("base", "Baseline", {}),
            ("mem_cf", None, {}),                                # memory of cash-flow news net of the state's discount-rate news
            ("mo_h1", "$h=1$ targeted for $F$ and $I_L$", {"mo_h1": ("yF", "uL")}),
            ("no_shares", "$\\varpi_F=1$", {"no_shares": True}),
            ("r2zx", None, {}),                                  # R^2(z|X) from RoBERTa's valuation R2 at the baseline (eq. D20)
            ("no_perm", "$\\Var(\\bar z)=0$", {"no_perm": True}),
            ("ups_FL", "Separate $\\Upsilon$ for $F$ and $I_L$", {"ups_FL": True}))


def variants():
    """the joint search under each of VARIANTS: look-ahead shares of F and E_L before each encoder's split, Llama-2's phi(eta), the
    half-life, the permanent share, the criterion, the Sobol starts (of SOBOL_N) reaching the minimum and the parameters at a bound;
    cal_variants.tex and .json. The data-driven values (the cash-flow share 1 - dr_share, R^2(z|X)) come from the baseline, which
    must equal calibration.json"""
    d, res = load(), {}
    base = json.load(open(HERE / "calibration.json"))
    for key, lab, spec in VARIANTS:
        spec = dict(spec)
        if key == "mem_cf":
            spec["mem_scale"] = 1 - dr_share(res["base"]["params"])
            res["mem_cf_window_years"] = N_THETA / 12 / spec["mem_scale"]  # the same model as sigma_theta^2 with this window
            lab = f"${num(spec['mem_scale'])}\\,\\sigma_\\theta^2$ in memory"
        if key == "r2zx":
            vt = valuation_terms(res["base"]["params"])
            by = {m: r2_log_pb(m) / (vt["corr2"] * vt["third"][m]) for m in PANELS}
            spec["r2zx"], res["r2zx_by_encoder"] = by[RB], by     # RoBERTa's: Llama-2's firm identity may hold remembered valuations
            lab = f"$R^2(z\\mid X)={num(spec['r2zx'])}$"
        p, fit, _ = search(d, spec)
        if key == "base":
            bp = base["params"]
            assert set(stored(p)) == set(bp) and all(abs(p[k] - bp[k]) <= 1e-9 * max(1.0, abs(bp[k])) for k in bp), \
                "baseline differs from calibration.json"
            assert abs(fit["criterion"] - base["fit"]["criterion"]) <= 1e-9 * base["fit"]["criterion"]
        res[key] = {"label": lab, "spec": spec, "params": stored(p), "fit": fit,
                    "half_life": float(np.log(2) / -np.log(p["rho"])), "perm": p["Bz"] / (p["Bz"] + p["Bx"]),
                    "phi_eta_L": phi(p["eta"]), "phi_eta_R": phi(p["etaR"]),
                    **{f"la_{e}_{part}": cutoff_ratio(p, part, e)[1] for e in "LR" for part in ("F", "lev")}}
        print(key, {k: round(res[key][k], 4) for k in ("half_life", "perm", "la_L_F", "la_L_lev", "la_R_F", "la_R_lev", "phi_eta_L")},
              fit["starts_at_min"], fit["at_bounds"], flush=True)
    keys = [k for k, _, _ in VARIANTS]
    res["ranges"] = {q: (min(res[k][q] for k in keys), max(res[k][q] for k in keys))
                     for q in ("la_L_F", "la_L_lev", "la_R_F", "la_R_lev", "half_life", "perm", "phi_eta_L")}
    res["provenance"] = provenance()
    variants_table(res)
    dump(res, "cal_variants.json")


def variants_table(res):
    """Table D5 from the variants' results: memory shares, phi(eta), half-life, rho_z(36) = Var(zbar) + rho_m^36 Var(x) (the firm state's
    autocorrelation at 36 months), loss (the minimized criterion) and the Sobol starts reaching the minimum"""
    rz = lambda v: v["perm"] + (1 - v["perm"]) * v["params"]["rho"] ** MO_H
    rows = [[res[k]["label"], *[num(100 * res[k][f"la_{e}_{q}"], 0) for e in "LR" for q in ("F", "lev")],
             num(res[k]["phi_eta_L"]), num(res[k]["half_life"], 0), num(rz(res[k])),
             num(res[k]["fit"]["criterion"]), f"{res[k]['fit']['starts_at_min']}/{res[k]['fit']['starts']}"]
            for k, _, _ in VARIANTS]
    write_tab("cal_variants.tex", "lccccccccc",
              [" & \\multicolumn{2}{c}{Llama-2 memory (\\%)} & \\multicolumn{2}{c}{RoBERTa memory (\\%)} & Llama-2 & Half-life of $x$ & & & Convergence to \\\\",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
               "Specification & $F$ & $E_L$ & $F$ & $E_L$ & $\\phi(\\eta)$ & (months) & $\\rho_z(36)$ & Loss & same minimum \\\\"],
              [" & ".join(r) + " \\\\" for r in rows])


# ---- main ------------------------------------------------------------------------------------------------------------------
def main():
    d = load()
    p, fit, mom = search(d)
    enc, lag = encoders(p), lagged_level(p)
    cut = cutoff_table(p)
    st = check_stats(d, p, enc, lag, mom)
    params_table(p, enc, lag)
    shares_table()
    targets_table(mom, fit)
    checks_table(st)
    sys.dont_write_bytecode = True                                         # no __pycache__ next to the paper's files
    import calibration_figures
    lic = level_ic_by_horizon()
    calibration_figures.draw(d, p, mom, sys.modules[__name__], HERE)
    calibration_figures.cutoff_figure(lic, cut, sys.modules[__name__], HERE / "cal_cutoff_ic.pdf")
    calibration_figures.cutoff_figure(lic, cut, sys.modules[__name__], HERE / "cal_cutoff_placebo.pdf", panels="BC")
    mom.to_csv(HERE / "calibration_fit.csv", index=False)
    dump({"params": stored(p), "fit": fit, "cutoff": cut, "level_ic_by_horizon": lic, "encoders": enc, "lagged_level": lag, "checks": st,
          "text_numbers": text_numbers(d, p, fit, lag), "provenance": provenance()}, "calibration.json")


if __name__ == "__main__":
    if sys.argv[1:] == ["variants"]:
        variants()
    elif sys.argv[1:] == ["variants-table"]:                               # Table D5 from cal_variants.json, no search
        variants_table(json.load(open(HERE / "cal_variants.json")))
    else:
        main()
