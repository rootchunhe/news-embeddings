"""The ridge kernel shared by the CKX replication (2_replication/ckx/01_regress.py) and the paper (3_paper/lib/ridge.py): moments of
the centred features from summed statistics and the coefficients for a list of per-observation penalties, in float64."""
import numpy as np


def moments(n, sx, sxx, sxy, sy):
    """mean, target mean, covariance of the centred features (X'X / n) and their covariance with the target (X'(y - ybar) / n)"""
    mu = sx / n
    return mu, sy / n, (sxx - n * np.outer(mu, mu)) / n, (sxy - mu * sy) / n


def coefficients(G, g, lams):
    """ridge coefficients (G + lam I)^-1 g for every per-observation penalty in lams, from one eigendecomposition of G: d x len(lams)"""
    evals, evecs = np.linalg.eigh(G)
    Vg = evecs.T @ g
    return np.column_stack([evecs @ (Vg / (evals + lam)) for lam in lams])
