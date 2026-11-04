"""Closed-form results of Section 3 (linear Gaussian discriminant model)."""
import numpy as np


def logodds_exact(h, J, nu, D, pi1):
    """Theorem 1: posterior log-odds from the information statistic (h, J).
    nu, D: lists [nu_0, nu_1], [D_0, D_1]."""
    q = len(h)

    def g(k):
        Dinv = np.linalg.inv(D[k]); a = h + Dinv @ nu[k]
        return (-0.5 * np.linalg.slogdet(np.eye(q) + D[k] @ J)[1]
                + 0.5 * a @ np.linalg.solve(J + Dinv, a) - 0.5 * nu[k] @ Dinv @ nu[k])
    return np.log(pi1 / (1 - pi1)) + g(1) - g(0)


def eb_coefficient(J, D, nu0, nu1, pi1):
    """Corollary 2: coefficient of the pooled empirical Bayes estimate in the log-odds (common D)."""
    dnu = nu1 - nu0
    Dbar = D + pi1 * (1 - pi1) * np.outer(dnu, dnu)
    return (J + np.linalg.inv(Dbar)) @ np.linalg.solve(J + np.linalg.inv(D), np.linalg.solve(D, dnu))


def calibration_slope_misspecified(D, S_working, S_true):
    """Proposition 2 (q = 1, common D): calibration slope of the working discriminant model."""
    return (D + S_working) / (D + S_true)
