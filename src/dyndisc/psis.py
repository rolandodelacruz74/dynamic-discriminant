"""Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao and Gabry, 2024),
following the reference implementation (Zhang and Stephens, 2009, GPD fit)."""
import numpy as np


def _gpdfit(x):
    """Empirical Bayes estimate of generalized Pareto (k, sigma) for exceedances x (sorted)."""
    prior_bs, prior_k = 3, 10
    n = len(x)
    m_est = 30 + int(n ** 0.5)
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b_ary /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b_ary += 1 / x[-1]
    k_ary = np.log1p(-b_ary[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]; b_ary = b_ary[real_idxs]
    weights /= weights.sum()
    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * x).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def _gpinv(probs, kappa, sigma):
    if kappa == 0:
        return -sigma * np.log1p(-probs)
    return sigma * np.expm1(-kappa * np.log1p(-probs)) / kappa


def psis(log_ratios):
    """Return (normalized log weights, khat) for a 1-d array of log importance ratios."""
    lw = np.asarray(log_ratios, float).copy()
    S = lw.size
    lw -= lw.max()
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    idx = np.argsort(lw)
    cutoff = lw[idx[-M - 1]]
    tail_idx = idx[-M:]
    khat = np.inf
    if M >= 5:
        x = np.exp(lw[tail_idx]) - np.exp(cutoff)
        x = np.sort(x)
        if np.all(x > 0):
            khat, sigma = _gpdfit(x)
            if np.isfinite(khat):
                p = (np.arange(1, M + 1) - 0.5) / M
                smoothed = np.log(_gpinv(p, khat, sigma) + np.exp(cutoff))
                lw[tail_idx] = smoothed      # tail_idx is in increasing order of lw
                lw[lw > 0] = 0
    lw -= np.logaddexp.reduce(lw)
    return lw, khat
