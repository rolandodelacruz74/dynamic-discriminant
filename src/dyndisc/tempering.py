"""Tempered predictions (Section 4.3): logit p* = a + rho * logit p.

single:   one (a, rho) for all landmarks
separate: landmark-specific (a_s, rho_s)
shrunk:   landmark-specific (a_s, rho_s) shrunk towards their mean by the method of moments of
          DerSimonian and Laird (1986), coefficient by coefficient
All are estimated by cross-fitting across subjects, so that a subject's label never enters its own
tempered prediction."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def fit_with_cov(x, y):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(x)[:, None], y)
    th = np.r_[m.intercept_[0], m.coef_[0, 0]]
    X = np.c_[np.ones(len(x)), x]; p = expit(X @ th)
    return th, np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))


def shrink(est):
    """est: dict landmark -> (theta, cov). Returns (dict landmark -> shrunk theta, tau2 of rho)."""
    keys = sorted(est); out = {s: np.zeros(2) for s in keys}; tau2s = []
    for j in range(2):
        th = np.array([est[s][0][j] for s in keys]); v = np.array([est[s][1][j, j] for s in keys]); w = 1 / v
        muf = (w * th).sum() / w.sum(); Q = (w * (th - muf) ** 2).sum()
        tau2 = max(0.0, (Q - (len(th) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        w2 = 1 / (v + tau2); mu = (w2 * th).sum() / w2.sum(); tau2s.append(tau2)
        for k, s in enumerate(keys):
            out[s][j] = mu + tau2 / (tau2 + v[k]) * (th[k] - mu)
    return out, tau2s[1]


def cross_fitted(subject, landmark, logit_p, y, n_subjects, subject_class, folds=10, seed=1):
    """Cross-fitted tempered probabilities for every (subject, landmark) row.
    subject: integer index 0..n_subjects-1 per row. Returns dict method -> array of probabilities."""
    subject = np.asarray(subject); landmark = np.asarray(landmark); x = np.asarray(logit_p, float); y = np.asarray(y)
    hs = sorted(set(landmark.tolist()))
    out = {m: np.full(len(x), np.nan) for m in ("single", "separate", "shrunk")}
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n_subjects), subject_class):
        a = np.isin(subject, tr); b = np.isin(subject, te)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(x[a, None], y[a]); out["single"][b] = m.predict_proba(x[b, None])[:, 1]
        est = {h: fit_with_cov(x[a & (landmark == h)], y[a & (landmark == h)]) for h in hs}
        sh, _ = shrink(est)
        for h in hs:
            c = b & (landmark == h)
            out["separate"][c] = expit(est[h][0][0] + est[h][0][1] * x[c])
            out["shrunk"][c] = expit(sh[h][0] + sh[h][1] * x[c])
    return out


# ----------------------------------------------------------------------------------------------
# Multivariate shrinkage of landmark-specific coefficients with bootstrap covariance (DF-Th)
def _logit_coef(x, y):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(x)[:, None], y)
    return np.r_[m.intercept_[0], m.coef_[0, 0]]


def _reml_T(th, V, S):
    """REML estimate of the 2x2 between-landmark covariance T in th = (1_S kron I_2) mu + delta + e,
    delta_s ~ N(0, T) independent, e ~ N(0, V). th: (2S,), V: (2S, 2S)."""
    from scipy.optimize import minimize
    X = np.kron(np.ones((S, 1)), np.eye(2))

    def unpack(p):
        L = np.array([[np.exp(p[0]), 0.0], [p[1], np.exp(p[2])]]); return L @ L.T

    def nll(p):
        Sig = np.kron(np.eye(S), unpack(p)) + V
        Si = np.linalg.inv(Sig); A = X.T @ Si @ X; mu = np.linalg.solve(A, X.T @ Si @ th); r = th - X @ mu
        return 0.5 * (np.linalg.slogdet(Sig)[1] + np.linalg.slogdet(A)[1] + r @ Si @ r)

    best = None
    for start in (np.array([-6.0, 0.0, -6.0]), np.array([-2.0, 0.0, -2.0])):
        r = minimize(nll, start, method="L-BFGS-B", bounds=[(-15, 5), (-10, 10), (-15, 5)])
        best = r if best is None or r.fun < best.fun else best
    T = unpack(best.x); Sig = np.kron(np.eye(S), T) + V; Si = np.linalg.inv(Sig)
    mu = np.linalg.solve(X.T @ Si @ X, X.T @ Si @ th)
    return T, mu, Sig


def shrink_mv(x, y, subject, landmark, B=200, seed=0):
    """Landmark-specific tempering coefficients (a_s, rho_s) shrunk towards a common value.
    The joint covariance V of the estimates across landmarks and between intercept and slope is
    estimated by a bootstrap over subjects; the between-landmark covariance T and the common value mu
    are estimated by restricted maximum likelihood, and the shrunk coefficients are the best linear
    unbiased predictors mu + T_S Sigma^{-1} (theta_hat - 1 mu).
    Returns (dict landmark -> (a, rho), T)."""
    x = np.asarray(x, float); y = np.asarray(y); subject = np.asarray(subject); landmark = np.asarray(landmark)
    hs = sorted(set(landmark.tolist())); S = len(hs)
    th = np.concatenate([_logit_coef(x[landmark == h], y[landmark == h]) for h in hs])
    subs = np.unique(subject); rng = np.random.default_rng(seed)
    rows_of = {s: np.where(subject == s)[0] for s in subs}
    boot = []
    for _ in range(B):
        idx = np.concatenate([rows_of[s] for s in rng.choice(subs, len(subs), replace=True)])
        xb, yb, hb = x[idx], y[idx], landmark[idx]
        try:
            if any(len(np.unique(yb[hb == h])) < 2 for h in hs):
                continue
            boot.append(np.concatenate([_logit_coef(xb[hb == h], yb[hb == h]) for h in hs]))
        except ValueError:
            continue
    V = np.cov(np.array(boot).T)
    T, mu, Sig = _reml_T(th, V, S)
    delta = np.kron(np.eye(S), T) @ np.linalg.solve(Sig, th - np.kron(np.ones(S), mu))
    out = {h: mu + delta[2 * k:2 * k + 2] for k, h in enumerate(hs)}
    return out, T


def fit_tempering(x, y, subject, landmark, B=200, seed=0):
    """All three tempering rules fitted to (x, y): returns dict method -> function(x_new, landmark_new)."""
    x = np.asarray(x, float); y = np.asarray(y); landmark = np.asarray(landmark)
    hs = sorted(set(landmark.tolist()))
    g = _logit_coef(x, y)
    sep = {h: _logit_coef(x[landmark == h], y[landmark == h]) for h in hs}
    sh, T = shrink_mv(x, y, subject, landmark, B=B, seed=seed)
    f = lambda co: (lambda xn, hn: expit(np.array([co(h)[0] for h in hn]) + np.array([co(h)[1] for h in hn]) * np.asarray(xn)))
    return {"single": f(lambda h: g), "separate": f(lambda h: sep[h]), "shrunk": f(lambda h: sh[h])}, {"single": g, "separate": sep, "shrunk": sh, "T": T}
