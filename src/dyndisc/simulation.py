"""Simulation study for paper 1 (Section 5).

Data: weekly log hCG at t = 2..7, class C ~ Bernoulli(0.32), MCAR missingness 40% per week
(women with no measurement are discarded), linear trajectories by class with random intercept/slope.
Factors (2 x 2 x 2 x 2):
  regime  'M': classes differ in mean trajectory (GTN values), common D (average of GTN D_k)
          'H': mean difference reduced to 25%, D_1 = 3.6 D_0 (GTN slope-variance ratio)
          'J': generated from the joint model: theta ~ N(m, G) with the pooled moments of regime M,
               logit P(C = 1 | theta) = g0 + gamma'(theta - m), gamma = D^{-1} Delta nu of regime M,
               g0 such that P(C = 1) = 0.32 (only with iid errors and Gaussian random effects)
  errors  'iid' or 'ar' (AR(1), corr 0.6, same marginal variance)
  re      'gauss' or 'mix': u = L (sqrt(1-w) z + sqrt(w) B), B in {-1,1}^2, w = 0.8
          (same mean and covariance, bimodal)
  n       150 or 450 training women; test set of 3000 women.
Methods (maximum likelihood with exact marginal likelihood):
  TS   two-stage: pooled LMM, empirical Bayes estimates from complete histories, logistic regression
  TSL  two-stage by landmark: logistic regression at landmark s on empirical Bayes estimates computed
       from the training histories truncated at s
  JM   joint model: pooled LMM + logit P(C|u) linear in standardised u, P(C|y) by 60-node Gauss-Hermite quadrature
  JMQ  joint model with logit P(C|u) quadratic in standardised u; E{expit(.) | y} by 7 x 7 Gauss-Hermite
  DF   discriminant model: class-specific LMM (Gaussian, independent errors), Theorem 1
  DFT  DF tempered with one (a, rho), estimated on 5-fold out-of-fold training predictions
  DFTw DF tempered with landmark-specific (a, rho)
  DFTh DF tempered with landmark-specific (a, rho) shrunk towards a common value (tempering.shrink_mv:
       bootstrap covariance across landmarks, REML between-landmark covariance)
  OR   oracle: true P(C | y) under the generating model
Metrics on the test set by landmark week s = 2..7 (women with >= 1 measurement up to s):
  AUC, Brier score, calibration slope, calibration intercept (logistic recalibration with the predicted
  logit as offset), Brier excess over the oracle.
Use run_scenario(regime, errors, re, n_train, reps, seed); see scripts/02_simulation.py.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .tempering import shrink_mv

GH7 = np.polynomial.hermite_e.hermegauss(7)          # probabilists' Gauss-Hermite nodes and weights
GH60 = np.polynomial.hermite_e.hermegauss(60)

HEADER = "regime,errors,re,n,rep,week,method,auc,brier,slope,brier_excess,citl"


def run_scenario(REG, ERR, RE, NTR, R, SEED, verbose=False, start=0, callback=None, jmq_nodes=7, jmq_check=None, jm_nodes=60):
    """Run R replicates of one scenario; returns a list of CSV lines (see HEADER).
    start: number of replicates already done (their random draws are regenerated and skipped);
    callback: called with the lines of each finished replicate.
    jm_nodes: Gauss-Hermite nodes of the linear joint model (60 in the paper);
    jmq_nodes: Gauss-Hermite nodes per dimension for the quadratic joint model (7 in the paper);
    jmq_check: optional list; for each replicate, appends the JMQ negative log-likelihood at its estimate
    evaluated with jmq_nodes and with 25 x 25 nodes (accuracy of the objective)."""
    T = np.arange(2, 8.0); K = len(T); Zf = np.c_[np.ones(K), T]
    PI1, MISS, RHO_AR, W = 0.32, 0.40, 0.6, 0.8
    nu0 = np.array([2.445, -0.313]); nu1g = np.array([2.579, 0.008])
    D0g = np.array([[0.332, -0.0145], [-0.0145, 0.00635]]); D1g = np.array([[0.411, -0.034], [-0.034, 0.0231]])
    S2 = 0.035
    Dm = 0.68 * D0g + 0.32 * D1g
    if REG in ("M", "J"):
        NU = [nu0, nu1g]; D = [Dm, Dm]
    else:
        NU = [nu0, nu0 + 0.25 * (nu1g - nu0)]; D = [D0g, 3.6 * D0g]
    LD = [np.linalg.cholesky(d) for d in D]
    Re = S2 * (RHO_AR ** np.abs(np.subtract.outer(T, T)) if ERR == "ar" else np.eye(K))
    if REG == "J":                                      # joint-model generating mechanism
        dnu = nu1g - nu0; MJ = 0.68 * nu0 + 0.32 * nu1g; GJ = Dm + 0.68 * 0.32 * np.outer(dnu, dnu)
        GAM = np.linalg.solve(Dm, dnu); sdl = np.sqrt(GAM @ GJ @ GAM)
        from scipy.optimize import brentq
        G0 = brentq(lambda g: np.sum(GH60[1] * expit(g + sdl * GH60[0])) / np.sqrt(2 * np.pi) - PI1, -20, 20)
        LJ = np.linalg.cholesky(GJ)


    def simulate(n, rng):
        if REG == "J":
            th = MJ + rng.standard_normal((n, 2)) @ LJ.T
            C = (rng.random(n) < expit(G0 + (th - MJ) @ GAM)).astype(int)
            Y = th @ Zf.T + rng.multivariate_normal(np.zeros(K), Re, size=n)
            M = rng.random((n, K)) > MISS; keep = M.any(1); Y[~M] = np.nan
            return Y[keep], C[keep]
        C = (rng.random(n) < PI1).astype(int)
        z = rng.standard_normal((n, 2))
        if RE == "mix":
            z = np.sqrt(1 - W) * z + np.sqrt(W) * rng.choice([-1.0, 1.0], size=(n, 2))
        U = np.stack([LD[c] @ z[i] for i, c in enumerate(C)])
        mu = np.stack([Zf @ NU[c] for c in C]) + np.einsum("kq,nq->nk", Zf, U)
        Y = mu + rng.multivariate_normal(np.zeros(K), Re, size=n)
        M = rng.random((n, K)) > MISS
        keep = M.any(1)
        Y[~M] = np.nan
        return Y[keep], C[keep]


    def groups(Y):
        """Group rows by observation pattern: list of (rows, cols)."""
        obs = ~np.isnan(Y); out = {}
        for i, o in enumerate(obs):
            out.setdefault(tuple(o), []).append(i)
        return [(np.array(r), np.where(k)[0]) for k, r in out.items() if any(k)]


    def unpack(p):
        L = np.array([[np.exp(p[2]), 0], [p[3], np.exp(p[4])]])
        return p[:2], L @ L.T, np.exp(2 * p[5])


    def mvn_rows(Yo, mean, V):
        L = np.linalg.cholesky(V); a = np.linalg.solve(L, (Yo - mean).T)
        return -0.5 * (a ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * Yo.shape[1] * np.log(2 * np.pi)


    def lmm_ll(p, Y, G):
        b, Dk, s2 = unpack(p); ll = 0.0
        for rows, cols in G:
            Z = Zf[cols]; ll += mvn_rows(Y[np.ix_(rows, cols)], Z @ b, Z @ Dk @ Z.T + s2 * np.eye(len(cols))).sum()
        return ll


    def fit_lmm(Y):
        G = groups(Y)
        p0 = np.array([2.5, -0.2, np.log(0.6), 0.0, np.log(0.1), np.log(0.2)])
        def f(p):
            try:
                v = -lmm_ll(p, Y, G)
            except np.linalg.LinAlgError:
                return 1e10
            return v if np.isfinite(v) else 1e10
        r = minimize(f, p0, method="L-BFGS-B")
        return r.x


    def post_u(par, yo, cols):
        """Posterior mean and covariance of theta = b + u given observed y (pooled LMM)."""
        b, Dk, s2 = unpack(par); Z = Zf[cols]
        V = Z @ Dk @ Z.T + s2 * np.eye(len(cols)); Kg = Dk @ Z.T @ np.linalg.inv(V)
        return (yo - Z @ b) @ Kg.T, Dk - Kg @ Z @ Dk, V


    def jm_prob(par, g, Y, rows_cols):
        b, Dk, s2 = unpack(par); sd = np.sqrt(np.diag(Dk)); a = g[1:] / sd
        out = np.zeros(len(Y)); ll = 0.0
        for rows, cols in rows_cols:
            m, P, V = post_u(par, Y[np.ix_(rows, cols)], cols)
            mu = g[0] + m @ a; sdl = np.sqrt(a @ P @ a)
            out[rows] = (expit(mu[:, None] + sdl * GHJ[0][None, :]) * GHJ[1]).sum(1) / np.sqrt(2 * np.pi)
        return out


    def fit_jm(Y, C, par_init, g_init):
        G = groups(Y)
        def nll(x):
            par, g = x[:6], x[6:]
            try:
                p = np.clip(jm_prob(par, g, Y, G), 1e-9, 1 - 1e-9)
                v = -(lmm_ll(par, Y, G) + np.sum(C * np.log(p) + (1 - C) * np.log(1 - p)))
            except np.linalg.LinAlgError:          # degenerate covariance during the line search
                return 1e10
            return v if np.isfinite(v) else 1e10
        r = minimize(nll, np.r_[par_init, g_init], method="L-BFGS-B")
        return r.x[:6], r.x[6:]


    def _grid(n):
        gh = np.polynomial.hermite_e.hermegauss(n)
        return np.array([[a, b] for a in gh[0] for b in gh[0]]), np.array([a * b for a in gh[1] for b in gh[1]]) / (2 * np.pi)
    _gx, _gw = _grid(jmq_nodes)

    def jmq_prob(par, g, Y, rows_cols, grid=None):
        """g = (g0, a1, a2, b11, b12, b22) on standardised random effects."""
        b, Dk, s2 = unpack(par); sd = np.sqrt(np.diag(Dk))
        Bq = np.array([[g[3], g[4]], [g[4], g[5]]]); out = np.zeros(len(Y))
        gx, gw = (_gx, _gw) if grid is None else grid
        for rows, cols in rows_cols:
            m, P, V = post_u(par, Y[np.ix_(rows, cols)], cols)
            try:
                Lp = np.linalg.cholesky(P + 1e-12 * np.eye(2))
            except np.linalg.LinAlgError:          # numerically singular posterior covariance
                w_, V_ = np.linalg.eigh((P + P.T) / 2); Lp = V_ * np.sqrt(np.clip(w_, 0, None))
            U = m[:, None, :] + (gx @ Lp.T)[None, :, :]
            Zs = U / sd
            eta = g[0] + Zs @ g[1:3] + np.einsum("nki,ij,nkj->nk", Zs, Bq, Zs)
            out[rows] = (expit(eta) * gw).sum(1)
        return out

    def fit_jmq(Y, C, par_init, g_init):
        G = groups(Y)
        def nll(x):
            par, g = x[:6], x[6:]
            try:
                p = np.clip(jmq_prob(par, g, Y, G), 1e-9, 1 - 1e-9)
                v = -(lmm_ll(par, Y, G) + np.sum(C * np.log(p) + (1 - C) * np.log(1 - p)))
            except np.linalg.LinAlgError:          # degenerate covariance during the line search
                return 1e10
            return v if np.isfinite(v) else 1e10
        r = minimize(nll, np.r_[par_init, g_init, 0.0, 0.0, 0.0], method="L-BFGS-B")
        return r.x[:6], r.x[6:]

    def prefix(Y, s):
        Yp = Y.copy(); Yp[:, T > s] = np.nan; ok = (~np.isnan(Yp)).any(1)
        return Yp, ok


    def df_logodds(Yp, pars, pi1):
        lo = np.full(len(Yp), np.log(pi1 / (1 - pi1)))
        for rows, cols in groups(Yp):
            Z = Zf[cols]; l = []
            for p in pars:
                b, Dk, s2 = unpack(p); l.append(mvn_rows(Yp[np.ix_(rows, cols)], Z @ b, Z @ Dk @ Z.T + s2 * np.eye(len(cols))))
            lo[rows] += l[1] - l[0]
        return lo


    def oracle_logodds(Yp):
        if REG == "J":
            lo = np.zeros(len(Yp))
            for rows, cols in groups(Yp):
                Z = Zf[cols]; V = Z @ GJ @ Z.T + Re[np.ix_(cols, cols)]; Kg = GJ @ Z.T @ np.linalg.inv(V)
                mpost = MJ + (Yp[np.ix_(rows, cols)] - Z @ MJ) @ Kg.T; Pp = GJ - Kg @ Z @ GJ
                mu = G0 + (mpost - MJ) @ GAM; sd = np.sqrt(GAM @ Pp @ GAM)
                p = (expit(mu[:, None] + sd * GH60[0][None, :]) * GH60[1]).sum(1) / np.sqrt(2 * np.pi)
                lo[rows] = np.log(p / (1 - p))
            return lo
        lo = np.full(len(Yp), np.log(PI1 / (1 - PI1)))
        combos = [np.zeros(2)] if RE == "gauss" else [np.array(c) for c in itertools.product([-1.0, 1.0], repeat=2)]
        for rows, cols in groups(Yp):
            Z = Zf[cols]; yo = Yp[np.ix_(rows, cols)]; l = []
            for k in (0, 1):
                covu = LD[k] @ LD[k].T * ((1 - W) if RE == "mix" else 1.0)
                V = Z @ covu @ Z.T + Re[np.ix_(cols, cols)]
                comps = [mvn_rows(yo, Z @ (NU[k] + (np.sqrt(W) * LD[k] @ c if RE == "mix" else 0)), V) for c in combos]
                l.append(np.logaddexp.reduce(np.stack(comps), axis=0) - np.log(len(combos)))
            lo[rows] += l[1] - l[0]
        return lo


    def ts_fit(Y, C):
        par = fit_lmm(Y); E = np.zeros((len(Y), 2))
        for rows, cols in groups(Y):
            E[rows] = post_u(par, Y[np.ix_(rows, cols)], cols)[0]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(E, C)
        return par, lr


    def ts_prob(par, lr, Yp):
        E = np.zeros((len(Yp), 2))
        for rows, cols in groups(Yp):
            E[rows] = post_u(par, Yp[np.ix_(rows, cols)], cols)[0]
        return lr.predict_proba(E)[:, 1]



    def logit_fit_cov(x, y):
        """Logistic regression of y on (1, x): estimates and inverse Fisher information."""
        m = LogisticRegression(C=1e6, max_iter=1000).fit(x[:, None], y)
        th = np.r_[m.intercept_[0], m.coef_[0, 0]]
        X = np.c_[np.ones(len(x)), x]; p = expit(X @ th)
        return th, np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))


    def shrink_tempering(est):
        """Empirical Bayes (DerSimonian-Laird) shrinkage of landmark-specific (a, rho) towards their
        random-effects mean, coefficient by coefficient. est: dict s -> (theta_s, cov_s)."""
        keys = sorted(est); out = {s: np.zeros(2) for s in keys}
        for j in range(2):
            th = np.array([est[s][0][j] for s in keys]); v = np.array([est[s][1][j, j] for s in keys])
            w = 1 / v; mu_f = np.sum(w * th) / w.sum()
            Q = np.sum(w * (th - mu_f) ** 2); tau2 = max(0.0, (Q - (len(th) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
            w2 = 1 / (v + tau2); mu = np.sum(w2 * th) / w2.sum()
            for k, s in enumerate(keys):
                out[s][j] = mu + tau2 / (tau2 + v[k]) * (th[k] - mu)
        return out

    lgt = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))


    def citl(p, y):
        off = lgt(p); a = 0.0
        for _ in range(50):
            q = expit(a + off); a += np.sum(y - q) / np.sum(q * (1 - q))
        return a

    def metrics(p, y, por):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        slope = LogisticRegression(C=1e6, max_iter=1000).fit(lgt(p)[:, None], y).coef_[0, 0]
        return roc_auc_score(y, p), np.mean((p - y) ** 2), slope, np.mean((p - y) ** 2) - np.mean((por - y) ** 2), citl(p, y)


    GHJ = np.polynomial.hermite_e.hermegauss(jm_nodes)
    rng = np.random.default_rng(SEED)
    WEEKS = range(2, 8)
    lines = []
    for rep in range(R):
        Ytr, Ctr = simulate(NTR, rng); Yte, Cte = simulate(3000, rng)
        if rep < start:
            rng.permutation(len(Ytr)); continue
        n_before = len(lines)
        pi1 = Ctr.mean()
        pdf = [fit_lmm(Ytr[Ctr == k]) for k in (0, 1)]
        tpar, tlr = ts_fit(Ytr, Ctr)
        jpar, jg = fit_jm(Ytr, Ctr, tpar, np.r_[tlr.intercept_[0], tlr.coef_[0] * np.sqrt(np.diag(unpack(tpar)[1]))])
        qpar, qg = fit_jmq(Ytr, Ctr, jpar, jg)
        if jmq_check is not None:
            Gtr = groups(Ytr)
            def _nll(grid):
                p = np.clip(jmq_prob(qpar, qg, Ytr, Gtr, grid), 1e-9, 1 - 1e-9)
                return -(lmm_ll(qpar, Ytr, Gtr) + np.sum(Ctr * np.log(p) + (1 - Ctr) * np.log(1 - p)))
            jmq_check.append((rep, _nll(None), _nll(_grid(25)), np.r_[qpar, qg].tolist(), np.r_[jpar, jg].tolist()))
        tsl = {}
        for s in WEEKS:                                   # two-stage trained on histories truncated at s
            Yp, ok = prefix(Ytr, s); E = np.zeros((len(Yp), 2))
            for rows, cols in groups(Yp):
                E[rows] = post_u(tpar, Yp[np.ix_(rows, cols)], cols)[0]
            tsl[s] = LogisticRegression(C=1e6, max_iter=1000).fit(E[ok], Ctr[ok])
        # out-of-fold DF log-odds on the training set for tempering
        folds = rng.permutation(len(Ytr)) % 5
        oof = {s: np.full(len(Ytr), np.nan) for s in WEEKS}
        for f in range(5):
            tr, te = folds != f, folds == f
            pf = [fit_lmm(Ytr[tr & (Ctr == k)]) for k in (0, 1)]
            for s in WEEKS:
                Yp, ok = prefix(Ytr[te], s)
                lo = df_logodds(Yp, pf, Ctr[tr].mean()); lo[~ok] = np.nan
                oof[s][np.where(te)[0]] = lo
        xs = np.concatenate([oof[s] for s in WEEKS]); ys = np.concatenate([Ctr for s in WEEKS])
        okx = ~np.isnan(xs)
        tg = LogisticRegression(C=1e6, max_iter=1000).fit(xs[okx, None], ys[okx])
        tw = {s: LogisticRegression(C=1e6, max_iter=1000).fit(oof[s][~np.isnan(oof[s]), None], Ctr[~np.isnan(oof[s])]) for s in WEEKS}
        subj = np.concatenate([np.arange(len(Ytr)) for s in WEEKS]); lmk = np.concatenate([np.full(len(Ytr), s) for s in WEEKS])
        th_h, _ = shrink_mv(xs[okx], ys[okx], subj[okx], lmk[okx], B=200, seed=SEED * 1000 + rep)
        for s in WEEKS:
            Yp, ok = prefix(Yte, s); Yp, y = Yp[ok], Cte[ok]
            lo_df = df_logodds(Yp, pdf, pi1); lo_or = oracle_logodds(Yp); por = expit(lo_or)
            G = groups(Yp)
            P = {"TS": ts_prob(tpar, tlr, Yp), "TSL": ts_prob(tpar, tsl[s], Yp), "JM": jm_prob(jpar, jg, Yp, G),
                 "JMQ": jmq_prob(qpar, qg, Yp, G), "DF": expit(lo_df),
                 "DFT": tg.predict_proba(lo_df[:, None])[:, 1], "DFTw": tw[s].predict_proba(lo_df[:, None])[:, 1],
                 "DFTh": expit(th_h[s][0] + th_h[s][1] * lo_df), "OR": por}
            for m, p in P.items():
                a, b_, c, e, ci = metrics(p, y, por)
                lines.append(f"{REG},{ERR},{RE},{NTR},{rep},{s},{m},{a:.4f},{b_:.5f},{c:.3f},{e:.5f},{ci:.3f}")
        if callback is not None:
            callback(lines[n_before:])
        if rep == 0 and verbose:
            print("rep 0 done", flush=True)
    return lines
