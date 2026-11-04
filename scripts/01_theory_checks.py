"""Numerical checks of Section 3: Theorem 1 against the direct Gaussian computation, Corollary 2,
the calibration curve of Corollary 3 under the parameters of simulation regime M (Table 2, first row; closed form and Monte Carlo),
and the autoregressive example after Proposition 2."""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LogisticRegression
import _common  # noqa: F401
from dyndisc.theory import logodds_exact, eb_coefficient, calibration_slope_misspecified

rng = np.random.default_rng(0)
nu = [np.array([2.3, -0.30]), np.array([2.6, -0.10])]
D = [np.array([[0.36, -0.02], [-0.02, 0.007]]), np.array([[0.41, -0.035], [-0.035, 0.023]])]
worst = 0.0
for _ in range(200):
    n = rng.integers(1, 7); t = np.sort(rng.choice(np.arange(2, 8), n, replace=False)).astype(float)
    Z = np.c_[np.ones(n), t]; L = 0.04 * np.eye(n); y = Z @ nu[1] + rng.normal(0, 0.5, n)
    direct = np.log(0.32 / 0.68) + mvn(Z @ nu[1], Z @ D[1] @ Z.T + L).logpdf(y) - mvn(Z @ nu[0], Z @ D[0] @ Z.T + L).logpdf(y)
    Li = np.linalg.inv(L)
    worst = max(worst, abs(direct - logodds_exact(Z.T @ Li @ y, Z.T @ Li @ Z, nu, D, 0.32)))
print(f"Theorem 1: max |direct - formula| over 200 designs = {worst:.2e}")

Dc = np.array([[0.38, -0.02], [-0.02, 0.012]]); pi1 = 0.32; nub = (1 - pi1) * nu[0] + pi1 * nu[1]
Dbar = Dc + pi1 * (1 - pi1) * np.outer(nu[1] - nu[0], nu[1] - nu[0]); worst = 0.0
for n in (1, 2, 4, 6):
    t = np.arange(2, 2 + n).astype(float); Z = np.c_[np.ones(n), t]; L = 0.04 * np.eye(n); Li = np.linalg.inv(L); J = Z.T @ Li @ Z
    al = eb_coefficient(J, Dc, nu[0], nu[1], pi1)
    ys = [Z @ nu[0] + rng.normal(0, .3, n) for _ in range(2)]
    lo = [mvn(Z @ nu[1], Z @ Dc @ Z.T + L).logpdf(y) - mvn(Z @ nu[0], Z @ Dc @ Z.T + L).logpdf(y) for y in ys]
    bt = [np.linalg.solve(J + np.linalg.inv(Dbar), Z.T @ Li @ y + np.linalg.solve(Dbar, nub)) for y in ys]
    worst = max(worst, abs(lo[0] - lo[1] - al @ (bt[0] - bt[1])))
print(f"Corollary 2: max |log-odds difference - alpha'(EB difference)| = {worst:.2e}")

# Corollary 3 under regime M of the simulation (complete designs, large sample)
nu0 = np.array([2.445, -0.313]); nu1 = np.array([2.579, 0.008])
D0 = np.array([[0.332, -0.0145], [-0.0145, 0.00635]]); D1 = np.array([[0.411, -0.034], [-0.034, 0.0231]])
Dm = 0.68 * D0 + 0.32 * D1; s2 = 0.035; t = np.arange(2, 8.); N = 200000
rng = np.random.default_rng(5)
C = rng.random(N) < pi1; th = np.where(C[:, None], nu1, nu0) + rng.multivariate_normal([0, 0], Dm, N)
Y = th[:, [0]] + th[:, [1]] * t + rng.normal(0, np.sqrt(s2), (N, 6))
nub = (1 - pi1) * nu0 + pi1 * nu1; Db = Dm + pi1 * (1 - pi1) * np.outer(nu1 - nu0, nu1 - nu0)


def eb(Ys, ts):
    Z = np.c_[np.ones(len(ts)), ts]; A = np.linalg.inv(Z.T @ Z / s2 + np.linalg.inv(Db))
    return (Ys @ Z / s2 + np.linalg.solve(Db, nub)) @ A.T


lr = LogisticRegression(C=1e6).fit(eb(Y, t), C); slopes = []
for s in range(2, 8):
    p = lr.predict_proba(eb(Y[:, :s - 1], t[:s - 1]))[:, 1]
    slopes.append(LogisticRegression(C=1e6).fit(np.log(p / (1 - p))[:, None], C).coef_[0, 0])
print("Corollary 3, regime M, weeks 2-7, Monte Carlo:", np.round(slopes, 2))
u = np.linalg.solve(Dm, nu1 - nu0); c = pi1 * (1 - pi1) / (1 + pi1 * (1 - pi1) * (nu1 - nu0) @ u)
sig = lambda J: 1 - c * u @ np.linalg.solve(J + np.linalg.inv(Dm), u)
Jw = lambda n: np.c_[np.ones(n), t[:n]].T @ np.c_[np.ones(n), t[:n]] / s2
print("Corollary 3, regime M, weeks 2-7, closed form sigma(J)/sigma(J*):", np.round([sig(Jw(n)) / sig(Jw(6)) for n in range(1, 7)], 2))

# Remark after Proposition 2: least squares slope (intercept estimated per subject), AR(1) errors ignored
out = []
for n in range(2, 7):
    t = np.arange(2, 2 + n).astype(float); X = np.c_[np.ones(n), t]
    Rm = 0.6 ** np.abs(np.subtract.outer(t, t)); A = np.linalg.solve(X.T @ X, X.T)
    out.append(calibration_slope_misspecified(0.007, 0.04 * np.linalg.inv(X.T @ X)[1, 1], 0.04 * (A @ Rm @ A.T)[1, 1]))
print("Proposition 2 example (D = 0.007, sigma^2 = 0.04, phi = 0.6), n = 2..6:", np.round(out, 2))
