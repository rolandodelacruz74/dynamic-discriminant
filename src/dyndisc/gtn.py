"""GTN application (Section 6.1): weekly log10 hCG at weeks 2-7 after evacuation of a molar pregnancy.

Discriminant model (DF), one fit per class:
    log hCG_it = b0 + b1 t + u0i + u1i t + e_it,  (u0, u1) ~ N2(0, D),  e ~ N(0, s^2),
with the random effects integrated out exactly (padded batch of at most six measurements).
Joint model (JM), Bayesian version of the JMB of Dandis et al. (2020) without age:
    a single linear mixed model for all women and logit P(GTN = 1 | u) = g0 + g1 u0/sd0 + g2 u1/sd1.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .psis import psis
from .metrics import expit_normal_mean

WEEKS = range(2, 8)


def load(path):
    """Read the registry file of Dandis et al. (2020) (999 = missing). Women with no measurement are dropped."""
    d = pd.read_csv(path, na_values=["999"]).rename(columns={"PTD": "C", "weeknr": "t"})
    return d.dropna(subset=["loghcg"]).sort_values(["id", "t"])[["id", "t", "loghcg", "C"]]


def fit(d, model, draws=1000, tune=1500, chains=4, cores=2, seed=7):
    """model: 'df0', 'df1' (discriminant model for one class) or 'jm'. Returns (draws dict, diagnostics dict)."""
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.tensor.slinalg import cholesky, solve_triangular
    if model.startswith("df"):
        d = d[d.C == int(model[-1])]
    ids = np.sort(d.id.unique()); d = d.assign(ix=d.id.map({p: i for i, p in enumerate(ids)}))
    P = len(ids)
    with pm.Model():
        b = pm.Normal("b", [2.5, 0.0], [2.0, 1.0], shape=2)
        LD, _, _ = pm.LKJCholeskyCov("LD", n=2, eta=2, sd_dist=pm.HalfNormal.dist(np.array([1.0, 0.5])), compute_corr=True)
        D = pm.Deterministic("D", LD @ LD.T)
        s = pm.HalfNormal("s", 0.5)
        if model.startswith("df"):
            dmax = 6
            Y = np.zeros((P, dmax)); T = np.zeros((P, dmax)); M = np.zeros((P, dmax))
            for i, (_, g) in enumerate(d.groupby("ix")):
                n = len(g); Y[i, :n] = g.loghcg.values; T[i, :n] = g.t.values; M[i, :n] = 1
            Z = np.stack([M, T * M], axis=2)
            V = pt.as_tensor(Z) @ D @ pt.as_tensor(Z.transpose(0, 2, 1)) + s ** 2 * np.eye(dmax)[None] * M[:, :, None]
            V = V * (M[:, :, None] * M[:, None, :]) + np.eye(dmax)[None] * (1 - M)[:, :, None]
            L = cholesky(V)
            a = solve_triangular(L, ((Y - (b[0] + b[1] * T)) * M)[:, :, None], lower=True)[:, :, 0]
            pm.Potential("lik", pt.sum(-0.5 * pt.sum(a ** 2, axis=1) - pt.sum(pt.log(pt.diagonal(L, axis1=1, axis2=2)), axis=1)))
        else:
            z = pm.Normal("z", 0, 1, shape=(P, 2)); u = z @ LD.T
            ix = d.ix.values; t = d.t.values.astype(float)
            pm.Normal("y", b[0] + b[1] * t + u[ix, 0] + u[ix, 1] * t, s, observed=d.loghcg.values)
            g = pm.Normal("g", 0, 2.5, shape=3)
            pm.Bernoulli("c", logit_p=g[0] + g[1] * u[:, 0] / pt.sqrt(D[0, 0]) + g[2] * u[:, 1] / pt.sqrt(D[1, 1]),
                         observed=d.groupby("ix").C.first().values)
        idata = pm.sample(draws, tune=tune, chains=chains, cores=cores, target_accept=0.95, random_seed=seed, progressbar=False)
    names = ["b", "D", "s"] + (["g"] if model == "jm" else [])
    post = idata.posterior
    out = {v: post[v].values.reshape((-1,) + post[v].values.shape[2:]) for v in names}
    return out, diagnostics(idata, names, latent=["z"] if model == "jm" else [])


def diagnostics(idata, names, latent=()):
    import pymc as pm
    summ = pm.stats.summary(idata, var_names=list(names))
    res = {"divergences": int(idata.sample_stats["diverging"].values.sum()),
           "max_rhat": float(summ.r_hat.astype(float).max()), "min_ess_bulk": float(summ.ess_bulk.astype(float).min()),
           "min_ess_tail": float(summ.ess_tail.astype(float).min()),
           "ebfmi": [float(np.mean(np.diff(e) ** 2) / np.var(e)) for e in idata.sample_stats["energy"].values],
           "treedepth_hits": int((idata.sample_stats["tree_depth"].values >= 10).sum())}
    if latent:
        zs = pm.stats.summary(idata, var_names=list(latent))
        res.update(latent_max_rhat=float(zs.r_hat.astype(float).max()), latent_min_ess_bulk=float(zs.ess_bulk.astype(float).min()))
    return res, summ


def _marg(par, t):
    Z = np.c_[np.ones(len(t)), t]
    mean = par["b"][:, 0][:, None] + par["b"][:, 1][:, None] * t[None, :]
    V = Z[None] @ par["D"] @ Z.T[None] + (par["s"] ** 2)[:, None, None] * np.eye(len(t))[None]
    return mean, V, Z


def _mvn(y, mean, V):
    L = np.linalg.cholesky(V); a = np.linalg.solve(L, (y[None] - mean)[..., None])[..., 0]
    return -0.5 * (a ** 2).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1) - 0.5 * len(y) * np.log(2 * np.pi)


def loo_predictions(d, F, thin=2):
    """Leave-one-out (PSIS) probabilities of GTN at each landmark week for DF and JM.
    F: dict with draws for 'df0', 'df1', 'jm'. Returns (pred dict model -> {(i, s): p}, khat dict, subject classes)."""
    F = {m: {v: a[::thin] for v, a in F[m].items()} for m in F}
    pats = [(i, g.C.iloc[0], g.t.values.astype(float), g.loghcg.values) for i, g in d.groupby("id")]
    C = np.array([p[1] for p in pats]); N = len(pats); nk = np.array([(C == 0).sum(), (C == 1).sum()])
    pred = {"DF": {}, "JM": {}}; kh = {"DF": [], "JM": []}
    J = F["jm"]; sd = np.sqrt(np.stack([J["D"][:, 0, 0], J["D"][:, 1, 1]], 1)); aa = J["g"][:, 1:] / sd
    for i, (_, c, t, y) in enumerate(pats):
        pre = {s: (t[t <= s], y[t <= s]) for s in WEEKS if (t <= s).any()}
        lp = {k: {s: _mvn(yy, *_marg(F[f"df{k}"], tt)[:2]) for s, (tt, yy) in pre.items()} for k in (0, 1)}
        lw, k_ = psis(-lp[c][max(pre)]); kh["DF"].append(k_)
        n_ = nk - np.eye(2, dtype=int)[c]
        for s in pre:
            ll = [np.log(n_[k] / (N - 1)) + (logsumexp(lp[k][s] + lw) if k == c else logsumexp(lp[k][s]) - np.log(len(lp[k][s])))
                  for k in (0, 1)]
            pred["DF"][(i, s)] = expit(ll[1] - ll[0])
        terms = {}
        for s, (tt, yy) in pre.items():
            mean, V, Z = _marg(J, tt)
            K = J["D"] @ Z.T[None] @ np.linalg.inv(V)
            m = np.einsum("sqd,sd->sq", K, yy[None] - mean); Pc = J["D"] - K @ Z[None] @ J["D"]
            mu = J["g"][:, 0] + (aa * m).sum(1); s2 = np.einsum("sq,sqr,sr->s", aa, Pc, aa)
            terms[s] = (_mvn(yy, mean, V), expit_normal_mean(mu, np.sqrt(s2)))
        lpf, p1f = terms[max(pre)]
        lw, k_ = psis(-(lpf + np.log(np.where(c == 1, p1f, 1 - p1f)))); kh["JM"].append(k_)
        for s in pre:
            # posterior predictive: draws weighted by p(y^(s) | psi) as well as by the leave-one-out weights
            pred["JM"][(i, s)] = np.exp(logsumexp(lw + terms[s][0] + np.log(terms[s][1])) - logsumexp(lw + terms[s][0]))
    return pred, {m: np.array(v) for m, v in kh.items()}, C
