"""Pregnancy application (Section 6.2): log10 beta-HCG and estradiol in early pregnancy after assisted
reproduction; class 0 = normal, 1 = abnormal pregnancy.

Mean functions (class k):
    HCG_ij = a_i / (1 + exp((c_i - t_ij)/scale_k)) + e_ij1,   E2_ij = e_i + slope_k t_ij + e_ij2,
    (e_ij1, e_ij2) ~ N2(0, Sigma_k).
Models fitted here:
    'het0', 'het1'  fixed inflection (c_i = infl_k), (a_i, e_i) ~ N2(nu_k, D_k); all parameters class specific
    'hom'           as 'het' with D and Sigma common to the classes
    'jm'            joint model, fixed inflection, one random-effects distribution, logistic outcome model
    'infl0','infl1' random inflection: (a_i, e_i, c_i) ~ N3((A_k, E_k, INFL_k), D_k)
    'inflmix1'      as 'infl1' with a two-component normal mixture for (a_i, e_i) (sensitivity analysis)
    'jm_infl'       joint model with random inflection
Missing markers: likelihood of the observed components only (missing at random given the class).
In the random-inflection models (a_i, e_i) are integrated out analytically given c_i (conditional linearity);
c_i is sampled (non-centred) when fitting and integrated on a fine grid when predicting.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .psis import psis
from .metrics import expit_normal_mean

N_NORMAL = 124   # in the original file, patients 1..124 are normal and 125..161 abnormal pregnancies


def load(path):
    """Whitespace-separated file with columns order, patient, time, beta-HCG, estradiol (NA = missing), and
    optionally a sixth column 'class'. Without it, the class rule of the original data file is used."""
    d = pd.read_csv(path, sep=r"\s+")
    cols = ["order", "pid", "t", "hcg", "e2"] + (["cls"] if d.shape[1] == 6 else [])
    d.columns = cols
    d["pid"] = d.pid - 1
    if "cls" not in d:
        d["cls"] = (d.pid >= N_NORMAL).astype(int)
    return d[~(d.hcg.isna() & d.e2.isna())].sort_values(["pid", "t"]).reset_index(drop=True)


def patients(d):
    """List of (pid, class, array of visits (t, hcg, e2))."""
    return [(p, int(g.cls.iloc[0]), g[["t", "hcg", "e2"]].values.astype(float)) for p, g in d.groupby("pid")]


# ----------------------------------------------------------------------------------------- fitting
def _sample(pm, draws, tune, seed, target_accept=0.95):
    return pm.sample(draws, tune=tune, chains=4, cores=2, target_accept=target_accept, random_seed=seed, progressbar=False)


def _diagnostics(idata, names, latent=()):
    import pymc as pm
    summ = pm.stats.summary(idata, var_names=list(names))
    res = {"divergences": int(idata.sample_stats["diverging"].values.sum()),
           "max_rhat": float(summ.r_hat.astype(float).max()), "min_ess_bulk": float(summ.ess_bulk.astype(float).min()),
           "min_ess_tail": float(summ.ess_tail.astype(float).min()),
           "ebfmi": [round(float(np.mean(np.diff(e) ** 2) / np.var(e)), 2) for e in idata.sample_stats["energy"].values],
           "treedepth_hits": int((idata.sample_stats["tree_depth"].values >= 10).sum())}
    if latent:
        zs = pm.stats.summary(idata, var_names=list(latent))
        res.update(latent_max_rhat=float(zs.r_hat.astype(float).max()), latent_min_ess_bulk=float(zs.ess_bulk.astype(float).min()))
    return res, summ


def _draws(idata, names):
    post = idata.posterior
    return {v: post[v].values.reshape((-1,) + post[v].values.shape[2:]) for v in names}


def fit_fixed(d, model, draws=1000, tune=1500, seed=11):
    """'het0', 'het1', 'hom' or 'jm' (fixed inflection; random effects sampled, non-centred)."""
    import pymc as pm
    import pytensor.tensor as pt
    if model in ("het0", "het1"):
        d = d[d.cls == int(model[-1])]
    pids = np.sort(d.pid.unique()); d = d.assign(ix=d.pid.map({p: i for i, p in enumerate(pids)}))
    P = len(pids); pcls = d.groupby("ix").cls.first().values
    K = 2 if model == "hom" else 1
    kidx = pcls if model == "hom" else np.zeros(P, int)
    both = d[d.hcg.notna() & d.e2.notna()]; honly = d[d.hcg.notna() & d.e2.isna()]; eonly = d[d.hcg.isna() & d.e2.notna()]
    with pm.Model():
        if model == "jm":
            A = pm.Normal("A", 4.5, 1.0); E = pm.Normal("E", 2.5, 1.0)
            infl = pm.TruncatedNormal("infl", 15, 10, lower=0); scale = pm.LogNormal("scale", np.log(7), 0.75)
            slope = pm.Normal("slope", 0, 0.05)
            nu_i = pt.stack([A, E])[None, :] * pt.ones((P, 1))
        else:
            A = pm.Normal("A", 4.5, 1.0, shape=K); E = pm.Normal("E", 2.5, 1.0, shape=K)
            infl = pm.TruncatedNormal("infl", 15, 10, lower=0, shape=K)
            scale = pm.LogNormal("scale", np.log(7), 0.75, shape=K); slope = pm.Normal("slope", 0, 0.05, shape=K)
            nu_i = pt.stack([A[kidx], E[kidx]], axis=1)
        LD, _, _ = pm.LKJCholeskyCov("LD", n=2, eta=2, sd_dist=pm.HalfNormal.dist(1.0, shape=2), compute_corr=True)
        LS, _, _ = pm.LKJCholeskyCov("LS", n=2, eta=2, sd_dist=pm.HalfNormal.dist(0.5, shape=2), compute_corr=True)
        pm.Deterministic("D", LD @ LD.T); Sig = pm.Deterministic("Sigma", LS @ LS.T)
        z = pm.Normal("z", 0, 1, shape=(P, 2))
        theta = nu_i + z @ LD.T

        def means(rows):
            ix = rows.ix.values; t = rows.t.values.astype(float)
            if model == "jm":
                v = 1 / (1 + pt.exp((infl - t) / scale)); m2 = theta[ix, 1] + slope * t
            else:
                k = kidx[ix]; v = 1 / (1 + pt.exp((infl[k] - t) / scale[k])); m2 = theta[ix, 1] + slope[k] * t
            return theta[ix, 0] * v, m2

        m1, m2 = means(both)
        pm.MvNormal("y_both", mu=pt.stack([m1, m2], axis=1), chol=LS, observed=both[["hcg", "e2"]].values)
        if len(honly):
            m1, _ = means(honly); pm.Normal("y_h", m1, pt.sqrt(Sig[0, 0]), observed=honly.hcg.values)
        if len(eonly):
            _, m2 = means(eonly); pm.Normal("y_e", m2, pt.sqrt(Sig[1, 1]), observed=eonly.e2.values)
        if model == "jm":
            g = pm.Normal("g", 0, 2.5, shape=3)
            zt = (theta - nu_i) / pt.sqrt(pt.diag(LD @ LD.T))
            pm.Bernoulli("c", logit_p=g[0] + g[1] * zt[:, 0] + g[2] * zt[:, 1], observed=pcls)
        idata = _sample(pm, draws, tune, seed)
    names = ["A", "E", "infl", "scale", "slope", "D", "Sigma"] + (["g"] if model == "jm" else [])
    return _draws(idata, names), _diagnostics(idata, names)


def _padded(d):
    rows = []
    for _, g in d.groupby("pid"):
        ent = []
        for j, (t, h, e) in enumerate(g[["t", "hcg", "e2"]].values):
            if not np.isnan(h): ent.append((h, 0, t, j))
            if not np.isnan(e): ent.append((e, 1, t, j))
        rows.append(ent)
    P = len(rows); dmax = max(len(r) for r in rows)
    Y = np.zeros((P, dmax)); MK = np.zeros((P, dmax), int); TT = np.zeros((P, dmax)); VI = -np.ones((P, dmax), int); MS = np.zeros((P, dmax))
    for i, r in enumerate(rows):
        for a, (y, m, t, j) in enumerate(r):
            Y[i, a], MK[i, a], TT[i, a], VI[i, a], MS[i, a] = y, m, t, j, 1
    SAME = ((VI[:, :, None] == VI[:, None, :]) & (MS[:, :, None] * MS[:, None, :] > 0)).astype(float)
    return P, dmax, Y, MK, TT, MS, SAME


def fit_infl(d, cls, ncomp=1, exclude=None, draws=1000, tune=1500, seed=11, target_accept=0.95):
    """Random-inflection discriminant model for one class ('infl<cls>' or, with ncomp=2, 'inflmix<cls>').
    exclude: 0-based patient id to leave out (exact leave-one-out refit)."""
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.tensor.slinalg import cholesky, solve_triangular
    d = d[(d.cls == cls) & (d.pid != (-1 if exclude is None else exclude))]
    P, dmax, Y, MK, TT, MS, SAME = _padded(d)
    H = (MK == 0) * MS; Eo = ((MK == 1) * MS).astype(float)
    MM = MS[:, :, None] * MS[:, None, :]; EYE = np.eye(dmax)[None] * (1 - MS)[:, :, None]
    with pm.Model():
        if ncomp == 1:
            A = pm.Normal("A", 4.5, 1.0); E = pm.Normal("E", 2.5, 1.0); Am, Em, w = A[None], E[None], pt.ones(1)
        else:
            Am = pm.Normal("Am", 4.0, 1.0, shape=ncomp, transform=pm.distributions.transforms.ordered, initval=np.linspace(3.0, 4.6, ncomp))
            Em = pm.Normal("Em", 2.5, 1.0, shape=ncomp); w = pm.Dirichlet("wmix", a=2.0 * np.ones(ncomp))
            pm.Deterministic("A", pt.sum(w * Am)); pm.Deterministic("E", pt.sum(w * Em))
        INFL = pm.TruncatedNormal("infl", 15, 10, lower=0)
        scale = pm.LogNormal("scale", np.log(7), 0.75); slope = pm.Normal("slope", 0, 0.05)
        LD, _, _ = pm.LKJCholeskyCov("LD", n=3, eta=2, sd_dist=pm.HalfNormal.dist(np.array([1.0, 1.0, 5.0])), compute_corr=True)
        LS, _, _ = pm.LKJCholeskyCov("LS", n=2, eta=2, sd_dist=pm.HalfNormal.dist(0.5, shape=2), compute_corr=True)
        D = pm.Deterministic("D", LD @ LD.T); Sig = pm.Deterministic("Sigma", LS @ LS.T)
        b = D[:2, 2] / D[2, 2]; Dc = D[:2, :2] - pt.outer(D[:2, 2], D[:2, 2]) / D[2, 2]
        u = pm.Normal("u", 0, 1, shape=P); cq = INFL + pt.sqrt(D[2, 2]) * u
        v = pt.sigmoid((TT - cq[:, None]) / scale) * H
        Eb = pt.as_tensor(Eo)
        Z = pt.stack([v, Eb * pt.ones_like(v)], axis=2)
        R = Sig[MK[:, :, None], MK[:, None, :]] * SAME
        V = (Z @ Dc @ Z.transpose(0, 2, 1) + R) * MM + EYE
        L = cholesky(V); logdet = pt.sum(pt.log(pt.diagonal(L, axis1=1, axis2=2)), axis=1)
        comp = []
        for k in range(ncomp):
            m_ae = pt.stack([Am[k], Em[k]])[None, :] + b[None, :] * (cq - INFL)[:, None]
            mu = m_ae[:, 0, None] * v + (m_ae[:, 1, None] + slope * TT) * Eb
            a = solve_triangular(L, ((Y - mu) * MS)[:, :, None], lower=True)[:, :, 0]
            comp.append(pt.log(w[k]) - 0.5 * pt.sum(a ** 2, axis=1) - logdet)
        pm.Potential("lik", pt.sum(pt.logsumexp(pt.stack(comp, axis=1), axis=1) if ncomp > 1 else comp[0]))
        idata = _sample(pm, draws, tune, seed, target_accept)
    names = ["A", "E", "infl", "scale", "slope", "D", "Sigma"] + (["Am", "Em", "wmix"] if ncomp > 1 else [])
    return _draws(idata, names), _diagnostics(idata, names)


def fit_jm_infl(d, draws=2000, tune=3000, seed=11):
    """Joint model with random inflection (theta_i latent, non-centred)."""
    import pymc as pm
    import pytensor.tensor as pt
    pids = np.sort(d.pid.unique()); d = d.assign(ix=d.pid.map({p: i for i, p in enumerate(pids)}))
    P = len(pids); pcls = d.groupby("ix").cls.first().values
    both = d[d.hcg.notna() & d.e2.notna()]; honly = d[d.hcg.notna() & d.e2.isna()]; eonly = d[d.hcg.isna() & d.e2.notna()]
    with pm.Model():
        A = pm.Normal("A", 4.5, 1.0); E = pm.Normal("E", 2.5, 1.0)
        INFL = pm.TruncatedNormal("infl", 15, 10, lower=0)
        scale = pm.LogNormal("scale", np.log(7), 0.75); slope = pm.Normal("slope", 0, 0.05)
        LD, _, _ = pm.LKJCholeskyCov("LD", n=3, eta=2, sd_dist=pm.HalfNormal.dist(np.array([1.0, 1.0, 5.0])), compute_corr=True)
        LS, _, _ = pm.LKJCholeskyCov("LS", n=2, eta=2, sd_dist=pm.HalfNormal.dist(0.5, shape=2), compute_corr=True)
        D = pm.Deterministic("D", LD @ LD.T); Sig = pm.Deterministic("Sigma", LS @ LS.T)
        z = pm.Normal("z", 0, 1, shape=(P, 3)); nu = pt.stack([A, E, INFL]); theta = nu[None, :] + z @ LD.T

        def means(rows):
            ix = rows.ix.values; t = rows.t.values.astype(float)
            return theta[ix, 0] * pt.sigmoid((t - theta[ix, 2]) / scale), theta[ix, 1] + slope * t

        m1, m2 = means(both)
        pm.MvNormal("y_both", mu=pt.stack([m1, m2], axis=1), chol=LS, observed=both[["hcg", "e2"]].values)
        m1, _ = means(honly); pm.Normal("y_h", m1, pt.sqrt(Sig[0, 0]), observed=honly.hcg.values)
        if len(eonly):
            _, m2 = means(eonly); pm.Normal("y_e", m2, pt.sqrt(Sig[1, 1]), observed=eonly.e2.values)
        g = pm.Normal("g", 0, 2.5, shape=4)
        zt = (theta - nu[None, :]) / pt.sqrt(pt.diag(D))[None, :]
        pm.Bernoulli("c", logit_p=g[0] + zt @ g[1:], observed=pcls)
        idata = _sample(pm, draws, tune, seed)
    names = ["A", "E", "infl", "scale", "slope", "D", "Sigma", "g"]
    return _draws(idata, names), _diagnostics(idata, names, latent=["z"])


# -------------------------------------------------------------------------------------- prediction
def design(par, vis):
    """Observed vector, mean (S,d), Z (S,d,2), residual covariance (S,d,d) for a patient's visits."""
    y, mk, tt, vi = [], [], [], []
    for j, (t, h, e) in enumerate(vis):
        if not np.isnan(h): y.append(h); mk.append(0); tt.append(t); vi.append(j)
        if not np.isnan(e): y.append(e); mk.append(1); tt.append(t); vi.append(j)
    y, mk, tt, vi = map(np.array, (y, mk, tt, vi))
    S = len(par["A"]); dd = len(y)
    v = expit((tt[None, :] - par["infl"][:, None]) / par["scale"][:, None])
    Z = np.zeros((S, dd, 2)); Z[:, mk == 0, 0] = v[:, mk == 0]; Z[:, mk == 1, 1] = 1
    off = np.where(mk[None, :] == 1, par["slope"][:, None] * tt[None, :], 0.0)
    mean = np.einsum("sdq,sq->sd", Z, np.stack([par["A"], par["E"]], 1)) + off
    R = par["Sigma"][:, mk[:, None], mk[None, :]] * (vi[:, None] == vi[None, :])[None]
    return y, mean, Z, R


def mvn_logpdf(y, mean, cov):
    L = np.linalg.cholesky(cov); a = np.linalg.solve(L, (y[None, :] - mean)[..., None])[..., 0]
    return -0.5 * (a ** 2).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1) - 0.5 * len(y) * np.log(2 * np.pi)


def class_params(F, model, k):
    """Posterior draws for class k: model 'het' (fits het0/het1) or 'hom' (fit hom)."""
    keys = ("A", "E", "infl", "scale", "slope")
    if model == "het":
        f = F[f"het{k}"]; return {v: (f[v][:, 0] if f[v].ndim == 2 else f[v]) for v in keys} | {"D": f["D"], "Sigma": f["Sigma"]}
    f = F["hom"]; return {v: f[v][:, k] for v in keys} | {"D": f["D"], "Sigma": f["Sigma"]}


def logp_fixed(par, vis):
    y, mean, Z, R = design(par, vis)
    return mvn_logpdf(y, mean, Z @ par["D"] @ Z.transpose(0, 2, 1) + R)


def logp_infl(par, vis, G=281):
    """log p(y_obs | class): grid over the inflection (281 points on [-7, 7] sd), (a, e) | c analytic."""
    if "wmix" in par:
        out = []
        for c in range(par["Am"].shape[1]):
            pc = {k: v for k, v in par.items() if k != "wmix"}; pc["A"] = par["Am"][:, c]; pc["E"] = par["Em"][:, c]
            out.append(np.log(par["wmix"][:, c]) + logp_infl(pc, vis, G))
        return np.logaddexp.reduce(np.stack(out, 1), axis=1)
    ug = np.linspace(-7, 7, G); lw = np.log(np.exp(-0.5 * ug ** 2) / np.sqrt(2 * np.pi) * (ug[1] - ug[0]))
    D = par["D"]; b = D[:, :2, 2] / D[:, 2, 2][:, None]
    Dc = D[:, :2, :2] - np.einsum("si,sj->sij", D[:, :2, 2], D[:, :2, 2]) / D[:, 2, 2][:, None, None]
    out = []
    for u, l in zip(ug, lw):
        dc = np.sqrt(D[:, 2, 2]) * u
        pc = dict(par); pc["infl"] = par["infl"] + dc; pc["A"] = par["A"] + b[:, 0] * dc; pc["E"] = par["E"] + b[:, 1] * dc; pc["D"] = Dc
        out.append(l + logp_fixed(pc, vis))
    return np.logaddexp.reduce(np.stack(out, 1), axis=1)


def jm_terms(par, vis):
    """Joint model with fixed inflection: (log p(y), P(C=1 | y)) per draw (Gauss-Hermite quadrature)."""
    y, mean, Z, R = design(par, vis)
    G = par["D"]; V = Z @ G @ Z.transpose(0, 2, 1) + R
    K = G @ Z.transpose(0, 2, 1) @ np.linalg.inv(V)
    m = np.einsum("sqd,sd->sq", K, y[None, :] - mean); Pc = G - K @ Z @ G
    a = par["g"][:, 1:] / np.sqrt(np.stack([G[:, 0, 0], G[:, 1, 1]], 1))
    mu = par["g"][:, 0] + (a * m).sum(1); s2 = np.einsum("sq,sqr,sr->s", a, Pc, a)
    return mvn_logpdf(y, mean, V), expit_normal_mean(mu, np.sqrt(s2))


def jm_infl_terms(par, vis, G=281):
    """Joint model with random inflection: grid over the inflection, (a, e) | y, c exact."""
    ug = np.linspace(-7, 7, G); lw = np.log(np.exp(-0.5 * ug ** 2) / np.sqrt(2 * np.pi) * (ug[1] - ug[0]))
    D = par["D"]; sd = np.sqrt(np.stack([D[:, 0, 0], D[:, 1, 1], D[:, 2, 2]], 1))
    b = D[:, :2, 2] / D[:, 2, 2][:, None]
    Dc = D[:, :2, :2] - np.einsum("si,sj->sij", D[:, :2, 2], D[:, :2, 2]) / D[:, 2, 2][:, None, None]
    a_ = par["g"][:, 1:] / sd; LP, PC = [], []
    for u, l in zip(ug, lw):
        dc = np.sqrt(D[:, 2, 2]) * u
        pc = dict(par); pc["infl"] = par["infl"] + dc; pc["A"] = par["A"] + b[:, 0] * dc; pc["E"] = par["E"] + b[:, 1] * dc; pc["D"] = Dc
        y, mean, Z, R = design(pc, vis)
        V = Z @ Dc @ Z.transpose(0, 2, 1) + R
        LP.append(l + mvn_logpdf(y, mean, V))
        K = Dc @ Z.transpose(0, 2, 1) @ np.linalg.inv(V)
        m_ae = np.stack([pc["A"], pc["E"]], 1) + np.einsum("sqd,sd->sq", K, y[None, :] - mean); Pae = Dc - K @ Z @ Dc
        mu = par["g"][:, 0] + a_[:, 0] * (m_ae[:, 0] - par["A"]) + a_[:, 1] * (m_ae[:, 1] - par["E"]) + a_[:, 2] * dc
        PC.append(expit_normal_mean(mu, np.sqrt(np.einsum("sq,sqr,sr->s", a_[:, :2], Pae, a_[:, :2]))))
    LP = np.stack(LP, 1); PC = np.stack(PC, 1); lpy = np.logaddexp.reduce(LP, axis=1)
    return lpy, np.exp(np.logaddexp.reduce(LP + np.log(PC), axis=1) - lpy)


def loo_discriminant(pats, params, logp):
    """PSIS leave-one-out class probabilities for every prefix of visits.
    params: [draws class 0, draws class 1]. Returns (probabilities: list per patient, LPML, khat, logp cache)."""
    C = np.array([c for _, c, _ in pats]); N = len(pats); nk = np.array([(C == 0).sum(), (C == 1).sum()])
    lp = [[[logp(params[k], vis[:j]) for j in range(1, len(vis) + 1)] for (_, _, vis) in pats] for k in (0, 1)]
    out, kh, lpml = [], [], 0.0
    for i in range(N):
        own = lp[C[i]][i][-1]
        lpml += -(logsumexp(-own) - np.log(len(own)))
        lw, k_ = psis(-own); kh.append(k_); n_ = nk - np.eye(2, dtype=int)[C[i]]; pr = []
        for j in range(len(lp[0][i])):
            ll = [np.log(n_[k] / (N - 1)) + (logsumexp(lp[k][i][j] + lw) if k == C[i] else logsumexp(lp[k][i][j]) - np.log(len(lp[k][i][j])))
                  for k in (0, 1)]
            pr.append(expit(ll[1] - ll[0]))
        out.append(pr)
    return out, lpml, np.array(kh), lp


def exact_loo_patient(pats, i, params_without_i, params, logp):
    """Exact leave-one-out probabilities for patient i, with its class refitted without it."""
    C = np.array([c for _, c, _ in pats]); N = len(pats); nk = np.array([(C == 0).sum(), (C == 1).sum()])
    c = C[i]; vis = pats[i][2]; par = [params_without_i if k == c else params[k] for k in (0, 1)]
    n_ = nk - np.eye(2, dtype=int)[c]; pr = []
    for j in range(1, len(vis) + 1):
        ll = [np.log(n_[k] / (N - 1)) + logsumexp(logp(par[k], vis[:j])) - np.log(len(par[k]["A"])) for k in (0, 1)]
        pr.append(expit(ll[1] - ll[0]))
    return pr


def loo_joint(pats, par, terms):
    """PSIS leave-one-out posterior predictive probabilities for a joint model; terms(par, visits) -> (log p(y | psi), P(C=1 | y, psi)) per draw."""
    out, kh = [], []
    for _, c, vis in pats:
        tt = [terms(par, vis[:j]) for j in range(1, len(vis) + 1)]
        lpf, p1f = tt[-1]
        lw, k_ = psis(-(lpf + np.log(np.where(c == 1, p1f, 1 - p1f)))); kh.append(k_)
        # posterior predictive: draws weighted by p(y prefix | psi) as well as by the leave-one-out weights
        out.append([np.exp(logsumexp(lw + lp + np.log(p)) - logsumexp(lw + lp)) for lp, p in tt])
    return out, np.array(kh)


def thin(F, every):
    return {v: a[::every] for v, a in F.items()}


def exact_loo_joint(pats, i, par_without_i, terms):
    """Exact leave-one-out probabilities for patient i under a joint model refitted without it."""
    vis = pats[i][2]
    out = []
    for j in range(1, len(vis) + 1):
        lp, p = terms(par_without_i, vis[:j])
        out.append(float(np.exp(logsumexp(lp + np.log(p)) - logsumexp(lp))))
    return out
