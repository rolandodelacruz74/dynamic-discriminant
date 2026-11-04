"""Joint-model predictions: average of P(C=1 | y, psi) over posterior draws (used in the paper) versus the
exact posterior predictive sum_s w_s P(C=1 | y, psi_s), w_s proportional to p(y | psi_s), both with the
leave-one-out (PSIS) weights. Reports the largest absolute difference and AUC, Brier and calibration slope.
Only summaries are written (results/jm_predictive_check.json)."""
import json
import numpy as np
from scipy.special import logsumexp
from _common import ROOT, RESULTS
from dyndisc import gtn, pregnancy as pg
from dyndisc.psis import psis
from dyndisc.metrics import summary

out = {}


def compare(name, rows):
    """rows: list of (y, p_avg, p_exact) per subject and landmark."""
    y = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
    sa, sb = summary(a, y), summary(b, y)
    out[name] = {"max_abs_diff": float(np.max(np.abs(a - b))), "mean_abs_diff": float(np.mean(np.abs(a - b))),
                 **{f"{k}_avg": float(sa[k]) for k in ("auc", "brier", "slope")}, **{f"{k}_exact": float(sb[k]) for k in ("auc", "brier", "slope")}}
    print(name, out[name], flush=True)


# GTN
d = gtn.load(ROOT / "data" / "gtn" / "hCG_GTN.csv")
J = {v: x[::2] for v, x in dict(np.load(RESULTS / "gtn" / "draws_jm.npz")).items()}
sd = np.sqrt(np.stack([J["D"][:, 0, 0], J["D"][:, 1, 1]], 1)); aa = J["g"][:, 1:] / sd
rows = []
for i, g in d.groupby("id"):
    c = g.C.iloc[0]; t = g.t.values.astype(float); y = g.loghcg.values
    terms = {}
    for s in gtn.WEEKS:
        if not (t <= s).any():
            continue
        tt, yy = t[t <= s], y[t <= s]
        mean, V, Z = gtn._marg(J, tt); K = J["D"] @ Z.T[None] @ np.linalg.inv(V)
        m = np.einsum("sqd,sd->sq", K, yy[None] - mean); Pc = J["D"] - K @ Z[None] @ J["D"]
        mu = J["g"][:, 0] + (aa * m).sum(1); s2 = np.einsum("sq,sqr,sr->s", aa, Pc, aa)
        from dyndisc.metrics import expit_normal_mean
        terms[s] = (gtn._mvn(yy, mean, V), expit_normal_mean(mu, np.sqrt(s2)))
    lpf, p1f = terms[max(terms)]
    lw, _ = psis(-(lpf + np.log(np.where(c == 1, p1f, 1 - p1f))))
    for s, (lp, p) in terms.items():
        rows.append((c, np.exp(logsumexp(lw + np.log(p))), np.exp(logsumexp(lw + lp + np.log(p)) - logsumexp(lw + lp))))
compare("GTN JM", rows)

# pregnancy
dp = pg.load(ROOT / "data" / "pregnancy" / "datos.txt"); pats = pg.patients(dp)
for name, f, terms, th in (("pregnancy JM fixed", "jm", pg.jm_terms, 1), ("pregnancy JM random", "jm_infl", pg.jm_infl_terms, 4)):
    F = pg.thin(dict(np.load(RESULTS / "pregnancy" / f"draws_{f}.npz")), th); rows = []
    for _, c, vis in pats:
        tt = [terms(F, vis[:j]) for j in range(1, len(vis) + 1)]
        lpf, p1f = tt[-1]; lw, _ = psis(-(lpf + np.log(np.where(c == 1, p1f, 1 - p1f))))
        for lp, p in tt:
            rows.append((c, np.exp(logsumexp(lw + np.log(p))), np.exp(logsumexp(lw + lp + np.log(p)) - logsumexp(lw + lp))))
    compare(name, rows)
json.dump(out, open(RESULTS / "jm_predictive_check.json", "w"), indent=1)
