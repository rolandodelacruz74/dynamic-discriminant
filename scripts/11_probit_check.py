"""Calibration slopes of the joint-model leave-one-out predictions with Gauss-Hermite quadrature and with the
probit approximation E{expit(mu + tau Z)} ~ expit{mu (1 + pi tau^2 / 8)^(-1/2)}, by landmark (summaries only)."""
import json
import numpy as np
from scipy.special import expit
from _common import ROOT, RESULTS
from dyndisc import gtn, pregnancy as pg, metrics
from dyndisc.metrics import calibration_slope

GHf = metrics.expit_normal_mean
probit = lambda mu, sd: expit(np.asarray(mu) / np.sqrt(1 + np.pi * np.asarray(sd) ** 2 / 8))
res = {}
for lab, f in (("quadrature", GHf), ("probit", probit)):
    metrics.expit_normal_mean = f; gtn.expit_normal_mean = f; pg.expit_normal_mean = f
    d = gtn.load(ROOT / "data" / "gtn" / "hCG_GTN.csv")
    F = {m: dict(np.load(RESULTS / "gtn" / f"draws_{m}.npz")) for m in ("df0", "df1", "jm")}
    pred, _, C = gtn.loo_predictions(d, F)
    res[("GTN", lab)] = {s: calibration_slope(np.array([p for (i, w), p in pred["JM"].items() if w == s]), np.array([C[i] for (i, w) in pred["JM"] if w == s])) for s in gtn.WEEKS}
    dp = pg.load(ROOT / "data" / "pregnancy" / "datos.txt"); pats = pg.patients(dp); Cp = np.array([c for _, c, _ in pats])
    for name, fm, terms, th in (("JM fixed", "jm", pg.jm_terms, 1), ("JM random", "jm_infl", pg.jm_infl_terms, 4)):
        pr, _ = pg.loo_joint(pats, pg.thin(dict(np.load(RESULTS / "pregnancy" / f"draws_{fm}.npz")), th), terms)
        out = {}
        for h, j in (("all", None), ("1", 1), ("2", 2), ("3", 3)):
            idx = range(len(pats)) if j is None else [i for i in range(len(pats)) if len(pats[i][2]) >= j]
            out[h] = calibration_slope(np.array([pr[i][-1] if j is None else pr[i][j - 1] for i in idx]), Cp[list(idx)])
        res[(name, lab)] = out
summ = {k: max(abs(res[(k, "probit")][h] - res[(k, "quadrature")][h]) for h in res[(k, "quadrature")]) for k in ("GTN", "JM fixed", "JM random")}
print(res); print(summ)
json.dump({"max_abs_slope_change": summ}, open(RESULTS / "probit_check.json", "w"), indent=1)
