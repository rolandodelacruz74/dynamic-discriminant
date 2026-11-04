"""Two-stage approach on the GTN data (Section 6.1): calibration slope by landmark week.

Stage 1: a single linear mixed model (random intercept and slope, REML) is fitted to the training women;
Stage 2: a logistic regression of GTN on the empirical Bayes estimates computed from all measurements of
each training woman. At landmark week s a test woman's empirical Bayes estimates use her measurements up
to week s, as in Dandis et al. (2020). Five-fold cross-validation repeated four times; the calibration slope
at each week is computed from the pooled out-of-fold predictions.

  python scripts/03b_gtn_two_stage.py            uses data/gtn/hCG_GTN.csv if present, otherwise synthetic data
Output: results/gtn/two_stage.csv (or results/gtn_synthetic/two_stage.csv)."""
import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from _common import ROOT, outdir
from dyndisc import gtn
from dyndisc.metrics import summary

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None); a = ap.parse_args()
real = ROOT / "data" / "gtn" / "hCG_GTN.csv"
path = a.data or (real if real.exists() else ROOT / "data" / "gtn" / "gtn_synthetic.csv")
out = outdir("gtn" if str(path).endswith("hCG_GTN.csv") else "gtn_synthetic")
d = gtn.load(path)
ids = np.sort(d.id.unique()); C = d.groupby("id").C.first().loc[ids].values
groups = {i: g for i, g in d.groupby("id")}


def fit_lmm(df):
    m = smf.mixedlm("loghcg ~ t", df, groups=df["id"], re_formula="~t").fit(reml=True, method="lbfgs")
    return m.fe_params.values, np.asarray(m.cov_re), m.scale


def eb(par, t, y):
    beta, D, s2 = par; X = np.c_[np.ones(len(t)), t]
    return D @ X.T @ np.linalg.solve(X @ D @ X.T + s2 * np.eye(len(t)), y - X @ beta)


pred = {s: ([], []) for s in gtn.WEEKS}
for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=2).split(ids, C):
    par = fit_lmm(d[d.id.isin(ids[tr])])
    lr = LogisticRegression(C=1e6).fit(np.array([eb(par, groups[i].t.values, groups[i].loghcg.values) for i in ids[tr]]), C[tr])
    for j in te:
        g = groups[ids[j]]
        for s in gtn.WEEKS:
            h = g[g.t <= s]
            if len(h):
                pred[s][0].append(lr.predict_proba(eb(par, h.t.values, h.loghcg.values)[None])[0, 1]); pred[s][1].append(C[j])
from scipy.special import expit
from dyndisc.metrics import logit


def citl(p, y):
    off = logit(p); b = 0.0
    for _ in range(50):
        q = expit(b + off); b += np.sum(y - q) / np.sum(q * (1 - q))
    return b


rows = [{"week": s, **summary(np.array(pred[s][0]), np.array(pred[s][1])), "intercept": citl(np.array(pred[s][0]), np.array(pred[s][1]))}
        for s in gtn.WEEKS]
tab = pd.DataFrame(rows)[["week", "auc", "brier", "slope", "intercept"]]
tab.to_csv(out / "two_stage.csv", index=False)
print(tab.round(3).to_string(index=False))
