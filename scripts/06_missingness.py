"""Class-dependent missingness (Section 3.5 and Section 6): does using the missingness pattern improve prediction?

By the remark on missing measurements, a classifier that uses the pattern R adds to the discriminant log-odds
    log Pr(R | y_obs, C = 1) - log Pr(R | y_obs, C = 0),
and the expected gain in log score is I(C; R | Y_obs). For each application, Pr(R | y_obs, C) is modelled by a
logistic regression, fitted with ten-fold cross-fitting across subjects (a subject's own pattern and label
never enter its own correction), and added to the leave-one-out log-odds of the discriminant model.
  GTN:       R_ij = 1 if week j is missing; logit Pr(R_ij = 1) = C * week + last observed log hCG before week j
             + indicator of a previous observation (sequential model over weeks 2..s at landmark s).
  Pregnancy: R_ij = 1 if estradiol is missing at a visit with beta-HCG observed;
             logit Pr(R_ij = 1) = C + log beta-HCG + day.
The estimate of I(C; R | Y_obs) is the mean reduction in log loss.
Needs the leave-one-out predictions written by 03_gtn.py and 04_pregnancy.py (original data).
Output: results/missingness.json"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from _common import ROOT, RESULTS, write_json
from dyndisc import gtn, pregnancy as pg
from dyndisc.metrics import logit

FOLDS = StratifiedKFold(10, shuffle=True, random_state=1)


def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(p0, delta, y):
    p1 = expit(logit(p0) + delta)
    return {"n": int(len(y)), "logloss_without_R": logloss(p0, y), "logloss_with_R": logloss(p1, y),
            "I_hat": logloss(p0, y) - logloss(p1, y), "auc_without_R": roc_auc_score(y, p0), "auc_with_R": roc_auc_score(y, p1)}


out = {}
# ------------------------------------------------------------------ GTN
raw = pd.read_csv(ROOT / "data" / "gtn" / "hCG_GTN.csv", na_values=["999"]).sort_values(["id", "weeknr"])
W = raw.pivot(index="id", columns="weeknr", values="loghcg"); Cg = raw.groupby("id").PTD.first().loc[W.index].values.astype(int)
miss = np.isnan(W.values)
out["gtn_missing_rate"] = {"no_GTN": float(miss[Cg == 0].mean()), "GTN": float(miss[Cg == 1].mean()),
                           "GTN_week7": float(miss[Cg == 1, -1].mean()), "women": int(len(W))}
W = W[~miss.all(1)]; Y = W.values; Cg = Cg[~miss.all(1)]          # the 439 women of the analysis, in id order
m2 = np.isnan(Y)
out["gtn_missing_rate_analysis"] = {"no_GTN": float(m2[Cg == 0].mean()), "GTN": float(m2[Cg == 1].mean()), "women": int(len(Y))}
rows = []
for i, yi in enumerate(Y):
    last = np.nan
    for j, wk in enumerate(gtn.WEEKS):
        rows.append(dict(i=i, C=Cg[i], week=wk, R=int(np.isnan(yi[j])), prev=0.0 if np.isnan(last) else last, has_prev=int(not np.isnan(last))))
        if not np.isnan(yi[j]):
            last = yi[j]
V = pd.DataFrame(rows)
gee = smf.gee("R ~ C * week + prev + has_prev", "i", V, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
out["gtn_gee"] = {"coef": gee.params.to_dict(), "p": gee.pvalues.to_dict(),
                  "wald_p_class_terms": float(gee.wald_test("C = 0, C:week = 0", scalar=True).pvalue)}
Xcols = lambda df, c: np.c_[np.full(len(df), c), df.week, c * df.week, df.prev, df.has_prev]
corr = np.zeros((len(Y), len(gtn.WEEKS)))      # cumulative log Pr(R|.,1) - log Pr(R|.,0) up to each week
for tr, te in FOLDS.split(np.zeros(len(Y)), Cg):
    a = V.i.isin(tr); m = LogisticRegression(C=1e4, max_iter=5000).fit(Xcols(V[a], V[a].C.values)[:, :], V[a].R)
    for i in te:
        vi = V[V.i == i]; lp = [m.predict_log_proba(Xcols(vi, c)) for c in (0, 1)]
        ll = [np.where(vi.R == 1, l[:, 1], l[:, 0]) for l in lp]
        corr[i] = np.cumsum(ll[1] - ll[0])
L = np.load(RESULTS / "gtn" / "loo_predictions.npz")
delta = corr[L["subject"], L["week"] - 2]
out["gtn"] = {"all_landmarks": compare(L["DF"], delta, L["y"])}
for s in gtn.WEEKS:
    c = L["week"] == s; out["gtn"][f"week_{s}"] = compare(L["DF"][c], delta[c], L["y"][c])

# ------------------------------------------------------------------ pregnancy
d = pg.load(ROOT / "data" / "pregnancy" / "datos.txt"); pats = pg.patients(d); Cp = np.array([c for _, c, _ in pats])
v = d[d.hcg.notna()].copy(); v["R"] = v.e2.isna().astype(int); v["C"] = v.cls
out["pregnancy_missing_rate"] = {"normal": float(v.R[v.C == 0].mean()), "abnormal": float(v.R[v.C == 1].mean()),
                                 "visits_with_hcg_missing": int(d.hcg.isna().sum())}
gee = smf.gee("R ~ C + hcg + t", "pid", v, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
out["pregnancy_gee"] = {"odds_ratio_class": float(np.exp(gee.params["C"])), "p": float(gee.pvalues["C"])}
pid_index = {p: i for i, (p, _, _) in enumerate(pats)}; v["i"] = v.pid.map(pid_index)
corr = [None] * len(pats)
for tr, te in FOLDS.split(np.zeros(len(pats)), Cp):
    a = v.i.isin(tr); m = LogisticRegression(C=1e4, max_iter=5000).fit(np.c_[v[a].C, v[a].hcg, v[a].t], v[a].R)
    for i in te:
        p, _, vis = pats[i]; dd = np.zeros(len(vis))
        for j, (t, h, e) in enumerate(vis):
            if not np.isnan(h):
                lp = [m.predict_log_proba(np.array([[c, h, t]]))[0] for c in (0, 1)]
                r = int(np.isnan(e)); dd[j] = lp[1][r] - lp[0][r]
        corr[i] = np.cumsum(dd)
P = np.load(RESULTS / "pregnancy" / "loo_predictions.npy", allow_pickle=True)[0]["DF, random inflection"]
out["pregnancy"] = {}
for lab, j in (("all_visits", None), ("visit_1", 1), ("visit_2", 2), ("visit_3", 3)):
    idx = [i for i in range(len(pats)) if j is None or len(pats[i][2]) >= j]
    k = (lambda i: -1) if j is None else (lambda i: j - 1)
    out["pregnancy"][lab] = compare(np.array([P[i][k(i)] for i in idx]), np.array([corr[i][k(i)] for i in idx]), Cp[idx])
write_json(out, RESULTS / "missingness.json")
for app in ("gtn", "pregnancy"):
    for k, r in out[app].items():
        print(f"{app:9s} {k:14s} n={r['n']:4d}  I_hat={r['I_hat']:+.4f}  AUC {r['auc_without_R']:.3f} -> {r['auc_with_R']:.3f}")
print({k: out[k] for k in ("gtn_missing_rate", "pregnancy_missing_rate", "pregnancy_gee")}, out["gtn_gee"]["wald_p_class_terms"])
