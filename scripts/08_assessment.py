"""Assessment of the applications (Sections 6.1 and 6.2): AUC, Brier score, calibration slope and intercept by
landmark with 95% bootstrap intervals over subjects, paired differences between models, and calibration curves
(Figure 1). Needs the outputs of 03_gtn.py, 03b_gtn_two_stage.py, 04_pregnancy.py and 07_nested_cv_tempering.py (nested cross-validation).
  python scripts/08_assessment.py [--boot 2000] [--synthetic | --suffix _synthetic_quick]
Outputs results/assessment_gtn.csv, results/assessment_pregnancy.csv, results/paired_differences.csv,
results/figure1_calibration.pdf/png."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from _common import RESULTS
from dyndisc.metrics import logit

ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--synthetic", action="store_true", help="use the outputs obtained with the synthetic data")
ap.add_argument("--suffix", default=None, help="suffix of the result folders, e.g. _synthetic_quick")
a = ap.parse_args()
SFX = a.suffix if a.suffix is not None else ("_synthetic" if a.synthetic else "")
rng = np.random.default_rng(2026)


def newton(x, y, slope_fixed=False, it=30):
    """Logistic regression of y on x: (intercept, slope); with slope_fixed, intercept with x as offset."""
    b = np.zeros(1 if slope_fixed else 2)
    for _ in range(it):
        eta = (b[0] + x) if slope_fixed else (b[0] + b[1] * x)
        p = expit(eta); w = p * (1 - p) + 1e-12
        X = np.ones((len(x), 1)) if slope_fixed else np.c_[np.ones(len(x)), x]
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-10 * np.eye(len(b)), X.T @ (y - p)); b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def metrics(p, y):
    z = logit(p)
    return {"auc": roc_auc_score(y, p), "brier": np.mean((p - y) ** 2), "slope": newton(z, y)[1], "intercept": newton(z, y, True)[0]}


def assess(frame, models, landmarks, label):
    """frame: rows (subject, landmark, y, model columns). Returns metrics with intervals and paired differences."""
    out, diffs = [], []
    subs = np.unique(frame.subject)
    boot_idx = [rng.choice(subs, len(subs), replace=True) for _ in range(a.boot)]
    for h in landmarks:
        f = frame[frame.landmark == h]
        rows_of = {s: np.where(f.subject.values == s)[0] for s in np.unique(f.subject)}
        y = f.y.values; P = {m: f[m].values for m in models if m in f}
        est = {m: metrics(P[m], y) for m in P}
        bs = {m: [] for m in P}
        for bi in boot_idx:
            idx = np.concatenate([rows_of[s] for s in bi if s in rows_of])
            if len(np.unique(y[idx])) < 2:
                continue
            for m in P:
                bs[m].append(metrics(P[m][idx], y[idx]))
        for m in P:
            r = {"application": label, "landmark": h, "n": len(y), "model": m}
            for k in ("auc", "brier", "slope", "intercept"):
                v = np.array([b[k] for b in bs[m]]); r[k] = est[m][k]; r[k + "_lo"], r[k + "_hi"] = np.percentile(v, [2.5, 97.5])
            out.append(r)
        for m1, m2 in [(x, yy) for x, yy in PAIRS if x in P and yy in P]:
            for k in ("auc", "brier"):
                v = np.array([b1[k] - b2[k] for b1, b2 in zip(bs[m1], bs[m2])])
                diffs.append({"application": label, "landmark": h, "comparison": f"{m1} - {m2}", "metric": k,
                              "estimate": est[m1][k] - est[m2][k], "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)})
    return pd.DataFrame(out), pd.DataFrame(diffs)


# ------------------------------------------------------------------------------------------- GTN
L = np.load(RESULTS / f"gtn{SFX}" / "loo_predictions.npz"); E = np.load(RESULTS / f"gtn{SFX}" / "external_cv.npz")
g = pd.DataFrame({"subject": L["subject"], "landmark": L["week"], "y": L["y"], "DF": L["DF"], "JM": L["JM"]})
ge = pd.DataFrame({"subject": E["subject"], "landmark": E["landmark"], **{f"{m} (outer)" if m == "DF" else m: E[m] for m in ("DF", "DF-T", "DF-Tw", "DF-Th")}})
g = g.merge(ge, on=["subject", "landmark"])
PAIRS = [("DF", "JM"), ("DF-T", "DF (outer)"), ("DF-Tw", "DF (outer)"), ("DF-Th", "DF (outer)"), ("DF-Tw", "JM"), ("DF-Th", "JM"), ("DF-T", "JM")]
gm = ["DF", "JM", "DF (outer)", "DF-T", "DF-Tw", "DF-Th"]
ga, gd = assess(g, gm, range(2, 8), "GTN")

# ------------------------------------------------------------------------------------- pregnancy
pred, ids = np.load(RESULTS / f"pregnancy{SFX}" / "loo_predictions.npy", allow_pickle=True)
Cp = np.array([c for _, c in ids]); rows = []
for i in range(len(Cp)):
    nv = len(pred["DF, random inflection"][i])
    for lab, j in (("all", nv), (1, 1), (2, 2), (3, 3)):
        if j <= nv:
            rows.append({"subject": i, "landmark": str(lab), "position": j - 1, "y": Cp[i], **{m: pred[m][i][j - 1] for m in pred}})
p = pd.DataFrame(rows)
E = np.load(RESULTS / f"pregnancy{SFX}" / "external_cv.npz")
pe = pd.DataFrame({"subject": E["subject"], "position": E["position"], **{f"{m} (outer)" if m == "DF" else m: E[m] for m in ("DF", "DF-T", "DF-Tw", "DF-Th")}})
p = p.merge(pe, on=["subject", "position"])
PAIRS = [("DF, random inflection", "JM, random inflection"), ("DF, fixed inflection", "JM, fixed inflection"),
         ("DF, random inflection", "DF, fixed inflection"), ("DF-T", "DF (outer)"), ("DF-Tw", "DF (outer)"), ("DF-Th", "DF (outer)"),
         ("DF-T", "JM, random inflection"), ("DF-Th", "JM, random inflection")]
pm_ = list(pred.keys()) + ["DF (outer)", "DF-T", "DF-Tw", "DF-Th"]
pa, pdf_ = assess(p, pm_, ["all", "1", "2", "3"], "pregnancy")

ga.to_csv(RESULTS / f"assessment_gtn{SFX}.csv", index=False); pa.to_csv(RESULTS / f"assessment_pregnancy{SFX}.csv", index=False)
pd.concat([gd, pdf_]).to_csv(RESULTS / f"paired_differences{SFX}.csv", index=False)
pd.set_option("display.width", 250)
print(ga.round(3).to_string(index=False)); print(pa.round(3).to_string(index=False)); print(pd.concat([gd, pdf_]).round(3).to_string(index=False))


# --------------------------------------------------------------------------- calibration curves
def curve(pv, yv, grid):
    """Logistic calibration curve: regression of y on a natural cubic spline of the predicted logit (3 df)."""
    from patsy import dmatrix
    z = logit(pv); kn = np.percentile(z, [33.3, 66.7]); lo, hi = z.min(), z.max()
    B = lambda v: np.asarray(dmatrix("cr(v, knots=kn, lower_bound=lo, upper_bound=hi) - 1", {"v": np.clip(v, lo, hi), "kn": kn, "lo": lo, "hi": hi}))
    X = B(z); Xg = B(logit(grid))
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e4, max_iter=2000, fit_intercept=False).fit(X, yv)
    return m.predict_proba(Xg)[:, 1]


INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
COL = {"DF": "#2a78d6", "Tempered DF": "#1baf7a", "JM": "#eb6834"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
panels = [("GTN, week 2", g[g.landmark == 2], {"DF": "DF", "Tempered DF": "DF-Tw", "JM": "JM"}),
          ("GTN, week 6", g[g.landmark == 6], {"DF": "DF", "Tempered DF": "DF-Tw", "JM": "JM"}),
          ("Pregnancy, all visits", p[p.landmark == "all"], {"DF": "DF, random inflection", "Tempered DF": "DF-T", "JM": "JM, random inflection"})]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
for axi, (title, f, mm) in zip(ax, panels):
    axi.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    subs = np.unique(f.subject)
    for lab, col in mm.items():
        pv, yv = f[col].values, f.y.values
        grid = np.linspace(max(0.01, np.percentile(pv, 5)), min(0.99, np.percentile(pv, 95)), 60)
        est = curve(pv, yv, grid); bs = []
        for _ in range(300):
            bi = rng.choice(len(subs), len(subs), replace=True)
            idx = np.concatenate([np.where(f.subject.values == subs[b])[0] for b in bi])
            try:
                bs.append(curve(pv[idx], yv[idx], grid))
            except Exception:
                pass
        lo_, hi_ = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        axi.fill_between(grid, lo_, hi_, color=COL[lab], alpha=0.15, lw=0)
        axi.plot(grid, est, color=COL[lab], lw=1.8, label=lab)
    axi.set_xlim(0, 1); axi.set_ylim(0, 1); axi.set_title(title, fontsize=9, loc="left")
    axi.set_xlabel("Predicted probability"); axi.set_aspect("equal")
ax[0].set_ylabel("Observed proportion"); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(RESULTS / f"figure1_calibration{SFX}.pdf", bbox_inches="tight"); fig.savefig(RESULTS / f"figure1_calibration{SFX}.png", dpi=300, bbox_inches="tight")
print("written figure1_calibration")
