"""Nested cross-validation of the tempered discriminant model (Sections 4.3, 6.1 and 6.2).

For each of five outer folds of subjects (stratified by class):
  1. the discriminant model is refitted to the subjects of the other four folds (training set);
  2. leave-one-out logits of the training subjects are computed from that fit by Pareto-smoothed importance
     sampling, and the tempering coefficients (single, landmark-specific, shrunk) are estimated from them;
  3. the refitted model and the estimated coefficients are applied to the subjects of the held-out fold.
No quantity used for a held-out subject depends on that subject's data or label. Training subjects whose Pareto
shape estimate exceeds 0.7 are refitted exactly (results/<app>/external_cv_fold<f>_refits.json).
  python scripts/07_nested_cv_tempering.py gtn|pregnancy [--quick]
Outputs results/<app>/external_cv.npz and external_cv_coefficients.json; folds already computed are reused."""
import argparse
import json
import numpy as np
from scipy.special import expit, logsumexp
from sklearn.model_selection import StratifiedKFold
from _common import ROOT, outdir, write_json
from dyndisc import gtn, pregnancy as pg
from dyndisc.psis import psis
from dyndisc.metrics import logit
from dyndisc.tempering import fit_tempering

ap = argparse.ArgumentParser(); ap.add_argument("app", choices=["gtn", "pregnancy"]); ap.add_argument("--quick", action="store_true")
ap.add_argument("--draws", type=int, default=None); ap.add_argument("--tune", type=int, default=None); ap.add_argument("--target-accept", type=float, default=0.95)
ap.add_argument("--suffix", default="")
a = ap.parse_args()
dr, tu = (300, 500) if a.quick else (a.draws or 1000, a.tune or 1500)
FOLDS = 5


def gtn_logp(F, t, y):
    mean, V, _ = gtn._marg(F, t); return gtn._mvn(y, mean, V)


if a.app == "gtn":
    real = ROOT / "data" / "gtn" / "hCG_GTN.csv"
    path = real if real.exists() else ROOT / "data" / "gtn" / "gtn_synthetic.csv"
    out = outdir(("gtn" if real.exists() else "gtn_synthetic") + ("_quick" if a.quick else "") + a.suffix)
    d = gtn.load(path)
    subj = [(i, int(g.C.iloc[0]), g.t.values.astype(float), g.loghcg.values) for i, g in d.groupby("id")]
    # landmark-specific prefixes: (week, t, y)
    pref = [[(s, t[t <= s], y[t <= s]) for s in gtn.WEEKS if (t <= s).any()] for _, _, t, y in subj]
else:
    real = ROOT / "data" / "pregnancy" / "datos.txt"
    path = real if real.exists() else ROOT / "data" / "pregnancy" / "pregnancy_synthetic.txt"
    out = outdir(("pregnancy" if real.exists() else "pregnancy_synthetic") + ("_quick" if a.quick else "") + a.suffix)
    d = pg.load(path); pats = pg.patients(d)
    subj = [(p, c, None, vis) for p, c, vis in pats]
    pref = [[(min(j, 3), vis[:j]) for j in range(1, len(vis) + 1)] for _, _, _, vis in subj]

C = np.array([s[1] for s in subj]); N = len(subj)
fold_file = lambda f: out / f"external_cv_fold{f}.json"


def class_logp(F, k, i):
    """Log class-conditional density of each prefix of subject i under draws F (list over prefixes)."""
    if a.app == "gtn":
        return [gtn_logp(F, t, y) for _, t, y in pref[i]]
    return [pg.logp_infl(F, vis) for _, vis in pref[i]]


for f, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=1).split(np.zeros(N), C)):
    if fold_file(f).exists():
        print("fold", f, "already done"); continue
    if a.app == "gtn":
        ids_tr = [subj[i][0] for i in tr]; dtr = d[d.id.isin(ids_tr)]
        fits = [gtn.fit(dtr, f"df{k}", draws=dr, tune=tu) for k in (0, 1)]
        Fk = [{v: x[::2] for v, x in ft[0].items()} for ft in fits]; fit_diag = [ft[1][0] for ft in fits]
    else:
        ids_tr = [subj[i][0] for i in tr]; dtr = d[d.pid.isin(ids_tr)]
        fits = [pg.fit_infl(dtr, k, 1, None, dr, tu, target_accept=a.target_accept) for k in (0, 1)]
        Fk = [pg.thin(ft[0], 4) for ft in fits]; fit_diag = [ft[1][0] for ft in fits]
    nk = np.array([(C[tr] == 0).sum(), (C[tr] == 1).sum()])
    lp = {i: [class_logp(Fk[k], k, i) for k in (0, 1)] for i in list(tr) + list(te)}
    # leave-one-out logits of training subjects (PSIS on the subject's own class fit)
    rows_tr, khat = [], []
    for i in tr:
        c = C[i]; own = lp[i][c][-1]; lw, kh = psis(-own); khat.append(kh); n_ = nk - np.eye(2, dtype=int)[c]
        for j, pr in enumerate(pref[i]):
            ll = [np.log(n_[k] / (len(tr) - 1)) + (logsumexp(lp[i][k][j] + lw) if k == c else logsumexp(lp[i][k][j]) - np.log(len(lp[i][k][j])))
                  for k in (0, 1)]
            rows_tr.append((int(i), int(pr[0]), ll[1] - ll[0], int(c)))
    # exact leave-one-out refits for training subjects with Pareto shape estimate above 0.7
    ck_file = out / f"external_cv_fold{f}_refits.json"
    ck = json.load(open(ck_file)) if ck_file.exists() else {}
    for pos_i, i in enumerate(tr):
        if khat[pos_i] <= 0.7:
            continue
        c = C[i]; key = str(int(i))
        if key not in ck:
            pid = subj[i][0]
            if a.app == "gtn":
                ft = gtn.fit(dtr[dtr.id != pid], f"df{c}", draws=dr, tune=tu); Fi = {v: x_[::2] for v, x_ in ft[0].items()}
            else:
                ft = pg.fit_infl(dtr, c, 1, pid, dr, tu, target_accept=max(0.99, a.target_accept)); Fi = pg.thin(ft[0], 4)
            own_lp = class_logp(Fi, c, i); n_ = nk - np.eye(2, dtype=int)[c]; zz = []
            for j in range(len(pref[i])):
                ll = [np.log(n_[k] / (len(tr) - 1)) + logsumexp(own_lp[j] if k == c else lp[i][k][j]) - np.log(len(own_lp[j] if k == c else lp[i][k][j]))
                      for k in (0, 1)]
                zz.append(float(ll[1] - ll[0]))
            ck[key] = zz; ck["diag_" + key] = ft[1][0]; json.dump(ck, open(ck_file, "w"), default=float)
            print(f"  fold {f}: exact refit without training subject {int(i)} (khat {khat[pos_i]:.2f})", flush=True)
        it = iter(ck[key])                                   # rows of subject i are stored in prefix order
        rows_tr = [(r[0], r[1], next(it), r[3]) if r[0] == int(i) else r for r in rows_tr]
    x = np.array([r[2] for r in rows_tr]); y = np.array([r[3] for r in rows_tr])
    sub = np.array([r[0] for r in rows_tr]); lm = np.array([r[1] for r in rows_tr])
    rules, coef = fit_tempering(x, y, sub, lm, B=500, seed=f)
    bad = {int(i) for i, kh in zip(tr, khat) if not kh <= 0.7}
    # held-out subjects: plain posterior predictive of the training fit
    rows_te = []
    for i in te:
        for j, pr in enumerate(pref[i]):
            ll = [np.log(nk[k] / len(tr)) + logsumexp(lp[i][k][j]) - np.log(len(lp[i][k][j])) for k in (0, 1)]
            z = ll[1] - ll[0]; h = int(pr[0])
            rows_te.append({"subject": int(i), "landmark": h, "position": j, "y": int(C[i]), "DF": float(expit(z)),
                            **{m: float(rules[k]([z], [h])[0]) for m, k in (("DF-T", "single"), ("DF-Tw", "separate"), ("DF-Th", "shrunk"))}})
    json.dump({"fold": f, "test": rows_te, "khat_max_train": float(np.max(khat)), "n_khat_above_0.7": len(bad), "refitted": sorted(bad),
               "train_loo": [list(r) for r in rows_tr], "train_khat": [float(k) for k in khat], "fit_diagnostics": fit_diag,
               "coef": {"single": coef["single"].tolist(), "separate": {int(h): v.tolist() for h, v in coef["separate"].items()},
                        "shrunk": {int(h): v.tolist() for h, v in coef["shrunk"].items()}, "T": coef["T"].tolist()}},
              open(fold_file(f), "w"), indent=1, default=float)
    print("fold", f, "done; khat max", round(float(np.max(khat)), 2), flush=True)

if all(fold_file(f).exists() for f in range(FOLDS)):
    R = [json.load(open(fold_file(f))) for f in range(FOLDS)]
    rows = [r for fr in R for r in fr["test"]]
    np.savez(out / "external_cv.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    write_json({fr["fold"]: {"coef": fr["coef"], "khat_max_train": fr["khat_max_train"], "n_khat_above_0.7": fr["n_khat_above_0.7"]} for fr in R},
               out / "external_cv_coefficients.json")
    print("written", out / "external_cv.npz")
