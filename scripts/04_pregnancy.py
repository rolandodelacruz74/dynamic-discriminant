"""Pregnancy application (Section 6.2, Table 4 and the text): fits, leave-one-out evaluation and tempering.

  python scripts/04_pregnancy.py                 uses data/pregnancy/datos.txt if present (original data, available
                                                 from the author on request), otherwise the synthetic data set
  python scripts/04_pregnancy.py --data FILE --quick
Outputs in results/pregnancy/ (or results/pregnancy_synthetic/): posterior draws, diagnostics, leave-one-out
predictions (exact refits where the Pareto shape estimate exceeds 0.7), LPML and table_loo.csv. Tempering is
evaluated by nested cross-validation in 07_nested_cv_tempering.py."""
import argparse
import numpy as np
import pandas as pd
from _common import ROOT, outdir, write_json
from dyndisc import pregnancy as pg
from dyndisc.metrics import summary, logit

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None); ap.add_argument("--quick", action="store_true")
ap.add_argument("--skip-fits", action="store_true", help="reuse posterior draws already in the output folder")
a = ap.parse_args()
real = ROOT / "data" / "pregnancy" / "datos.txt"
path = a.data or (real if real.exists() else ROOT / "data" / "pregnancy" / "pregnancy_synthetic.txt")
tag = "pregnancy" if str(path).endswith("datos.txt") else "pregnancy_synthetic"
out = outdir(tag + ("_quick" if a.quick else ""))
print("data:", path)
d = pg.load(path); pats = pg.patients(d); C = np.array([c for _, c, _ in pats]); N = len(pats)
print("women:", N, " abnormal:", int(C.sum()), " visits:", len(d))
q = a.quick
dr, tu = (300, 500) if q else (1000, 1500)
df_, tf_ = (dr, tu) if q else (2000, 2000)   # longer chains for the fixed-inflection models (random effects sampled)
specs = {"het0": lambda: pg.fit_fixed(d, "het0", df_, tf_), "het1": lambda: pg.fit_fixed(d, "het1", df_, tf_),
         "hom": lambda: pg.fit_fixed(d, "hom", df_, tf_), "jm": lambda: pg.fit_fixed(d, "jm", df_, tf_),
         "infl0": lambda: pg.fit_infl(d, 0, 1, None, dr, tu), "infl1": lambda: pg.fit_infl(d, 1, 1, None, dr, tu),
         "inflmix1": lambda: pg.fit_infl(d, 1, 2, None, dr, tu),
         "jm_infl": lambda: pg.fit_jm_infl(d, *((300, 500) if q else (2000, 3000)))}
import json
prev_diag = json.load(open(out / "convergence.json")) if (out / "convergence.json").exists() else {}
F, diag = {}, {}
for m, f in specs.items():
    if a.skip_fits and (out / f"draws_{m}.npz").exists():
        F[m] = dict(np.load(out / f"draws_{m}.npz")); diag[m] = prev_diag.get(m, {}); continue
    F[m], (dg, summ) = f(); np.savez(out / f"draws_{m}.npz", **F[m]); diag[m] = dg
    summ.to_csv(out / f"posterior_summary_{m}.csv"); print(m, dg, flush=True)
if diag:
    write_json(diag, out / "convergence.json")

th = 4   # thinning for the grid-integrated models, as in the paper
res = {}
res["DF fixed"] = pg.loo_discriminant(pats, [pg.class_params(F, "het", k) for k in (0, 1)], pg.logp_fixed)
res["DF hom"] = pg.loo_discriminant(pats, [pg.class_params(F, "hom", k) for k in (0, 1)], pg.logp_fixed)
infl = [pg.thin(F["infl0"], th), pg.thin(F["infl1"], th)]
res["DF infl"] = pg.loo_discriminant(pats, infl, pg.logp_infl)
res["DF infl mix"] = pg.loo_discriminant(pats, [infl[0], pg.thin(F["inflmix1"], th)], pg.logp_infl)
for m in res:
    print(f"{m}: LPML {res[m][1]:.2f}, khat max {res[m][2].max():.2f}, n(khat > 0.7) {(res[m][2] > 0.7).sum()}")
# exact leave-one-out refits wherever the Pareto shape estimate exceeds 0.7
import json as _json
prev_exact = _json.load(open(out / "exact_loo_refits.json")) if (a.skip_fits and (out / "exact_loo_refits.json").exists()) else {}
exact = {"DF, random inflection": {}, "JM, random inflection": {}, "JM, fixed inflection": {}}
for i in [int(i) for i in np.where(res["DF infl"][2] > 0.7)[0]]:
    pid, c, _ = pats[i]; key = str(pid + 1)
    old = prev_exact.get("DF, random inflection", {}).get(key)
    if old is None:
        Fi, (dg, _) = pg.fit_infl(d, c, 1, pid, *((dr, tu) if q else (2000, 3000)), target_accept=0.99)
        old = {"diagnostics": dg, "probabilities": pg.exact_loo_patient(pats, i, pg.thin(Fi, th), infl, pg.logp_infl)}
    print(f"DF exact refit without woman {pid + 1}: PSIS {np.round(res['DF infl'][0][i], 3)} -> exact {np.round(old['probabilities'], 3)}")
    exact["DF, random inflection"][key] = {**old, "khat": float(res["DF infl"][2][i])}; res["DF infl"][0][i] = old["probabilities"]
jm_fixed, kh1 = pg.loo_joint(pats, F["jm"], pg.jm_terms)
jm_infl, kh2 = pg.loo_joint(pats, pg.thin(F["jm_infl"], th), pg.jm_infl_terms)
print(f"JM fixed: khat max {kh1.max():.2f};  JM infl: khat max {kh2.max():.2f}")
for name, kh, prd, fitter, terms, thin_ in (("JM, fixed inflection", kh1, jm_fixed, lambda dd: pg.fit_fixed(dd, "jm", df_, tf_), pg.jm_terms, 1),
                                           ("JM, random inflection", kh2, jm_infl, lambda dd: pg.fit_jm_infl(dd, *((300, 500) if q else (2000, 3000))), pg.jm_infl_terms, th)):
    for i in [int(i) for i in np.where(kh > 0.7)[0]]:
        pid = pats[i][0]; key = str(pid + 1)
        old = prev_exact.get(name, {}).get(key)
        if old is None:
            Fi, (dg, _) = fitter(d[d.pid != pid])
            old = {"diagnostics": dg, "probabilities": pg.exact_loo_joint(pats, i, pg.thin(Fi, thin_), terms)}
        print(f"{name} exact refit without woman {pid + 1}: PSIS {np.round(prd[i], 3)} -> exact {np.round(old['probabilities'], 3)}", flush=True)
        exact[name][key] = {**old, "khat": float(kh[i])}; prd[i] = old["probabilities"]
        write_json(exact, out / "exact_loo_refits.json")
write_json(exact, out / "exact_loo_refits.json")

pred = {"JM, fixed inflection": jm_fixed, "JM, random inflection": jm_infl,
        "DF, fixed inflection": res["DF fixed"][0], "DF, random inflection": res["DF infl"][0],
        "DF mixture, random inflection": res["DF infl mix"][0]}
tab = []
for m, pr in pred.items():
    for lab, j in (("all", None), ("1", 1), ("2", 2), ("3", 3)):
        idx = list(range(N)) if j is None else [i for i in range(N) if len(pats[i][2]) >= j]
        p = np.array([pr[i][-1] if j is None else pr[i][j - 1] for i in idx])
        sm = summary(p, C[idx]); tab.append({"model": m, "visits": lab, "n": len(idx), **sm})
tab = pd.DataFrame(tab); tab.to_csv(out / "table_loo.csv", index=False)
pd.set_option("display.width", 220)
print(tab.pivot(index="model", columns="visits", values=["auc", "brier", "slope"]).round(3).to_string())
print(tab[tab.visits == "all"][["model", "error", "sens", "spec"]].round(3).to_string(index=False))
write_json({m: {"LPML": res[m][1], "khat_max": float(res[m][2].max())} for m in res} |
           {"JM khat max": {"fixed": float(kh1.max()), "random": float(kh2.max())}}, out / "lpml.json")
np.save(out / "loo_predictions.npy", np.array([pred, [(int(p_), int(c)) for p_, c, _ in pats]], dtype=object), allow_pickle=True)  # no measurements stored
