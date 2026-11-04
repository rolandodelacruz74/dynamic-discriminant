"""GTN application (Section 6.1, Table 3): Bayesian discriminant model, joint model and tempering.

  python scripts/03_gtn.py                       uses data/gtn/hCG_GTN.csv if present (see data/README.md),
                                                 otherwise the synthetic data set
  python scripts/03_gtn.py --data FILE --quick   shorter chains for a smoke test
Outputs in results/gtn/ (or results/gtn_synthetic/): posterior draws, convergence diagnostics,
leave-one-out predictions and table_loo.csv (tempering: 07_nested_cv_tempering.py)."""
import argparse
import numpy as np
import pandas as pd
from _common import ROOT, outdir, write_json
from dyndisc import gtn
from dyndisc.metrics import summary, logit

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None); ap.add_argument("--quick", action="store_true")
ap.add_argument("--skip-fits", action="store_true", help="reuse posterior draws already in the output folder")
a = ap.parse_args()
real = ROOT / "data" / "gtn" / "hCG_GTN.csv"
path = a.data or (real if real.exists() else ROOT / "data" / "gtn" / "gtn_synthetic.csv")
tag = "gtn" if str(path).endswith("hCG_GTN.csv") else "gtn_synthetic"
out = outdir(tag + ("_quick" if a.quick else ""))
print("data:", path)
d = gtn.load(path)
print("women:", d.id.nunique(), " GTN:", int(d.groupby("id").C.first().sum()), " measurements:", len(d))
draws, tune = (300, 500) if a.quick else (1000, 1500)
F, diag = {}, {}
import json
prev = json.load(open(out / "convergence.json")) if (out / "convergence.json").exists() else {}
for m in ("df0", "df1", "jm"):
    if a.skip_fits and (out / f"draws_{m}.npz").exists():
        F[m] = dict(np.load(out / f"draws_{m}.npz")); diag[m] = prev.get(m, {}); continue
    F[m], (dg, summ) = gtn.fit(d, m, draws=draws, tune=tune)
    np.savez(out / f"draws_{m}.npz", **F[m]); diag[m] = dg
    summ.to_csv(out / f"posterior_summary_{m}.csv")
    print(m, dg)
write_json(diag, out / "convergence.json")
pred, kh, C = gtn.loo_predictions(d, F)
print("PSIS khat max:", {m: round(float(v.max()), 2) for m, v in kh.items()})
write_json({m: float(v.max()) for m, v in kh.items()}, out / "psis_khat_max.json")
keys = list(pred["DF"].keys())
sub = np.array([k[0] for k in keys]); wk = np.array([k[1] for k in keys]); y = C[sub]
P = {"DF": np.array([pred["DF"][k] for k in keys]), "JM": np.array([pred["JM"][k] for k in keys])}
rows = []
for s in gtn.WEEKS:
    c = wk == s; r = {"week": s, "n": int(c.sum())}
    for m in P:
        sm = summary(P[m][c], y[c]); r.update({f"auc_{m}": sm["auc"], f"brier_{m}": sm["brier"], f"slope_{m}": sm["slope"]})
    rows.append(r)
tab = pd.DataFrame(rows); tab.to_csv(out / "table_loo.csv", index=False)
print(tab.round(3).to_string(index=False))
np.savez(out / "loo_predictions.npz", subject=sub, week=wk, y=y, **P)
