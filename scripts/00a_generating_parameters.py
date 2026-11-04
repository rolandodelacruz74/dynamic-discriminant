"""Extract the aggregate quantities used to generate the synthetic data sets (requires the original data and
the posterior draws of the fits in results/). Run by the author; the output, data/synthetic_parameters.json,
is distributed with the code. It contains only posterior means and aggregate frequencies."""
import json
import sys
import numpy as np
import pandas as pd
from _common import RESULTS, ROOT
from dyndisc import pregnancy, gtn

preg_path, gtn_path = sys.argv[1], sys.argv[2]
d = pregnancy.load(preg_path); P = {}
for k in (0, 1):
    f = dict(np.load(RESULTS / "pregnancy" / f"draws_infl{k}.npz"))
    g = d[d.cls == k]
    P[str(k)] = {"n": int(g.pid.nunique()),
                 "mean": [float(f["A"].mean()), float(f["E"].mean()), float(f["infl"].mean())],
                 "D": f["D"].mean(0).tolist(), "Sigma": f["Sigma"].mean(0).tolist(),
                 "scale": float(f["scale"].mean()), "slope": float(f["slope"].mean()),
                 "visits_freq": g.groupby("pid").size().value_counts(normalize=True).sort_index().to_dict(),
                 "day_quantiles": np.quantile(g.t, np.linspace(0, 1, 21)).tolist(),
                 "p_missing_e2": float(g.e2.isna().mean()), "p_missing_hcg": float(g.hcg.isna().mean())}
G = {}
dg = pd.read_csv(gtn_path, na_values=["999"]).rename(columns={"PTD": "C", "weeknr": "t"})
dg = dg[dg.id.isin(dg.dropna(subset=["loghcg"]).id)]
for k in (0, 1):
    f = dict(np.load(RESULTS / "gtn" / f"draws_df{k}.npz")); g = dg[dg.C == k]
    G[str(k)] = {"n": int(g.id.nunique()), "b": f["b"].mean(0).tolist(), "D": f["D"].mean(0).tolist(), "s": float(f["s"].mean()),
                 "p_missing_by_week": g.groupby("t").loghcg.apply(lambda s: float(s.isna().mean())).to_dict()}
json.dump({"pregnancy": P, "gtn": G}, open(ROOT / "data" / "synthetic_parameters.json", "w"), indent=1, default=float)
print("written data/synthetic_parameters.json")
