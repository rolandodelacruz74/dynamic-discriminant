"""Generate the synthetic data sets in data/ from data/synthetic_parameters.json.
They have the structure of the original data (class sizes, numbers and timing of visits, missingness rates)
and values simulated from the fitted discriminant models, so all scripts can be run end to end.
Results obtained with them are not the results of the paper."""
import json
import numpy as np
from _common import ROOT

rng = np.random.default_rng(2026)
par = json.load(open(ROOT / "data" / "synthetic_parameters.json"))

# pregnancy: random-inflection model, two markers
rows, pid, order = [], 0, 0
for k in ("0", "1"):
    p = par["pregnancy"][k]
    nv = np.array([int(float(x)) for x in p["visits_freq"]]); pv = np.array(list(p["visits_freq"].values()), float); pv /= pv.sum()
    q = np.array(p["day_quantiles"]); Sig = np.array(p["Sigma"])
    for _ in range(p["n"]):
        pid += 1; n = rng.choice(nv, p=pv)
        days = np.sort(np.unique(np.round(np.interp(rng.random(n), np.linspace(0, 1, len(q)), q))))
        a, e, c = rng.multivariate_normal(p["mean"], np.array(p["D"]))
        for t in days:
            eps = rng.multivariate_normal([0, 0], Sig)
            h = a / (1 + np.exp((c - t) / p["scale"])) + eps[0]; e2 = e + p["slope"] * t + eps[1]
            mh, me = rng.random() < p["p_missing_hcg"], rng.random() < p["p_missing_e2"]
            if mh and me:
                me = False
            order += 1
            rows.append(f"{order} {pid} {int(t)} {'NA' if mh else f'{h:.6f}'} {'NA' if me else f'{e2:.6f}'} {k}")
(ROOT / "data" / "pregnancy" / "pregnancy_synthetic.txt").write_text("Order Patient Time SubUnit Estradiol Class\n" + "\n".join(rows) + "\n")

# GTN: class-specific linear mixed models, weeks 2-7
rows, wid = [], 0
for k in ("0", "1"):
    p = par["gtn"][k]; b = np.array(p["b"]); D = np.array(p["D"])
    for _ in range(p["n"]):
        wid += 1; u = rng.multivariate_normal([0, 0], D)
        while True:
            obs = [rng.random() >= p["p_missing_by_week"][str(w)] for w in range(2, 8)]
            if any(obs):
                break
        for w, o in zip(range(2, 8), obs):
            y = b[0] + u[0] + (b[1] + u[1]) * w + rng.normal(0, p["s"])
            rows.append(f"{wid},{w},{y:.4f},{k}" if o else f"{wid},{w},999,{k}")
(ROOT / "data" / "gtn" / "gtn_synthetic.csv").write_text("id,weeknr,loghcg,PTD\n" + "\n".join(rows) + "\n")
print("written data/pregnancy/pregnancy_synthetic.txt and data/gtn/gtn_synthetic.csv")
