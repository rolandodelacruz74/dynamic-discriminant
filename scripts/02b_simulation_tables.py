"""Simulation tables (Section 5, Tables 1 and 2; Web Tables in Web Appendix D) from the CSV files of 02_simulation.py.
  python scripts/02b_simulation_tables.py [--quick]
Writes CSV summaries and LaTeX table bodies to results/simulation/."""
import argparse
import glob
import numpy as np
import pandas as pd
from _common import RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
folder = RESULTS / ("simulation_quick" if a.quick else "simulation")
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(folder / "res_*.csv")))])
d["sc"] = d.regime + "-" + d.errors + "-" + d.re
print("Replicates per scenario:", d.groupby(["sc", "n"]).rep.nunique().to_dict())
auc = d.groupby(["n", "sc", "method"]).auc.mean().unstack()
exb = d.groupby(["n", "sc", "method"]).brier_excess.mean().unstack() * 1000
slope = d.groupby(["n", "sc", "method", "week"]).slope.mean().unstack()
citl = d.groupby(["n", "sc", "method", "week"]).citl.mean().unstack()
# Monte Carlo standard errors of the mean AUC difference DF - JM and of mean slopes
w = d.pivot_table(index=["n", "sc", "rep"], columns="method", values="auc", aggfunc="mean")
mcse_auc = (w.DF - w.JM).groupby(level=[0, 1]).agg(lambda v: v.std(ddof=1) / np.sqrt(len(v)))
mcse_slope = d.groupby(["n", "sc", "method", "week"]).slope.agg(lambda v: v.std(ddof=1) / np.sqrt(len(v)))
for name, t in (("auc", auc), ("excess_brier", exb), ("slope", slope), ("citl", citl)):
    t.to_csv(folder / f"summary_{name}.csv")
print("max MCSE of AUC(DF) - AUC(JM):", round(float(mcse_auc.max()), 4), "; max MCSE of mean slope (weeks 3-7):",
      round(float(mcse_slope.xs(3, level="week").max() if 3 in mcse_slope.index.get_level_values("week") else 0), 3))

ORDER = ["M-iid-gauss", "M-iid-mix", "M-ar-gauss", "M-ar-mix", "H-iid-gauss", "H-iid-mix", "H-ar-gauss", "H-ar-mix", "J-iid-gauss"]
lab = lambda sc: {"M": "M", "H": "H", "J": "J"}[sc[0]] + ", " + ("iid" if "iid" in sc else "AR") + ", " + ("Gaussian" if "gauss" in sc else "bimodal")
f3 = lambda v: f"{v:.3f}"; f1 = lambda v: f"{v:.1f}"; f2 = lambda v: f"{v:.2f}"


def table1(n):
    rows = []
    for sc in ORDER:
        if (n, sc) not in auc.index:
            continue
        A = auc.loc[(n, sc)]; B = exb.loc[(n, sc)]
        rows.append(lab(sc) + " & " + " & ".join([f3(A[m]) for m in ("TS", "JM", "JMQ", "DF", "OR")] +
                                               [f1(B[m]) for m in ("TS", "TSL", "JM", "JMQ", "DF", "DFTw")]) + r"\\")
    return "\n".join(rows)


def slope_rows(n, spec, tab=slope):
    rows = []
    for sc, methods in spec:
        first = True
        for m, name in methods:
            if (n, sc, m) not in tab.index:
                continue
            v = tab.loc[(n, sc, m)]
            rows.append((lab(sc) if first else "") + f" & {name} & " + " & ".join(f2(v[w_]) for w_ in range(2, 8)) + r"\\")
            first = False
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1])


SPEC2 = [("M-iid-gauss", [("TS", "TS"), ("TSL", "TSL"), ("JM", "JM"), ("JMQ", "JMQ"), ("DF", "DF")]),
         ("M-iid-mix", [("DF", "DF"), ("DFT", "DF-T"), ("DFTw", "DF-Tw")]),
         ("H-iid-mix", [("JMQ", "JMQ"), ("DF", "DF"), ("DFT", "DF-T"), ("DFTw", "DF-Tw")]),
         ("J-iid-gauss", [("TS", "TS"), ("JM", "JM"), ("DF", "DF"), ("DFTw", "DF-Tw")])]
open(folder / "table1_body.tex", "w").write(table1(450))
open(folder / "table2_body.tex", "w").write(slope_rows(450, SPEC2))
ALL = [(sc, [(m, m.replace("DFT", "DF-T").replace("DF-Tw", "DF-Tw").replace("DF-Th", "DF-Th")) for m in
             ("TS", "TSL", "JM", "JMQ", "DF", "DFT", "DFTw", "DFTh", "OR")]) for sc in ORDER]
for n in (450, 150):
    open(folder / f"webtable_slope_{n}.tex", "w").write(slope_rows(n, ALL))
    open(folder / f"webtable_citl_{n}.tex", "w").write(slope_rows(n, ALL, citl))
    rows = []
    for sc in ORDER:
        if (n, sc) not in auc.index:
            continue
        A = auc.loc[(n, sc)]; B = exb.loc[(n, sc)]
        rows.append(lab(sc) + " & " + " & ".join([f3(A[m]) for m in ("TS", "TSL", "JM", "JMQ", "DF", "OR")] +
                                               [f1(B[m]) for m in ("TS", "TSL", "JM", "JMQ", "DF", "DFT", "DFTw", "DFTh")]) + r"\\")
    open(folder / f"webtable_auc_brier_{n}.tex", "w").write("\n".join(rows))
pd.set_option("display.width", 220)
print(auc.round(3)); print(exb.round(2))
