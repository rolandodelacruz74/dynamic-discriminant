"""Figure 2: trajectories and dynamically updated probabilities for four women (two GTN, two pregnancy).
Requires the outputs of 03_gtn.py and 04_pregnancy.py.
  python scripts/05_figure_patients.py [--synthetic]"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import ROOT, RESULTS
from dyndisc import gtn, pregnancy as pg

ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--temper-gtn", default="DF-Tw"); ap.add_argument("--temper-pregnancy", default="DF-T"); a = ap.parse_args()
sfx = "_synthetic" if a.synthetic else ""
G = np.load(RESULTS / f"gtn{sfx}" / "loo_predictions.npz")
gpath = ROOT / "data" / "gtn" / ("gtn_synthetic.csv" if a.synthetic else "hCG_GTN.csv")
gd = gtn.load(gpath); gp = [(i, g.t.values, g.loghcg.values) for i, g in gd.groupby("id")]
gpred = {m: {(int(s), int(w)): p for s, w, p in zip(G["subject"], G["week"], G[m])} for m in ("DF", "JM")}
GE = np.load(RESULTS / f"gtn{sfx}" / "external_cv.npz")          # tempered predictions from the outer folds of the nested cross-validation
gpred["DF_Th"] = {(int(s), int(w)): p for s, w, p in zip(GE["subject"], GE["landmark"], GE[a.temper_gtn])}
Cg = np.array([int(g.C.iloc[0]) for _, g in gd.groupby("id")])
pred, ids = np.load(RESULTS / f"pregnancy{sfx}" / "loo_predictions.npy", allow_pickle=True)
ppath = ROOT / "data" / "pregnancy" / ("pregnancy_synthetic.txt" if a.synthetic else "datos.txt")
pats = pg.patients(pg.load(ppath))
assert [(p, c) for p, c, _ in pats] == [tuple(x) for x in ids], "predictions and data file do not match"
Cp = np.array([c for _, c, _ in pats])

# selection rules (deterministic)
cand = []
for i, (_, t, _) in enumerate(gp):
    ws = [w for w in range(2, 8) if (i, w) in gpred["DF"]]
    if len(t) >= 5:
        df = [gpred["DF"][(i, w)] for w in ws]; cand.append((i, Cg[i], df[0], df[-1], max(df)))
gA = max([c for c in cand if c[1] == 1], key=lambda c: c[3] - c[2])[0]
gB = max([c for c in cand if c[1] == 0 and c[3] < 0.1], key=lambda c: c[4])[0]
het, inf, jm = pred["DF, fixed inflection"], pred["DF, random inflection"], pred["JM, random inflection"]
PE = np.load(RESULTS / f"pregnancy{sfx}" / "external_cv.npz")
th = [[float(PE[a.temper_pregnancy][(PE["subject"] == i) & (PE["position"] == j)][0]) for j in range(len(pats[i][2]))] for i in range(len(pats))]
nv = np.array([len(v) for _, _, v in pats])
pC = max([i for i in range(len(Cp)) if Cp[i] == 0 and nv[i] >= 3 and het[i][-1] > 0.8 and inf[i][-1] < 0.5] or
         [int(np.argmax([het[i][-1] - inf[i][-1] if Cp[i] == 0 and nv[i] >= 3 else -9 for i in range(len(Cp))]))],
         key=lambda i: het[i][-1] - inf[i][-1])
pD = max([i for i in range(len(Cp)) if Cp[i] == 1 and nv[i] >= 3], key=lambda i: inf[i][-1] - inf[i][0])

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
COL = {"DF": "#2a78d6", "DF-Th": "#1baf7a", "JM": "#eb6834", "DF fixed": "#8a8980"}
STY = {"DF": dict(ls="-", marker="o"), "DF-Th": dict(ls="--", marker="s"), "JM": dict(ls=":", marker="^"), "DF fixed": dict(ls="-.", marker="D")}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(4, 2, figsize=(7.2, 8.6), gridspec_kw={"height_ratios": [1, 1.25, 1, 1.25], "hspace": 0.75, "wspace": 0.28})


def prob_panel(axp, xs, series, xlabel):
    axp.axhline(0.5, color=GRID, lw=1, zorder=0)
    for name, ys in series.items():
        axp.plot(xs, ys, color=COL[name], lw=2, ms=5, markeredgecolor="white", markeredgewidth=0.8, **STY[name])
    axp.set_ylim(-0.03, 1.03); axp.set_ylabel("Probability"); axp.set_xlabel(xlabel); axp.yaxis.grid(True, color=GRID, lw=0.6)


for col, i in enumerate((gA, gB)):
    wid, t, y = gp[i]; axt = ax[0, col]
    axt.plot(t, y, color=INK, lw=1.2, marker="o", ms=4); axt.set_xlim(1.6, 7.4); axt.set_ylabel(r"$\log_{10}$ hCG")
    axt.yaxis.grid(True, color=GRID, lw=0.6)
    axt.set_title(f"GTN, woman {wid}: {'GTN' if Cg[i] else 'no GTN'}", loc="left", fontsize=9, color=INK)
    ws = [w for w in range(2, 8) if (i, w) in gpred["DF"]]
    prob_panel(ax[1, col], ws, {"DF": [gpred["DF"][(i, w)] for w in ws], "DF-Th": [gpred["DF_Th"][(i, w)] for w in ws],
                                "JM": [gpred["JM"][(i, w)] for w in ws]}, "Week after evacuation")
    ax[1, col].set_xlim(1.6, 7.4)
for col, i in enumerate((pC, pD)):
    pid, c, vis = pats[i]; t, h, e = vis[:, 0], vis[:, 1], vis[:, 2]; axt = ax[2, col]
    axt.plot(t, h, color=INK, lw=1.2, marker="o", ms=4, label=r"$\beta$-HCG")
    ok = ~np.isnan(e); axt.plot(t[ok], e[ok], color=MUTED, lw=1.2, marker="^", ms=4, ls="--", label="Estradiol")
    axt.set_ylabel(r"$\log_{10}$ concentration"); axt.yaxis.grid(True, color=GRID, lw=0.6)
    axt.set_title(f"Pregnancy, woman {pid + 1}: {'abnormal' if c else 'normal'}", loc="left", fontsize=9, color=INK)
    if col == 0:
        axt.legend(frameon=False, fontsize=7.5, loc="lower right")
    prob_panel(ax[3, col], t, {"DF fixed": het[i], "DF": inf[i], "DF-Th": th[i], "JM": jm[i]}, "Day of gestation")
    pad = 0.06 * (t.max() - t.min() + 1)
    for r in (2, 3):
        ax[r, col].set_xlim(t.min() - pad, t.max() + pad)
handles = [plt.Line2D([], [], color=COL[n], lw=2, ms=5, markeredgecolor="white", **STY[n]) for n in ("DF", "DF-Th", "JM", "DF fixed")]
fig.legend(handles, ["Discriminant model (DF)", "Tempered DF (nested cross-validation)", "Joint model (JM)", "DF, fixed inflection (pregnancy)"],
           loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
out = RESULTS / f"figure2_patients{sfx}"
fig.savefig(f"{out}.pdf", bbox_inches="tight"); fig.savefig(f"{out}.png", dpi=170, bbox_inches="tight")
print("written", out, "| GTN women", gp[gA][0], gp[gB][0], "| pregnancy women", pats[pC][0] + 1, pats[pD][0] + 1)
