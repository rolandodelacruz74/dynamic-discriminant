"""Accuracy of the 7 x 7 Gauss-Hermite quadrature of the quadratic joint model (JMQ) in the simulation.
For the first replicates of the scenarios where JMQ matters, each replicate is run twice, with 7 and with
15 nodes per dimension (same data, same seeds), and we compare (i) the JMQ objective at the 7-node estimate
evaluated with 7 and with 25 x 25 nodes, and (ii) the refitted JMQ predictions' AUC, Brier score and
calibration slope by landmark.  python scripts/09_jmq_quadrature_check.py [--reps 3]
Outputs results/jmq_check/*.csv and jmq_check.json."""
import argparse, json, itertools
import numpy as np
from _common import RESULTS
from dyndisc.simulation import run_scenario, HEADER
from concurrent.futures import ProcessPoolExecutor

GRID = list(itertools.product(["M", "H"], ["iid", "ar"], ["gauss", "mix"], [150, 450])) + [("J", "iid", "gauss", 150), ("J", "iid", "gauss", 450)]
SC = [("H", "iid", "mix", 450), ("M", "iid", "mix", 450), ("J", "iid", "gauss", 450), ("H", "iid", "mix", 150)]
out = RESULTS / "jmq_check"


def one(args):
    sc, nodes, reps = args[:3]; jm = args[3] if len(args) > 3 else 60
    seed = 1001 + GRID.index(sc); chk = []
    lines = run_scenario(*sc, reps, seed, jmq_nodes=nodes, jmq_check=chk, jm_nodes=jm)
    tag = f"{'_'.join(map(str, sc))}_n{nodes}" + ("" if jm == 60 else f"_jm{jm}")
    (out / f"{tag}.csv").write_text(HEADER + "\n" + "\n".join(lines) + "\n")
    return tag, chk


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--regime-m", action="store_true", help="regime M (Gaussian and bimodal): JM with 60 and 150 nodes, JMQ with 7 and 15")
    a = ap.parse_args()
    if a.regime_m:
        tasks = [(sc, n, a.reps, jm) for sc in [("M", "iid", "gauss", 450), ("M", "iid", "mix", 450)] for n, jm in ((7, 60), (15, 150))]; fn = "jm_regime_m_check.json"
    else:
        tasks = [(sc, n, a.reps) for sc in SC for n in (7, 15)]; fn = "jmq_check.json"
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for tag, chk in ex.map(one, tasks):
            res[tag] = chk; print("done", tag, flush=True)
    json.dump(res, open(out / fn, "w"), indent=1, default=float)
