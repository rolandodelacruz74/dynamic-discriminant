"""Simulation study (Section 5). Writes one CSV per scenario to results/simulation/.

  python scripts/02_simulation.py               all 18 scenarios, 50 replicates (about 6 h on 2 cores)
  python scripts/02_simulation.py --quick       2 scenarios x 2 replicates (a few minutes)
  python scripts/02_simulation.py --jobs 2      run scenarios in parallel
Seeds are 1001, 1002, ... in the order of GRID (regime x errors x random effects x n, then the two regime-J scenarios)."""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from _common import outdir
from dyndisc.simulation import run_scenario, HEADER

GRID = list(itertools.product(["M", "H"], ["iid", "ar"], ["gauss", "mix"], [150, 450])) + [("J", "iid", "gauss", 150), ("J", "iid", "gauss", 450)]


def one(args):
    (reg, err, re, n), reps, seed, out = args
    path = out / f"res_{reg}_{err}_{re}_{n}.csv"; part = out / f"partial_{reg}_{err}_{re}_{n}.csv"
    done = []
    if part.exists():
        done = [l for l in part.read_text().splitlines()[1:] if l]
    start = len({l.split(",")[4] for l in done})
    if not part.exists():
        part.write_text(HEADER + "\n")

    def save(lines):
        with open(part, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    run_scenario(reg, err, re, n, reps, seed, start=start, callback=save)
    part.rename(path)
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip scenarios whose output file already exists")
    a = ap.parse_args()
    out = outdir("simulation_quick" if a.quick else "simulation")
    tasks = [(sc, (2 if a.quick else a.reps), 1001 + i, out) for i, sc in enumerate(GRID)
             if not a.quick or sc in [("M", "iid", "gauss", 150), ("H", "iid", "mix", 150)]]
    if a.resume:
        tasks = [t for t in tasks if not (out / "res_{}_{}_{}_{}.csv".format(*t[0])).exists()]
    tasks.sort(key=lambda t: -t[0][3])
    with ProcessPoolExecutor(a.jobs) as ex:
        for p in ex.map(one, tasks):
            print("written", p, flush=True)
