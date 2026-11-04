"""Shared helpers for the scripts (path setup, output folders, JSON writing)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"


def outdir(name):
    p = RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
