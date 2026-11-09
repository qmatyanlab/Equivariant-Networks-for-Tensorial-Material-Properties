"""Per-record tensor/structure frame-consistency check.

For each record, build the point-group projector from the structure's own Cartesian symmetry operations
(spglib via pymatgen, symprec 1e-3) and report r = ||T - P_G T|| / ||T||. A record whose tensor and structure share
a frame has r ~ 0 (up to DFT noise). A large r means the tensor violates the symmetry of the structure it is paired
with: a frame mismatch or an unsymmetrised label.

Usage: python frame_check.py <target> <dataset_dir> <out.csv>
  target: dielectric | piezo | elastic
  dataset_dir: the folder holding train/validate/test/{data.json,material_ids.csv}
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.tensors import Tensor
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.piezo import PiezoTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

FIELD = {"dielectric": "dielectric_tensor", "piezo": "piezo_tensor", "elastic": "elasticity_tensor"}


def full_tensor(target, raw):
    a = np.array(raw, dtype=float)
    if target == "dielectric":
        return Tensor(a.reshape(3, 3))
    if target == "piezo":
        return PiezoTensor.from_voigt(a.reshape(3, 6))
    return ElasticTensor.from_voigt(a.reshape(6, 6))


def residual(t, ops):
    proj = np.mean([t.transform(op) for op in ops], axis=0)
    n = np.linalg.norm(t)
    return float(np.linalg.norm(t - proj) / n) if n > 0 else float("nan")


def main():
    target, root, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    rows = []
    for split in ["train", "validate", "test"]:
        ids = (root / split / "material_ids.csv").read_text().split("\n")
        data = json.loads((root / split / "data.json").read_text())
        assert len(ids) == len(data), (split, len(ids), len(data))
        for mid, d in zip(ids, data):
            s = Structure.from_dict(d["material"])
            t = full_tensor(target, d[FIELD[target]])
            row = {"material_id": mid, "split": split, "norm": float(np.linalg.norm(t))}
            for tag, prec in [("1e-3", 1e-3), ("1e-1", 1e-1)]:
                try:
                    sga = SpacegroupAnalyzer(s, symprec=prec)
                    ops = sga.get_point_group_operations(cartesian=True)
                    row[f"pg_{tag}"] = sga.get_point_group_symbol()
                    row[f"r_{tag}"] = residual(t, ops)
                except Exception as e:  # spglib failure: record and move on
                    row[f"pg_{tag}"] = f"ERR:{type(e).__name__}"
                    row[f"r_{tag}"] = float("nan")
            rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = np.array([x["r_1e-3"] for x in rows], dtype=float)
    agree = np.mean([x["pg_1e-3"] == x["pg_1e-1"] for x in rows])
    print(f"{target}: n={len(rows)}  point group agrees at 1e-3 vs 1e-1: {agree:.3f}")
    for thr in [0.01, 0.05, 0.2]:
        print(f"  fraction r>{thr}: {np.nanmean(r > thr):.3f}")
    print(f"  median r: {np.nanmedian(r):.2e}   90th pct: {np.nanpercentile(r, 90):.2e}")


if __name__ == "__main__":
    main()
