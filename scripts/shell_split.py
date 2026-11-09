"""Does the 12-nearest-neighbor truncation split a shell of equidistant neighbors (breaking the crystal symmetry of the
graph)? Compares the symmetry-forbidden piezoelectric components predicted for materials with and without a split
shell, and the share of the error they carry (Results 3.1, Supplementary Note S11).

The crystal graph (src/Dataset/CrystalSO3Dataset.py) keeps, for each atom, the 12 nearest neighbors within 8 A, sorted
by distance; when the 12th and 13th neighbors are equidistant, only part of the shell is kept and the graph has a lower
symmetry than the crystal. Forbidden components: those zeroed by the point-group projector of the structure (the
`allowed` mask written by eval_dump.py, spglib tolerance 1e-3 A).

Usage: python shell_split.py <piezo test dir with material_ids.csv and data.json> <npz dir from eval_dump.py>
       e.g. python scripts/shell_split.py src/Dataset/piezo/test analysis/eval/npz
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

RADIUS, MAX_NEIGHBORS, TOL = 8.0, 12, 1e-3

test_dir, npz_dir = Path(sys.argv[1]), Path(sys.argv[2])
ids = test_dir.joinpath("material_ids.csv").read_text().split("\n")
data = json.loads(test_dir.joinpath("data.json").read_text())
split = {}
for i, d in zip(ids, data):
    s = Structure.from_dict(d["material"])
    split[i] = False
    for nbrs in s.get_all_neighbors(RADIUS):
        dist = np.sort([n.nn_distance for n in nbrs])
        if len(dist) > MAX_NEIGHBORS and dist[MAX_NEIGHBORS] - dist[MAX_NEIGHBORS - 1] < TOL:
            split[i] = True
            break
print(f"piezoelectric test materials: {len(ids)}; with a split shell: {sum(split.values())}")

worst_unsplit = 0.0
for f in sorted(glob.glob(str(npz_dir / "R2026_*_-_Piezo_-_*.npz"))):
    z = np.load(f)
    A = z["allowed"].astype(bool)
    P, T = z["pred"].reshape(A.shape), z["true"].reshape(A.shape)
    fp = np.where(~A, np.abs(P), 0).reshape(len(P), -1).max(1)  # largest forbidden component per material
    sm = np.array([split[str(k)] for k in z["ids"]])
    E = np.abs(P - T)
    worst_unsplit = max(worst_unsplit, float(fp[~sm].max()))
    print(f"{Path(f).stem:60s} max forbidden: split {fp[sm].max():.3f} | not split {fp[~sm].max():.1e} C/m2; "
          f"forbidden share of the error {100 * E[~A].sum() / E.sum():.1f}%; "
          f"MAE {E.mean():.5f} -> {np.where(A, E, np.abs(T)).mean():.5f} with forbidden set to 0")
print(f"largest forbidden component for materials without a split shell, over all models: {worst_unsplit:.1e} C/m2")
