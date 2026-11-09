"""Does rotating the structure into its IEEE frame fix the tensor/structure frame mismatch? (sample of records)"""
import csv, json, os, sys, random, warnings, numpy as np
warnings.filterwarnings("ignore")
from pymatgen.core import Structure
from pymatgen.core.tensors import Tensor
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_check import full_tensor, residual
target, root, csvp, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = {r["material_id"]: r for r in csv.DictReader(open(csvp))}
recs = {}
for split in ["train", "validate", "test"]:
    ids = open(f"{root}/{split}/material_ids.csv").read().split("\n")
    for i, d in zip(ids, json.load(open(f"{root}/{split}/data.json"))): recs[i] = d
field = {"dielectric": "dielectric_tensor", "piezo": "piezo_tensor", "elastic": "elasticity_tensor"}[target]
random.seed(0)
for label, sel in [("bad", [k for k, r in rows.items() if float(r["r_1e-3"]) > 0.01]), ("good", [k for k, r in rows.items() if float(r["r_1e-3"]) <= 0.01])]:
    sample = random.sample(sel, min(n, len(sel))); before, after = [], []
    for k in sample:
        s = Structure.from_dict(recs[k]["material"]); t = full_tensor(target, recs[k][field])
        rot = Tensor(np.zeros((3, 3))).get_ieee_rotation(s)
        s2 = s.copy(); s2.apply_operation(SymmOp.from_rotation_and_translation(rot, [0, 0, 0]))
        ops = SpacegroupAnalyzer(s2, symprec=1e-3).get_point_group_operations(cartesian=True)
        before.append(float(rows[k]["r_1e-3"])); after.append(residual(t, ops))
    after = np.array(after)
    print(f"{label} sample n={len(sample)}: before median r {np.median(before):.2e}; after IEEE rotation: median r {np.median(after):.2e}, fraction r>0.01: {np.mean(after > 0.01):.2f}")
