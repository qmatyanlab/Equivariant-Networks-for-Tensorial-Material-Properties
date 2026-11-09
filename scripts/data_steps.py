"""Replays the data pipeline of generate_<target>_data.py from the cached MP records, read-only, and records the
outcome of every step for every record (SI data-processing section). Writes nothing into the Dataset folders.

  cd code/src/Dataset && python data_steps.py <target> <out_dir> <dir with the 2024 split IDs (code2024/src/Dataset)>
"""
import csv, importlib, json, sys, warnings
from pathlib import Path
import numpy as np
import os
sys.path.insert(0, os.getcwd())
from pymatgen.core import Structure
import frame_align as fa

warnings.filterwarnings("ignore")
target, out, ids2024 = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
mod = importlib.import_module({"piezo": "generate_piezo_data", "dielectric": "generate_dielectric_data",
                               "elastic": "generate_elastic_data"}[target])
key = {"piezo": "electric_contribution", "dielectric": "electric_contribution", "elastic": "elasticity_tensor"}[target]
raw = json.load(open(f"{target}/raw_data.json"))
split24 = {}
for sp in ["train", "validate", "test"]:
    for m in (ids2024 / target / sp / "material_ids.csv").read_text().split("\n"):
        if m:
            split24[m] = sp
counts = {"cached_records": len(raw)}
seen, data = set(), []
for d in raw:
    if d["material_id"] not in seen:
        seen.add(d["material_id"]); data.append(d)
counts["unique"] = len(data)
counts["duplicates_removed"] = counts["cached_records"] - counts["unique"]
rows, kept = [], []
for d in data:
    row = {"material_id": d["material_id"], "r_original": "", "r_ieee": "", "frame": "", "pos_def": "", "pruned": "",
           "split2024": split24.get(d["material_id"], "")}
    try:
        s = Structure.from_dict(d["material"]); t = fa.full_tensor(target, d[key])
        r0 = fa.symmetry_residual(t, s); row["r_original"] = r0
        best = ("original", r0, s)
        if r0 > fa.TOL:
            s1 = fa.ieee_rotated(s); r1 = fa.symmetry_residual(t, s1); row["r_ieee"] = r1
            if r1 < r0:
                best = ("ieee", r1, s1)
    except Exception:
        row["frame"] = "error"; rows.append(row); continue
    row["frame"] = best[0] if best[1] <= fa.TOL else "dropped"
    row["r_final"] = best[1]
    if row["frame"] != "dropped":
        kept.append((row, dict(d, material=best[2].as_dict())))
    rows.append(row)
for k in ["original", "ieee", "dropped", "error"]:
    counts[f"frame_{k}"] = sum(r["frame"] == k for r in rows)
recs = [d for _, d in kept]
if target == "elastic":
    ok = mod.remove_impossibilities(recs)
    okid = {d["material_id"] for d in ok}
    for r, d in kept:
        r["pos_def"] = d["material_id"] in okid
    counts["not_positive_definite"] = len(recs) - len(ok)
    recs = ok
mod.compute_spherical_harmonic_coefficients(recs)
pr = mod.prune_outliers(recs)
prid = {d["material_id"] for d in pr}
for r, d in kept:
    if target != "elastic" or r["pos_def"]:
        r["pruned"] = d["material_id"] not in prid
counts["outliers_pruned"] = len(recs) - len(pr)
counts["usable"] = len(pr)
counts["in_2024_splits"] = sum(1 for d in pr if d["material_id"] in split24)
counts["not_in_2024_splits"] = counts["usable"] - counts["in_2024_splits"]
counts["2024_split_size"] = len(split24)
for sp in ["train", "validate", "test"]:
    counts[f"final_{sp}"] = sum(1 for d in pr if split24.get(d["material_id"]) == sp)
r0 = np.array([r["r_original"] for r in rows if r["r_original"] != ""], float)
rf = np.array([r["r_final"] for r in rows if r.get("frame") in ("original", "ieee")], float)
counts["frac_r_original_gt_tol"] = float((r0 > fa.TOL).mean())
counts["median_r_original_of_violators"] = float(np.median(r0[r0 > fa.TOL])) if (r0 > fa.TOL).any() else 0.0
counts["max_r_final_kept"] = float(rf.max())
counts["median_r_final_kept"] = float(np.median(rf))
out.mkdir(parents=True, exist_ok=True)
json.dump(counts, open(out / f"steps_{target}.json", "w"), indent=1)
with open(out / f"steps_{target}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["material_id", "r_original", "r_ieee", "frame", "r_final", "pos_def", "pruned", "split2024"])
    w.writeheader(); w.writerows(rows)
print(json.dumps(counts, indent=1))
