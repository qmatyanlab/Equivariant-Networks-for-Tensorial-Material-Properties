"""Overlap of the downstream test sets with the sets used to train the transferred backbones (SI data section).

  python overlap.py <rebuilt Dataset dir> <2024 Dataset dir> <out.json>
Rebuilt splits are used where they exist (material_ids.csv written by generate_*_data.py); the 2024 splits otherwise.
"""
import json, sys
from pathlib import Path
new, old, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

def ids(root, t, sp):
    f = root / t / sp / "material_ids.csv"
    return {m.strip() for m in f.read_text().split("\n") if m.strip()} if f.exists() else None

def split(t, sp, prefer_new=True):
    s = ids(new, t, sp) if prefer_new else None
    return (s, "rebuilt") if s is not None and len(s) else (ids(old, t, sp), "2024")

res = {"sizes": {}, "overlap": []}
src = {}
for t in ["piezo", "dielectric", "elastic", "bandgap", "formation"]:
    for sp in ["train", "validate", "test"]:
        s, where = split(t, sp, prefer_new=t in ("piezo", "dielectric", "elastic"))
        src[(t, sp)] = s
        res["sizes"][f"{t}/{sp}"] = [len(s), where]
# Tensor backbones are trained on train + validate (no --validation flag). The scalar pretrainers (Pretrainer.py)
# train on all three splits, so for them the whole data set counts.
train_val = {t: src[(t, "train")] | src[(t, "validate")] for t in ["piezo", "dielectric", "elastic"]}
for t in ["bandgap", "formation"]:
    train_val[t] = src[(t, "train")] | src[(t, "validate")] | src[(t, "test")]
for down in ["piezo", "dielectric", "elastic"]:
    test = src[(down, "test")]
    for up in ["bandgap", "formation", "elastic", "dielectric", "piezo"]:
        if up == down:
            continue
        n = len(test & train_val[up])
        res["overlap"].append({"downstream_test": down, "upstream_train_val": up, "n_test": len(test), "n_overlap": n,
                               "fraction": round(n / len(test), 4)})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
