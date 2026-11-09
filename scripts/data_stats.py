"""Counts per split and component-wise mean/std of the processed data sets (for SI Table S-data)."""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
for name, key in [("piezo", "piezo_tensor"), ("dielectric", None), ("elastic", None)]:
    d = root / name
    rows, n = [], {}
    for split in ["train", "validate", "test"]:
        f = d / split / "data.json"
        if not f.exists():
            continue
        data = json.load(open(f))
        n[split] = len(data)
        if key is None:
            key = next(k for k in data[0] if "tensor" in k or k in ("dielectric", "elasticity"))
        rows += [np.array(x[key], float) for x in data]
    if not rows:
        print(name, "no data"); continue
    A = np.stack(rows)
    mt = (d / "train" / "data.json").stat().st_mtime
    print(f"== {name} key={key} shape={A.shape[1:]} n={n} total={sum(n.values())} mtime={mt:.0f}")
    print("mean\n", np.array2string(A.mean(0), precision=4, suppress_small=False))
    print("std\n", np.array2string(A.std(0), precision=4))
    if A.ndim == 3 and A.shape[1] == A.shape[2]:
        print("max asymmetry", np.abs(A - A.transpose(0, 2, 1)).max())
