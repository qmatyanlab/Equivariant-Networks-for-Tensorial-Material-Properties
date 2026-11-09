"""Equivariant normalization statistics, written to <t>/coef_stats.json (run after filter_gap.py).

The targets are normalized as (c - mean) / std per harmonic coefficient (Dataset/Normalization.py). For the
de-normalized prediction std * y + mean to stay equivariant when the network output y is, all 2l+1 coefficients of
an irrep must share one scale, and only l = 0 coefficients may have a nonzero mean. So: std = RMS of the irrep block
over materials and components, mean = per-coefficient mean for 0e blocks and 0 otherwise. Statistics come from the
training split only. The earlier per-coefficient statistics (all splits) are kept as coef_stats_percomponent.json.

  python equivariant_stats.py [elastic dielectric piezo]"""
import json, shutil, sys
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parent
# (stats prefix, data key, irrep blocks (l, start, end)); same blocks as scripts/analyze.py IRREP_BLOCKS
SPEC = {"elastic": [("symmetric", "symmetric_coefs", [(0, 0, 1), (2, 1, 6), (4, 6, 15)]),
                    ("asymmetric", "asymmetric_coefs", [(0, 0, 1), (2, 1, 6)])],
        "dielectric": [("sph", "sph_coefs", [(0, 0, 1), (2, 1, 6)])],
        "piezo": [("sph", "sph_coefs", [(1, 0, 3), (3, 3, 10), (1, 10, 13), (2, 13, 18)])]}


def stats(train, spec):
    out = {}
    for prefix, key, blocks in spec:
        C = np.array([p[key] for p in train], float)
        mu, sd = np.zeros(C.shape[1]), np.zeros(C.shape[1])
        for l, a, b in blocks:
            if l == 0:
                mu[a:b] = C[:, a:b].mean(0)
            sd[a:b] = np.sqrt(((C[:, a:b] - mu[a:b]) ** 2).mean())
        out[f"{prefix}_mean"], out[f"{prefix}_std"] = mu.tolist(), sd.tolist()
    return out


if __name__ == "__main__":
    for t in sys.argv[1:] or list(SPEC):
        f = B / t / "coef_stats.json"
        old = B / t / "coef_stats_percomponent.json"
        if f.exists() and not old.exists():
            shutil.copy2(f, old)
        train = json.load(open(B / t / "train" / "data.json"))
        out = stats(train, SPEC[t])
        json.dump(out, open(f, "w"), indent=2)
        print(t, f"n_train {len(train)}", {k: np.round(v, 4).tolist() for k, v in out.items() if k.endswith("std")})
