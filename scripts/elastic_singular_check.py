"""Near-singular / non-positive-definite predicted elasticity tensors and their effect on the derived moduli.

Run on neu from jobs/ (imports analyze.moduli): env/bin/python scripts/elastic_singular_check.py analysis/eval/npz
Positive definiteness is checked on the Mandel matrix W C W (W = diag(1,1,1,sqrt2,sqrt2,sqrt2)), which is congruent to
the Voigt matrix, so the sign pattern is the same and the eigenvalues are those of the stiffness as a 4th-order tensor.
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import moduli  # noqa: E402

W = np.diag([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])


def lam_min(C):
    M = W @ ((C + np.transpose(C, (0, 2, 1))) / 2) @ W
    return np.linalg.eigvalsh(M)[:, 0]


for f in sorted(glob.glob(str(Path(sys.argv[1]) / "R2026_*_-_Elastic_-_*.npz"))):
    z = np.load(f)
    P, T = z["pred"].astype(float), z["true"].astype(float)
    lp, lt = lam_min(P), lam_min(T)
    mp, mt = moduli(P), moduli(T)
    # Voigt-only moduli (no inversion) for comparison
    def voigt(C):
        KV = (C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2] + 2 * (C[:, 0, 1] + C[:, 1, 2] + C[:, 2, 0])) / 9
        GV = (C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2] - (C[:, 0, 1] + C[:, 1, 2] + C[:, 2, 0])
              + 3 * (C[:, 3, 3] + C[:, 4, 4] + C[:, 5, 5])) / 15
        return KV, GV
    KVp, GVp = voigt(P); KVt, GVt = voigt(T)
    bad = lp <= 0
    near = lp < 0.05 * lt  # predicted smallest eigenvalue below 5% of the DFT one
    name = Path(f).stem[6:]
    print(f"== {name}  n={len(P)}  true non-PD: {(lt <= 0).sum()}  pred non-PD: {bad.sum()}  "
          f"pred lam_min < 5% of DFT: {near.sum()}")
    for k in ("K_H", "G_H", "E"):
        e = np.abs(mp[k] - mt[k])
        order = np.argsort(e)[::-1]
        top5 = e[order[:5]]
        print(f"   {k}: MAE all {np.nanmean(e):.2f} | excl. non-PD {np.nanmean(e[~bad]):.2f} | excl. near {np.nanmean(e[~near]):.2f}"
              f" | median {np.nanmedian(e):.2f} | top-5 errors {np.round(top5, 0).tolist()} (lam_min pred/DFT of top-1:"
              f" {lp[order[0]]:.2f}/{lt[order[0]]:.2f}) | share of top 1% {100 * e[order[:len(e) // 100]].sum() / np.nansum(e):.0f}%")
    print(f"   Voigt only: K_V MAE {np.abs(KVp - KVt).mean():.2f}, G_V MAE {np.abs(GVp - GVt).mean():.2f}")
