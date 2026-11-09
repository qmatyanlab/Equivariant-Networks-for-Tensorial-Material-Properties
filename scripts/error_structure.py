"""Structure of the test errors of the models trained from scratch (Results 3.2 and Fig. 2(d)).

For each target and architecture: the share of the summed per-material error carried by the 5% and by the 10 test
materials with the largest errors, the Pearson correlation of the per-material MAE with the mean absolute DFT
component, and, per irreducible component, the MAE of the harmonic coefficients relative to their mean absolute DFT
value (1 = a prediction of zero). Per-material MAE over the independent Cartesian components, as in analyze.py.

Usage: python error_structure.py <npz dir from eval_dump.py>   (imports analyze.py from this folder)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import IRREP_BLOCKS, per_material_mae  # noqa: E402

npz = Path(sys.argv[1])
for target, key in [("elastic", "Elastic"), ("dielectric", "Dielectric"), ("piezoelectric", "Piezo")]:
    for arch in ["SEGNN", "SEConv", "SETransformer"]:
        z = np.load(npz / f"R2026_{arch}_-_{key}_-_Default.npz")
        e = per_material_mae(target, z["pred"], z["true"])
        scale = per_material_mae(target, np.zeros_like(z["true"]), z["true"])  # mean |DFT component| per material
        s = np.sort(e)[::-1]
        k = int(round(0.05 * len(e)))
        irr = []
        for lab, i0, i1 in IRREP_BLOCKS[target]:
            err = np.abs(z["pred_coef"][:, i0:i1] - z["true_coef"][:, i0:i1]).mean()
            irr.append(f"{lab} {err / np.abs(z['true_coef'][:, i0:i1]).mean():.2f}")
        print(f"{target:13s} {arch:13s} top 5%: {100 * s[:k].sum() / s.sum():.0f}%  top 10: {100 * s[:10].sum() / s.sum():.0f}%  "
              f"Pearson(MAE, |DFT|): {np.corrcoef(e, scale)[0, 1]:.2f}  | MAE/mean|coef|: " + ", ".join(irr))
