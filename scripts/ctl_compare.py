"""Equal-learning-rate transfer comparison for SETransformer on the dielectric tensor.

The three SETransformer dielectric models initialized from a pretrained backbone use 5e-4 (5e-3 diverged), while the
model trained from scratch in Table 4 uses 5e-3. The control run R2026ctl_..._Default_LR5e-4 is trained from scratch at
5e-4. This compares each transfer model, and the 5e-3 scratch model, with the control: paired bootstrap on the
per-material MAE over the independent components (same function, resamples and p-value as analyze.py), Holm over the
four comparisons.

Usage: python3 scripts/ctl_compare.py analysis/eval/npz analysis/eval_ctl/npz analysis/eval_ctl/ctl_paired.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import per_material_mae, paired_test, holm  # noqa: E402

main_npz, ctl_npz, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
CTL = "R2026ctl_SETransformer_-_Dielectric_-_Default_LR5e-4"
RUNS = ["R2026_SETransformer_-_Dielectric_-_Pretrained", "R2026_SETransformer_-_Dielectric_-_Pretrained_-_Formation_Energy",
        "R2026_SETransformer_-_Dielectric_-_From_Elastic", "R2026_SETransformer_-_Dielectric_-_Default"]


def err(path):
    d = np.load(path)
    return d["ids"], per_material_mae("dielectric", d["pred"], d["true"])


ids_c, e_c = err(ctl_npz / f"{CTL}.npz")
rows = []
for k, name in enumerate(RUNS):
    ids, e = err(main_npz / f"{name}.npz")
    assert (ids == ids_c).all(), name
    rows.append([name, CTL, *paired_test(e, e_c, 20000, [0, 3, k])])
for r, ph in zip(rows, holm([r[5] for r in rows])):
    r.append(float(ph))
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_a", "run_b", "mean_diff_a_minus_b", "ci_lo", "ci_hi", "p_value", "p_holm"])
    w.writerows(rows)
print(f"control MAE {e_c.mean():.4f}")
for r in rows:
    print(f"{r[0]:70s} {r[2]:+.4f} [{r[3]:+.4f}, {r[4]:+.4f}] p={r[5]:.3g} p_holm={r[6]:.3g}")
