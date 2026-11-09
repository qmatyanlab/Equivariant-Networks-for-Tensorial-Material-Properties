"""Transformation law of the original (2024/HEAD) piezo target coefficients, block by block (SI equivariance note)."""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from e3nn import o3
from pymatgen.core.tensors import Tensor
from pymatgen.analysis.piezo import PiezoTensor
from piezo_utils import real_harmonic_components_piezo as old
torch.manual_seed(3); rng = np.random.default_rng(3)
P = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
D = PiezoTensor.from_voigt(rng.standard_normal((3, 6)))
h = lambda d: np.real(np.asarray(old(np.asarray(PiezoTensor(d).voigt)), complex)).reshape(-1)
errs = {k: [] for k in ["a (1o)", "W (3o)", "b (1o)", "B (2o)", "B as 2e under inversion"]}
for trial in range(20):
    R = o3.rand_matrix().double().numpy(); M = P @ R @ P.T
    p0, p1 = h(D), h(Tensor(D).rotate(R, tol=1e-6))
    for name, irr, s in [("a (1o)", "1x1o", slice(0, 3)), ("W (3o)", "1x3o", slice(3, 10)), ("b (1o)", "1x1o", slice(10, 13)), ("B (2o)", "1x2o", slice(13, 18))]:
        pred = o3.Irreps(irr).D_from_matrix(torch.tensor(M)).numpy() @ p0[s]
        errs[name].append(np.linalg.norm(p1[s] - pred) / np.linalg.norm(p0[s]))
p0, pi = h(D), h(-np.asarray(D))
errs["B as 2e under inversion"].append(np.linalg.norm(pi[13:18] - p0[13:18]) / np.linalg.norm(p0[13:18]))
for k, v in errs.items():
    print(f"{k:26s} median {np.median(v):.2e}  max {np.max(v):.2e}  (n={len(v)})")
