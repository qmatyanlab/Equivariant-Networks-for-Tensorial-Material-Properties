"""How do the dataset's target coefficients transform when the Cartesian tensor is rotated?
Compare c(R.T) with D(R) c(T) and D(P R P^T) c(T), using the irreps the model assigns to each head.
Run from repo src/Dataset (CPU, seconds)."""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from e3nn import o3
from e3nn.io import CartesianTensor
from pymatgen.core.tensors import Tensor
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.piezo import PiezoTensor
torch.manual_seed(3); rng = np.random.default_rng(3)
R = o3.rand_matrix().double().numpy()
P = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
laws = {"D(R)": R, "D(PRP^T)": P @ R @ P.T}
def rot(t, M): return Tensor(t).rotate(M, tol=1e-6)
def report(name, irreps, c0, c1):
    irr = o3.Irreps(irreps)
    for tag, M in laws.items():
        pred = irr.D_from_matrix(torch.tensor(M)).numpy() @ c0
        print(f"{name:28s} {irreps:24s} vs {tag:9s}: rel. err {np.linalg.norm(c1 - pred) / np.linalg.norm(c0):.2e}")
# dielectric, HEAD convention (e3nn CartesianTensor)
A = rng.standard_normal((3, 3)); E = A + A.T
ct = CartesianTensor("ij=ji")
f = lambda e: ct.from_cartesian(torch.tensor(e)).numpy()
report("dielectric (e3nn, HEAD)", "1x0e+1x2e", f(E), f(rot(E, R)))
# dielectric, 2024 convention
from dielectric_utils import cart2realsphvec
g = lambda e: np.asarray(cart2realsphvec(e), float).reshape(-1)
report("dielectric (2024 utils)", "1x0e+1x2e", g(E), g(rot(E, R)))
# elastic (2024 utils): sym 15 = 0e+2e+4e, asym 6 = 0e+2e
from elastic_utils import voigt2realsphvectors
C = ElasticTensor.from_voigt((lambda m: m + m.T)(rng.standard_normal((6, 6))))
def el(c):
    asym, sym = voigt2realsphvectors(np.asarray(c.voigt))
    return np.real(np.asarray(sym, complex)).reshape(-1), np.real(np.asarray(asym, complex)).reshape(-1)
(s0, a0), (s1, a1) = el(C), el(ElasticTensor(rot(C, R)))
report("elastic sym", "1x0e+1x2e+1x4e", s0, s1)
report("elastic asym", "1x0e+1x2e", a0, a1)
# piezo (HEAD utils), model head 1o+3o+1o+2e; also test 2o for the mixed l=2 part
from piezo_utils import real_harmonic_components_piezo_model_basis as real_harmonic_components_piezo
D = PiezoTensor.from_voigt(rng.standard_normal((3, 6)))
h = lambda d: np.real(np.asarray(real_harmonic_components_piezo(np.asarray(PiezoTensor(d).voigt)), complex)).reshape(-1)
p0, p1 = h(D), h(rot(D, R))
report("piezo (fixed)", "1x1o+1x3o+1x1o+1x2o", p0, p1)
# parity: inversion flips a polar rank-3 tensor; which label does each block follow?
pinv = h(-np.asarray(D))
print("piezo under inversion, sign of each block (a 1o, R 3o, b 1o, B l=2):",
      [int(np.sign(np.dot(pinv[s], p0[s]))) for s in [slice(0, 3), slice(3, 10), slice(10, 13), slice(13, 18)]])
# per-block rotation law for piezo
blocks = [("a (1o)", "1x1o", slice(0, 3)), ("R (3o)", "1x3o", slice(3, 10)), ("b (1o)", "1x1o", slice(10, 13)), ("B (l=2)", "1x2o", slice(13, 18))]
for name, irr, s in blocks:
    report("piezo block " + name, irr, p0[s], p1[s])

# round trip: Cartesian -> model-basis coefficients -> Cartesian
import sys; sys.path.insert(0, "..")
from RecombinePiezoelectricCoefficients import reconstruct_from_harmonic_vec
back = reconstruct_from_harmonic_vec(torch.tensor(p0, dtype=torch.float32).reshape(1, -1)).reshape(3, 6).numpy()
v = np.asarray(PiezoTensor(D).voigt)
print(f"piezo round trip rel. err: {np.linalg.norm(np.real(back) - v) / np.linalg.norm(v):.2e}")
