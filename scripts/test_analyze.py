"""Synthetic tests for analyze.py (run on neu with the project env)."""
import sys, subprocess, tempfile, json, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import analyze as A

# 1. isotropic elasticity: lambda=60, mu=40 GPa -> K = lambda + 2mu/3 = 86.667, G = 40, E = mu(3l+2mu)/(l+mu) = 104
lam, mu = 60.0, 40.0
C = np.zeros((6, 6)); C[:3, :3] = lam; C[[0, 1, 2], [0, 1, 2]] = lam + 2 * mu; C[[3, 4, 5], [3, 4, 5]] = mu
m = A.moduli(C[None])
assert np.allclose([m["K_H"][0], m["G_H"][0], m["E"][0]], [lam + 2 * mu / 3, mu, mu * (3 * lam + 2 * mu) / (lam + mu)]), m
print("moduli ok:", {k: round(float(v[0]), 3) for k, v in m.items()})

# 2. CKA: identical -> 1, rotated copy -> 1, independent -> small
rng = np.random.default_rng(1); X = rng.standard_normal((300, 40))
Q, _ = np.linalg.qr(rng.standard_normal((40, 40)))
assert abs(A.linear_cka(X, X) - 1) < 1e-9 and abs(A.linear_cka(X, X @ Q) - 1) < 1e-9
print("cka ok: independent =", round(A.linear_cka(X, rng.standard_normal((300, 40))), 3))

# 2b. Holm: textbook example (m = 4)
ph = A.holm([0.01, 0.04, 0.03, 0.005])
assert np.allclose(ph, [0.03, 0.06, 0.06, 0.02]), ph
assert np.allclose(A.holm([0.5, 0.0]), [0.5, 0.0]) and np.allclose(A.holm([0.6, 0.6, 0.6]), [1.0, 1.0, 1.0])
print("holm ok:", ph)

# 3. unique components
assert A.unique_components("elastic", np.zeros((2, 6, 6))).shape == (2, 21)
assert A.unique_components("dielectric", np.zeros((2, 3, 3))).shape == (2, 6)
assert A.unique_components("piezoelectric", np.zeros((2, 3, 6))).shape == (2, 18)

# 3b. Frobenius norm from the stored Voigt forms equals the norm of the explicitly built Cartesian tensor
VP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
def full_elastic(Cv):
    F = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VP):
        for J, (k, l) in enumerate(VP):
            for (a1, b1) in {(i, j), (j, i)}:
                for (c1, d1) in {(k, l), (l, k)}:
                    F[a1, b1, c1, d1] = Cv[I, J]
    return F
def full_piezo(ev):
    F = np.zeros((3, 3, 3))
    for k in range(3):
        for J, (i, j) in enumerate(VP):
            F[k, i, j] = F[k, j, i] = ev[k, J]
    return F
Cv = rng.standard_normal((6, 6)); Cv = Cv + Cv.T
ev = rng.standard_normal((3, 6))
assert np.isclose(A.frobenius("elastic", Cv[None])[0], np.linalg.norm(full_elastic(Cv)))
assert np.isclose(A.frobenius("piezoelectric", ev[None])[0], np.linalg.norm(full_piezo(ev)))
print("frobenius ok")

# 3c. dielectric reference is symmetrized on load: an asymmetric stored tensor and a symmetric prediction of its
# symmetric part give zero error (tested end-to-end below)

# 4. end-to-end on synthetic runs: model B is worse than A by construction
with tempfile.TemporaryDirectory() as t:
    t = Path(t); n = 400
    ids = np.array([f"mp-{i}" for i in range(n)])
    def write(name, shape, ncoef, noise, true):
        pred = true + noise * rng.standard_normal(true.shape)
        np.savez(t / f"{name}.npz", ids=ids, pred=pred, true=true, pred_coef=rng.standard_normal((n, ncoef)),
                 true_coef=rng.standard_normal((n, ncoef)), allowed=np.ones((n,) + shape, bool),
                 emb=rng.standard_normal((n, 40)), probe_ids=ids[:100], probe_emb=rng.standard_normal((100, 40)))
    Ct = np.tile(C, (n, 1, 1)) * (1 + 0.1 * rng.random((n, 1, 1)))
    Ct = (Ct + Ct.transpose(0, 2, 1)) / 2
    write("R2026_SEConv_-_Elastic_-_Default", (6, 6), 21, 1.0, Ct)
    write("R2026_SEConv_-_Elastic_-_Pretrained", (6, 6), 21, 3.0, Ct)
    # a model that overestimates every component by 5%: all relative errors are exactly 0.05, and so are the moduli's
    np.savez(t / "R2026_SEGNN_-_Elastic_-_Default.npz", ids=ids, pred=1.05 * Ct, true=Ct,
             pred_coef=np.zeros((n, 21)), true_coef=np.zeros((n, 21)), allowed=np.ones((n, 6, 6), bool))
    Dt = rng.standard_normal((n, 3, 3)); Dt[:, 0, 1] += 0.5  # stored tensor with an asymmetric part
    Ds = (Dt + Dt.transpose(0, 2, 1)) / 2
    np.savez(t / "R2026_SEGNN_-_Dielectric_-_Default.npz", ids=ids, pred=Ds, true=Dt, pred_coef=np.zeros((n, 6)),
             true_coef=np.zeros((n, 6)), allowed=np.ones((n, 3, 3), bool))
    # piezo runs on a test set of a different size: paired tests must bootstrap each pair with its own target's
    # indices (regression for eval 60839, where the last target's indices were used for every pair)
    n2 = 150
    ids2 = np.array([f"mp-p{i}" for i in range(n2)])
    pz = rng.standard_normal((n2, 3, 6)) * 0.3
    for name, noise in [("R2026_SEGNN_-_Piezo_-_Default", 0.05), ("R2026_SEConv_-_Piezo_-_Default", 0.08)]:
        np.savez(t / f"{name}.npz", ids=ids2, pred=pz + noise * rng.standard_normal(pz.shape), true=pz,
                 pred_coef=rng.standard_normal((n2, 18)), true_coef=rng.standard_normal((n2, 18)),
                 allowed=np.ones((n2, 3, 6), bool), emb=rng.standard_normal((n2, 40)), probe_ids=ids[:100],
                 probe_emb=rng.standard_normal((100, 40)))
    o = t / "out"
    subprocess.run([sys.executable, str(Path(__file__).parent / "analyze.py"), str(t), str(o), "--n-boot", "500"], check=True)
    rows = {r["run"]: r for r in csv.DictReader(open(o / "metrics.csv"))}
    pa = list(csv.DictReader(open(o / "paired.csv")))
    print("metrics:", {k: round(float(v["MAE"]), 3) for k, v in rows.items()})
    print("paired:", [(p["comparison"], round(float(p["mean_diff_a_minus_b"]), 3), p["p_value"]) for p in pa])
    assert float(rows["R2026_SEConv_-_Elastic_-_Pretrained"]["MAE"]) > float(rows["R2026_SEConv_-_Elastic_-_Default"]["MAE"])
    assert float(pa[0]["p_value"]) < 0.01
    assert all(float(p["p_holm"]) >= float(p["p_value"]) > 0 for p in pa)  # p is never reported as zero
    assert abs(float(rows["R2026_SEGNN_-_Piezo_-_Default"]["MAE"]) - 0.05 * np.sqrt(2 / np.pi)) < 0.005
    r5 = rows["R2026_SEGNN_-_Elastic_-_Default"]
    for k in ["nMAE", "rel_median", "rel_pooled", "rel_q25", "rel_q75"]:
        assert abs(float(r5[k]) - 0.05) < 1e-9, (k, r5[k])
    mo = {r["run"]: r for r in csv.DictReader(open(o / "moduli.csv"))}["R2026_SEGNN_-_Elastic_-_Default"]
    for k in ["K_H", "G_H", "E"]:
        assert abs(float(mo[k + "_MAPE"]) - 0.05) < 1e-9 and abs(float(mo[k + "_MdAPE"]) - 0.05) < 1e-9, mo
        assert float(mo[k + "_within10"]) == 1.0 and float(mo[k + "_within20"]) == 1.0
    assert int(mo["n_nonPD"]) == 0 and abs(float(mo["K_H_all"]) - float(mo["K_H"])) < 1e-12
    # a non-positive-definite tensor is detected (and would be excluded from the moduli statistics)
    bad = np.eye(6)[None].repeat(2, 0); bad[1, 0, 0] = -1.0
    assert A.positive_definite(bad).tolist() == [True, False]
    print("positive-definite filter ok")
    pm = list(csv.DictReader(open(o / "paired_moduli.csv")))
    assert len(pm) == 3 * sum(p["target"] == "elastic" for p in pa) and all(float(p["p_holm"]) > 0 for p in pm)
    print("paired moduli ok:", len(pm))
    rp = rows["R2026_SEGNN_-_Piezo_-_Default"]
    assert 0 < int(rp["rel_n"]) <= n2 and float(rp["rel_pooled"]) > 0
    assert any(p["target"] == "piezoelectric" for p in pa) and any(p["target"] == "elastic" for p in pa)
    rd = rows["R2026_SEGNN_-_Dielectric_-_Default"]
    assert float(rd["MAE"]) < 1e-12 and float(rd["rel_pooled"]) < 1e-12, rd["MAE"]
    print("dielectric symmetric reference ok")
    re_rows = list(csv.DictReader(open(o / "relerr.csv")))
    assert len(re_rows) == 4 * n + 2 * n2
    print("relative errors ok:", {k: round(float(rp[k]), 4) for k in ["nMAE", "rel_median", "rel_pooled", "rel_n"]})
    print(open(o / "tables.tex").read())
    print(json.loads(open(o / "similarity.json").read())["error_correlation"][:2])
print("ALL TESTS PASSED")
