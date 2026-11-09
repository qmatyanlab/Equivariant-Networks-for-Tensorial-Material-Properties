"""Full-model equivariance test on a real checkpoint (CPU, seconds).
Rotate (and invert) a low-symmetry crystal; compare model(R x) with D(R) model(x) for each output head.
Tested twice: the raw network output, and (if the config normalizes) the de-normalized prediction std*y + mean with
Dataset/<t>/coef_stats.json, which is what the model predicts. Only the second is the equivariance of the model.
Run from repo src/: python equiv_test.py <target> <config> <checkpoint> [pooling_override]"""
import os, sys, json, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Lattice
from e3nn import o3
from torch_geometric.loader import DataLoader
target, cfg_path, ckpt = sys.argv[1:4]
mods = {"elastic": ("run_elastic", "ElasticRunner"), "dielectric": ("run_dielectric", "DielectricRunner"),
        "piezoelectric": ("run_piezoelectric", "PiezoelectricRunner")}
cls = getattr(__import__(mods[target][0]), mods[target][1])
cfg = json.load(open(cfg_path))
if len(sys.argv) > 4: cfg["pooling"] = sys.argv[4]
rng = np.random.default_rng(int(os.environ.get("EQUIV_SEED", 0)))  # EQUIV_SEED: another random test crystal
lat = Lattice.from_parameters(4.3, 5.1, 6.2, 81, 97, 104)   # triclinic, no symmetry
base = Structure(lat, ["Ba", "Ti", "O", "O", "O", "Zn"], rng.random((6, 3)))
if os.environ.get("EQUIV_TEST_ID"):  # use this test-set material instead of the random crystal
    _d = {"elastic": "elastic", "dielectric": "dielectric", "piezoelectric": "piezo"}[target]
    _ids = open(f"Dataset/{_d}/test/material_ids.csv").read().split("\n")
    base = Structure.from_dict(json.load(open(f"Dataset/{_d}/test/data.json"))[_ids.index(os.environ["EQUIV_TEST_ID"])]["material"])
def struct_rot(R):
    return Structure(Lattice(base.lattice.matrix @ R.T), base.species, base.frac_coords)
fields = {"elastic": {"elasticity_tensor": np.zeros((6, 6)).tolist(), "symmetric_coefs": [0.0]*15, "asymmetric_coefs": [0.0]*6},
          "dielectric": {"dielectric_tensor": np.zeros((3, 3)).tolist(), "sph_coefs": [0.0]*6},
          "piezoelectric": {"piezo_tensor": np.zeros((3, 6)).tolist(), "sph_coefs": [0.0]*18}}[target]
ai = json.load(open("Dataset/atom_init.json"))
r = object.__new__(cls)
torch.manual_seed(1); R = o3.rand_matrix().double()
mats = {"identity": torch.eye(3, dtype=torch.float64), "rotation": R, "inversion": -torch.eye(3, dtype=torch.float64),
        "rotoinversion": -R}
data = [{"material": struct_rot(m.numpy()).as_dict(), **fields} for m in mats.values()]
ds = r.dataset_class(data, ai, cfg["lmax_h"], radius=cfg["radius"], max_neighbors=cfg["max_neighbors"], normalize=False)
r.setup_model(cfg, ds[0].x.shape[-1])
if ckpt != "none":  # "none": test a freshly initialized model
    r.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
m = r.model.eval()
with torch.inference_mode():
    outs = m(next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False))))
from Dataset.Normalization import Normalizer
results = {"network output": [y.double() for y in outs]}
if cfg["normalize"]:
    stats = {"elastic": "elastic", "dielectric": "dielectric", "piezoelectric": "piezo"}[target]
    r.normalizer = Normalizer(json.load(open(f"Dataset/{stats}/coef_stats.json")), "cpu")
    den = r.unnormalize(*results["network output"])
    results["prediction"] = [den.double()] if torch.is_tensor(den) else [y.double() for y in den]
P = torch.tensor([[0., 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)  # xyz -> yzx, as in the dataset
for kind, ys in results.items():
    for h, (irr, y) in enumerate(zip(r.output_irreps_lst, ys)):
        for i, (name, Q) in enumerate(mats.items()):
            if i == 0: continue
            for tag, QQ in [("D(R)", Q), ("D(P R P^T)", P @ Q @ P.T)]:
                pred = irr.D_from_matrix(QQ) @ y[0]
                err = (y[i] - pred).norm() / y[0].norm()
                print(f"head {h} ({irr}) [{kind}] {name} vs {tag}: rel. error = {err:.2e}")
