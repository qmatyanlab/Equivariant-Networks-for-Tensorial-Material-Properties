"""Evaluate trained checkpoints on a test set and dump per-material predictions (GPU job on neu).

Run from the repo's `src/` directory (branch revision-2026-11), so the conversions match the training code.

Usage:
  python eval_dump.py <target> <outdir> <run_list.txt> [--probe <probe_target>]
    target: elastic | dielectric | piezoelectric
    run_list.txt: lines "<name> <config> <checkpoint>" (a runs_X.txt line also works: "<target> <name> <config> <pre>",
                  in which case the checkpoint is trained_models/<name>)
    --probe: also dump backbone embeddings on that target's test structures (the common set for CKA)

For each run, writes <outdir>/<name>.npz with
  ids         material IDs of the test set (order of test/material_ids.csv)
  pred, true  Cartesian tensors: elastic 6x6 Voigt (GPa), dielectric 3x3, piezoelectric 3x6 Voigt (C/m^2)
  pred_coef, true_coef  de-normalized harmonic coefficients (elastic: 15 symmetric + 6 mixed; dielectric 6; piezo 18)
  allowed     boolean mask of the components allowed by each structure's point group, in that structure's own frame
  emb         mean-pooled scalar (0e) channels of the backbone output, one row per test material
  probe_ids, probe_emb  the same embedding on the probe target's test structures (with --probe)
"""
import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from Dataset.Normalization import Normalizer

warnings.filterwarnings("ignore")

TARGETS = {
    "elastic": ("run_elastic", "ElasticRunner", "Dataset/elastic"),
    "dielectric": ("run_dielectric", "DielectricRunner", "Dataset/dielectric"),
    "piezoelectric": ("run_piezoelectric", "PiezoelectricRunner", "Dataset/piezo"),
}
KIND = {"elastic": "elastic", "dielectric": "dielectric", "piezoelectric": "piezo"}


def runner_class(target):
    mod, cls, _ = TARGETS[target]
    return getattr(__import__(mod), cls)


def load_split(target, split="test"):
    path = Path(TARGETS[target][2]) / split
    ids = path.joinpath("material_ids.csv").read_text().split("\n")
    data = json.loads(path.joinpath("data.json").read_text())
    assert len(ids) == len(data), (target, len(ids), len(data))
    return ids, data


def make_dataset(r, data, cfg, atom_init):
    # the dataset converts "material" dicts to Structures in place, so give it its own copy
    return r.dataset_class(copy.deepcopy(data), atom_init, cfg["lmax_h"], radius=cfg["radius"],
                           max_neighbors=cfg["max_neighbors"], normalize=cfg["normalize"])


def allowed_mask(target, structure_dict, rng):
    """Components left nonzero by the point-group projector of this structure, in its own frame."""
    from pymatgen.core import Structure
    from Dataset.frame_align import full_tensor, SYMPREC
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    s = Structure.from_dict(structure_dict)
    ops = SpacegroupAnalyzer(s, symprec=SYMPREC).get_point_group_operations(cartesian=True)
    shape = {"elastic": (6, 6), "dielectric": (3, 3), "piezoelectric": (3, 6)}[target]
    out = np.zeros(shape, bool)
    for _ in range(2):  # two random tensors, so that an accidental zero does not hide an allowed component
        a = rng.standard_normal(shape)
        if target != "piezoelectric":
            a = a + a.T
        t = full_tensor(KIND[target], a)
        proj = np.mean([t.transform(op) for op in ops], axis=0)
        p = t.__class__(proj)
        v = np.asarray(p.voigt if target != "dielectric" else p)
        out |= np.abs(v) > 1e-6 * np.abs(v).max()
    return out


def predict(target, r, out, data):
    """Returns (pred Cartesian, true Cartesian, pred coef, true coef), all de-normalized."""
    if target == "elastic":
        from RecombineElasticCoefficients import realsphvectors2voigt
        sym, asym = out
        tsym, tasym = data.symmetric_coefs, data.asymmetric_coefs
        if r.config["normalize"]:
            sym, asym = r.unnormalize(sym, asym)
            tsym, tasym = r.unnormalize(tsym, tasym)
        pred = realsphvectors2voigt(asym, sym).reshape(-1, 6, 6)
        return pred, data.voigt_tensor.reshape(-1, 6, 6), torch.cat([sym, asym], -1), torch.cat([tsym, tasym], -1)
    sph, tsph = out[0], data.sph_coefs.reshape(out[0].shape)
    if r.config["normalize"]:
        sph, tsph = r.unnormalize(sph), r.unnormalize(tsph)
    if target == "dielectric":
        from RecombineDielectricCoefficients import realsphvec2cart
        return realsphvec2cart(sph).reshape(-1, 3, 3), data.dielectric_tensor.reshape(-1, 3, 3), sph, tsph
    from RecombinePiezoelectricCoefficients import reconstruct_from_harmonic_vec
    return reconstruct_from_harmonic_vec(sph).reshape(-1, 3, 6), data.piezoelectric_tensor.reshape(-1, 3, 6), sph, tsph


def embed(model, loader, device, n_scalar):
    feats = {}
    h = model.graph_nn.register_forward_hook(lambda _m, _i, o: feats.__setitem__("x", o))
    embs = []
    with torch.inference_mode():
        for d in loader:
            d = d.to(device)
            model(d)
            embs.append(global_mean_pool(feats["x"][:, :n_scalar], d.batch).cpu().numpy())
    h.remove()
    return np.concatenate(embs)


def parse_runs(path):
    for line in Path(path).read_text().split("\n"):
        f = line.split()
        if len(f) == 3:
            yield f[0], f[1], f[2]
        elif len(f) == 4:  # a runs_X.txt line
            yield f[1], f[2], f"trained_models/{f[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=list(TARGETS))
    ap.add_argument("outdir")
    ap.add_argument("run_list")
    ap.add_argument("--probe", choices=list(TARGETS))
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atom_init = json.loads(Path("Dataset/atom_init.json").read_text())
    cls = runner_class(a.target)
    ids, data = load_split(a.target)
    probe = load_split(a.probe) if a.probe else None
    # probe structures are read with the probe target's dataset class (its records carry that target's labels);
    # only the graphs enter the embedding
    probe_runner = object.__new__(runner_class(a.probe)) if a.probe else None
    stats = json.loads(Path(TARGETS[a.target][2], "coef_stats.json").read_text())
    rng = np.random.default_rng(0)
    allowed = np.stack([allowed_mask(a.target, d["material"], rng) for d in data])
    from Model.BalancedIrreps import BalancedIrreps
    cache = {}
    for name, cfg_path, ckpt in parse_runs(a.run_list):
        if not Path(ckpt).exists():
            print(f"{name}: checkpoint {ckpt} missing, skipped", flush=True)
            continue
        cfg = {**json.loads(Path(cfg_path).read_text()), "validation": False}
        r = object.__new__(cls)
        r.config, r.device = cfg, device
        r.normalizer = Normalizer(stats, device)
        key = (cfg["lmax_h"], cfg["radius"], cfg["max_neighbors"], cfg["normalize"])
        if key not in cache:  # graphs depend only on these settings
            cache[key] = (make_dataset(r, data, cfg, atom_init),
                          make_dataset(probe_runner, probe[1], cfg, atom_init) if probe else None)
        ds, pds = cache[key]
        r.setup_model(cfg, ds[0].x.shape[-1])
        r.model.load_state_dict(torch.load(ckpt, map_location=device))
        model = r.model.to(device).eval()
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
        cols = {k: [] for k in ["pred", "true", "pred_coef", "true_coef"]}
        with torch.inference_mode():
            for d in loader:
                d = d.to(device)
                for k, v in zip(cols, predict(a.target, r, model(d), d)):
                    cols[k].append(v.reshape(d.num_graphs, *v.shape[1:]).real.float().cpu().numpy()
                                   if torch.is_complex(v) else v.reshape(d.num_graphs, *v.shape[1:]).float().cpu().numpy())
        n_scalar = BalancedIrreps(cfg["lmax_h"], cfg["hidden_features"])[0].mul
        res = {k: np.concatenate(v) for k, v in cols.items()}
        res.update(ids=np.array(ids), allowed=allowed, emb=embed(model, loader, device, n_scalar))
        if pds is not None:
            res["probe_ids"] = np.array(probe[0])
            res["probe_emb"] = embed(model, DataLoader(pds, batch_size=cfg["batch_size"], shuffle=False), device, n_scalar)
        np.savez_compressed(out_dir / f"{name}.npz", **res)
        mae = np.abs(res["pred"] - res["true"]).mean()
        print(f"{name}: n={len(ids)} full-tensor MAE={mae:.4g}", flush=True)


if __name__ == "__main__":
    main()
