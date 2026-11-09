"""BatchNorm diagnostic for the trained dielectric (and one piezoelectric) models (CPU job on neu, run from code/src).

Question: are the large epoch-to-epoch swings of the dielectric validation/test MAE caused by the running statistics of
the e3nn BatchNorm layers (used at evaluation time), or by a few materials with large epsilon?

For each checkpoint, the test-set full-tensor MAE (Cartesian components, same as eval_dump.py) is computed with
  eval        standard evaluation: BatchNorm uses the running statistics stored in the checkpoint (what we report)
  batch       BatchNorm uses the statistics of each test batch (train-mode BatchNorm, batch size of the config,
              test set in file order); the model weights are unchanged
  recomputed  running statistics recomputed as the exact average over all batches of the training data of the final
              runs (train + validate), in a fixed shuffled order, then standard evaluation
and, for each mode, the share of the total absolute error carried by the 10 test materials with the largest error and
the MAE without them. Writes <out>/bn_check.json and prints a table.

Usage: python bn_check.py <out_dir> <target> <name> [<name> ...]   (checkpoints trained_models/<name>, config from the
run lists in jobs/)
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from e3nn.nn import BatchNorm

import eval_dump as ed
from Dataset.Normalization import Normalizer

T = Path("/shared/home/qimin/tensor_response")


def config_of(name):
    for f in ["runs_A.txt", "runs_B.txt", "runs_C.txt", "runs_SETd.txt"]:
        for line in (T / "jobs" / f).read_text().split("\n"):
            p = line.split()
            if len(p) == 4 and p[1] == name:
                return p[2]
    raise KeyError(name)


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, BatchNorm) and not m.instance]


def predict_all(target, r, model, loader, device):
    pred, true = [], []
    with torch.inference_mode():
        for d in loader:
            d = d.to(device)
            p, t, _, _ = ed.predict(target, r, model(d), d)
            pred.append(p.real.float().cpu().numpy() if torch.is_complex(p) else p.float().cpu().numpy())
            true.append(t.float().cpu().numpy())
    return np.concatenate(pred), np.concatenate(true)


def summary(pred, true):
    err = np.abs(pred - true).reshape(len(pred), -1).mean(1)  # per-material MAE over components
    top = np.argsort(err)[::-1][:10]
    keep = np.setdiff1d(np.arange(len(err)), top)
    return {"MAE": float(err.mean()), "top10_share": float(err[top].sum() / err.sum()),
            "MAE_without_top10": float(err[keep].mean()), "top10_idx": top.tolist()}


def main():
    out, target, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(0)
    atom_init = json.loads(Path("Dataset/atom_init.json").read_text())
    cls = ed.runner_class(target)
    ids, test = ed.load_split(target, "test")
    train = ed.load_split(target, "train")[1] + ed.load_split(target, "validate")[1]
    stats = json.loads(Path(ed.TARGETS[target][2], "coef_stats.json").read_text())
    res = {}
    for name in names:
        cfg = {**json.loads(Path(config_of(name)).read_text()), "validation": False}
        r = object.__new__(cls)
        r.config, r.device, r.normalizer = cfg, device, Normalizer(stats, device)
        ds_test = ed.make_dataset(r, test, cfg, atom_init)
        ds_train = ed.make_dataset(r, train, cfg, atom_init)
        r.setup_model(cfg, ds_test[0].x.shape[-1])
        r.model.load_state_dict(torch.load(f"trained_models/{name}", map_location=device))
        base = r.model.to(device).eval()
        bs = cfg["batch_size"]
        test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False)
        row = {"n_bn_layers": len(bn_layers(base))}
        row["eval"] = summary(*predict_all(target, r, base, test_loader, device))

        m = copy.deepcopy(base)
        for b in bn_layers(m):
            b.train()
        row["batch"] = summary(*predict_all(target, r, m, test_loader, device))

        m = copy.deepcopy(base)
        layers = bn_layers(m)
        for b in layers:
            b.running_mean.zero_()
            b.running_var.fill_(1.0)
            b.train()
        g = torch.Generator().manual_seed(13)
        with torch.inference_mode():
            for k, d in enumerate(DataLoader(ds_train, batch_size=bs, shuffle=True, generator=g)):
                for b in layers:
                    b.momentum = 1.0 / (k + 1)  # exact running average over the k+1 batches seen
                m(d.to(device))
        m.eval()
        row["recomputed"] = summary(*predict_all(target, r, m, test_loader, device))
        res[name] = row
        print(f"{name} (BatchNorm layers: {row['n_bn_layers']})", flush=True)
        for mode in ["eval", "batch", "recomputed"]:
            s = row[mode]
            print(f"  {mode:10s} MAE {s['MAE']:.4g}  top-10 share {s['top10_share']:.1%}  "
                  f"MAE without top 10 {s['MAE_without_top10']:.4g}", flush=True)
        res[name]["top10_ids_eval"] = [ids[i] for i in row["eval"]["top10_idx"]]
    (out / f"bn_check_{target}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
