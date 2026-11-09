"""Parameter counts and irreps of the nine tensor models (SI model-details table). Builds the models at random init only."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
import torch
from run_dielectric import DielectricRunner
from run_piezoelectric import PiezoelectricRunner
from run_elastic import ElasticRunner
nfeat = len(next(iter(json.load(open("Dataset/atom_init.json")).values())))
out = []
for tgt, cls in [("dielectric", DielectricRunner), ("piezoelectric", PiezoelectricRunner), ("elastic", ElasticRunner)]:
    for arch in ["seconv", "segnn", "setransformer"]:
        cfg = json.load(open(f"configs/{tgt}_{arch}.json"))
        r = cls.__new__(cls)
        torch.manual_seed(0)
        r.setup_model(cfg, nfeat)
        m = r.model
        n = lambda mod: sum(p.numel() for p in mod.parameters())
        row = dict(target=tgt, arch=arch, total=n(m), graph=n(m.graph_nn), head=n(m) - n(m.graph_nn),
                   hidden=str(getattr(m, "hidden_irreps", "")), head_irreps=str(getattr(m, "head_irreps", "")),
                   outputs=[str(o) for o in r.output_irreps_lst],
                   **{k: cfg[k] for k in ["num_graph_layers", "num_prepool_layers", "num_postpool_layers", "learning_rate",
                                          "batch_size", "epochs", "pooling", "radius", "max_neighbors", "layer_norm", "lmax_h",
                                          "hidden_features"]})
        out.append(row); print(row, flush=True)
print("nfeat", nfeat)
json.dump(out, open(sys.argv[1], "w"), indent=1)
