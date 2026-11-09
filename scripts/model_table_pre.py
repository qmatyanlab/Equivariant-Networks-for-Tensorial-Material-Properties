"""Parameter counts of the band-gap and formation-energy pretraining models, and a check that the 2024 backbones match."""
import json, os, sys
sys.path.insert(0, ".")
import torch
from run_bandgap_pretrainer import BandGapPretrainer
from run_formation_pretrainer import FormationEnergyPretrainer
nfeat = len(next(iter(json.load(open("Dataset/atom_init.json")).values())))
CK = os.environ.get("CKPT2024", "trained_models")  # folder with the 2024 pretraining checkpoints (release asset)
ARCH = {"seconv": "SEConv", "segnn": "SEGNN", "setransformer": "SETransformer"}
out = []
for tgt, cls, suffix in [("bandgap", BandGapPretrainer, "Pretrain"), ("formation", FormationEnergyPretrainer, "Pretrain_-_Formation_Energy")]:
    for arch in ARCH:
        cfg = json.load(open(f"configs/{tgt}_{arch}.json"))
        r = cls.__new__(cls); r.setup_model(cfg, nfeat); m = r.model
        n = lambda mod: sum(p.numel() for p in mod.parameters())
        sd = torch.load(f"{CK}/{ARCH[arch]}_-_{suffix}_graphnn", map_location="cpu")
        ok = m.graph_nn.load_state_dict(sd) is not None
        row = dict(target=tgt, arch=arch, total=n(m), graph=n(m.graph_nn), head=n(m) - n(m.graph_nn), ckpt_loads=ok,
                   **{k: cfg[k] for k in ["num_graph_layers", "num_prepool_layers", "num_postpool_layers", "learning_rate",
                                          "batch_size", "epochs", "pooling"]})
        out.append(row); print(row, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
