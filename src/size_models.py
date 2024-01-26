from argparse import ArgumentParser
from pathlib import Path
import json
from train_utils import setup_model

def compute_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    with open(args.config) as f:
        config = json.load(f)
    with open("Dataset/atom_init.json") as f:
        atom_feats = len(json.load(f)["1"])
    model = setup_model(config, atom_feats, False)
    print(compute_parameters(model))

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("config", help="The config file to get the model from", type=Path)
    args = args.parse_args()
    main(args)