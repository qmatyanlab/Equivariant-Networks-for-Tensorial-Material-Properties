from tqdm import tqdm
import json
import wandb
from train_utils import (SEED, PROJECT, run_setup,
                         run_epoch, setup_model, setup_optimizer, setup_scheduler)
from argparse import ArgumentParser
from pathlib import Path
from math import inf

def main(args):
    with args.best_config.open() as best_config:
        config = {
            **json.load(best_config),
            "seed": SEED,
            "validation": args.validation
        }

    (device, 
     train_loader, 
     validation_loader, 
     num_atom_feats) = run_setup(validation=args.validation, radius=config["radius"],  max_neighbors=config["max_neighbors"],
                                 normalize=config["normalize"], batch_size=config["batch_size"], lmax_h=config.get("lmax_h", 0))

    with wandb.init(config=config, name=args.name, project=PROJECT, tags=config["tags"] + 
                    [config["model"], "validation" if args.validation else "testing", "normalized" if config["normalize"] else "unnormalized"]):
        # Build model
        model = setup_model(config, num_atom_feats, component_only=False)
        model = model.to(device)

        # Setup optimizer and scheduled learning rate
        optimizer = setup_optimizer(config, model)
        scheduler = setup_scheduler(config, optimizer)

        # Set up best recording
        best_validation_mae, best_validation_rmse = inf, inf

        for epoch in tqdm(range(config["epochs"]), "Epoch"):
            logs = {}
            logs.update({"learning_rate": scheduler.get_last_lr()[0]})
            logs.update(run_epoch(model, device, train_loader, optimizer, train=True, normalize=config["normalize"]))
            logs.update(run_epoch(model, device, validation_loader, None, train=False, normalize=config["normalize"]))
            wandb.log(logs)
            scheduler.step()
            if logs["Validation/Voigt MAE"] < best_validation_mae:
                best_validation_mae = logs["Validation/Voigt MAE"]
                best_validation_rmse = logs["Validation/Voigt RMSE"]
                best_epoch = epoch

        # Log summary
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_validation_mae"] = best_validation_mae
        wandb.run.summary["best_validation_rmse"] = best_validation_rmse

if __name__ == "__main__":
    args = ArgumentParser(description="Runs a full training run using the train and test sets.")
    args.add_argument("name", type=str, help="The name of the final run. Appended to Final Run - ")
    args.add_argument("best_config", type=Path, help="The path to the best config to run")
    args.add_argument("--validation", action="store_true", help="Flag to indiciate this is a validation run.")
    args = args.parse_args()
    main(args)