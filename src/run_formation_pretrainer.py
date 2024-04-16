from Pretrainer import Pretrainer
from typing import List, Dict
from Dataset.FormationEnergyDataset import FormationEnergyDataset
from torch_geometric.loader import DataLoader
from e3nn.o3 import Irreps
from pathlib import Path
import torch
from tqdm import tqdm
from torch.nn.functional import l1_loss, mse_loss

class FormationEnergyPretrainer(Pretrainer):
    @property
    def default_data_path(self) -> str:
        return "Dataset/formation"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> FormationEnergyDataset:
        return FormationEnergyDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def run_epoch(self, dataloader: DataLoader, train: bool) -> Dict[str, int]:
        """
        Runs an epoch on the model using the dataloader

        Args:
            dataloader: The data loader holding the data to run an epoch on
            train: Whether or not this is a training run
        
        Returns:
            The logs of the training metrics.
        """
        assert train, "Validation is not supported"
        self.model.train()

        total_mae_loss = 0
        total_mse_loss = 0
        num_batches = len(dataloader)

        for data in tqdm(dataloader, f"Pretrain Batch", leave=False):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            formation_energy_pred = self.model(data)[0]

            # Compute losses
            loss = l1_loss(formation_energy_pred, data.formation_energy) 

            assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

            # Do training
            loss.backward()
            self.optimizer.step()

            # Scale loss if normalized
            if self.config["normalize"]:
                formation_energy_pred = self.unnormalize(formation_energy_pred)
                formation_energy = self.unnormalize(data.formation_energy)
                loss = l1_loss(formation_energy_pred, formation_energy) 
                
            # Deal with accuracy
            total_mae_loss += loss.item()
            total_mse_loss += mse_loss(formation_energy_pred, formation_energy).item()
            
        return {
            "MAE": total_mae_loss / num_batches,
            "RMSE": (total_mse_loss / num_batches)**(1/2)
        }
    
    def unnormalize(self, formation_energy: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the band gap.

        Args:
            formation_energy: The band gaps to unnormalize.

        Returns:
            the unnormalized band gaps.
        """
        assert formation_energy.shape[-1] == 1, "Shape of band gap mismatch expected."
        coef_dict = {"formation_energy": formation_energy}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["formation_energy"]
    
    def record_best(self, epoch: int, logs: dict):
        if self.best_logs is None:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
        elif logs["MAE"] < self.best_logs["MAE"]:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e")]

if __name__ == "__main__":
    runner = FormationEnergyPretrainer()
    runner.main()
    # Save results
    Path("trained_models").mkdir(exist_ok=True)
    torch.save(runner.best_model_state_dict, f"trained_models/{runner.name.replace(' ', '_')}")
    runner.model.load_state_dict(runner.best_model_state_dict)
    torch.save(runner.model.graph_nn.state_dict(), f"trained_models/{runner.name.replace(' ', '_')}_graphnn")