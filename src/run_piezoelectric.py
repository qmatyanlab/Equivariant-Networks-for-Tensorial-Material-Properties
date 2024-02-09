from torch_geometric.loader import DataLoader
from Dataset.PiezoelectricDataset import PiezoelectricDataset
from Runner import Runner
from torch.nn.functional import l1_loss, mse_loss
import torch
from contextlib import nullcontext
from tqdm import tqdm
from typing import List, Tuple, Dict
from RecombinePiezoelectricCoefficients import reconstruct_from_harmonic_vec
from e3nn.o3 import Irreps

class PiezoelectricRunner(Runner):
    @property
    def default_data_path(self) -> str:
        return "Dataset/piezo"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> PiezoelectricDataset:
        return PiezoelectricDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def run_epoch(self, dataloader: DataLoader, train: bool) -> Dict[str, int]:
        """
        Runs an epoch on the model using the dataloader

        Args:
            dataloader: The data loader holding the data to run an epoch on
            train: Whether or not this is a training run
        
        Returns:
            The logs of the training metrics.
        """
        if train:
            self.model.train()
            location = "Train"
        else:
            self.model.eval()
            location = "Validation"

        total_loss = 0
        piezoelectric_total_mae = 0
        piezoelectric_total_mse = 0
        num_batches = len(dataloader)

        with nullcontext() if train else torch.inference_mode():
            for data in tqdm(dataloader, f"{location} Batch", leave=False):
                data = data.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                sph_pred = self.model(data)[0]

                # Compute losses
                loss = l1_loss(sph_pred, data.sph_coefs) 

                assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

                # Do training
                if train:
                    loss.backward()
                    self.optimizer.step()

                # Scale loss if normalized
                if self.config["normalize"]:
                    sph_pred = self.unnormalize(sph_pred)
                    sph_coefs = self.unnormalize(data.sph_coefs)
                    loss = l1_loss(sph_pred, sph_coefs) 
                    
                # Deal with accuracy
                total_loss += loss.item()
                piezoelectric_mae, piezoelectric_mse = self.evaluate_loss_on_piezoelectric_tensor(sph_pred, data.piezoelectric_tensor)
                piezoelectric_total_mae += piezoelectric_mae.item()
                piezoelectric_total_mse += piezoelectric_mse.item()
            
        return {
            f"{location} MAE": total_loss / num_batches,
            f"{location}/Piezoelectric MAE": piezoelectric_total_mae / num_batches,
            f"{location}/Piezoelectric RMSE": (piezoelectric_total_mse / num_batches)**(1/2)
        }
    
    def unnormalize(self, sph_coefs: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the spherical coefs.

        Args:
            sph_coefs: The spherical coefs to unnormalize.

        Returns:
            the unnormalized spherical coefficients.
        """
        assert sph_coefs.shape[-1] == 18, "Shape of coefs mismatch expected."
        coef_dict = {"sph": sph_coefs}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["sph"]
    
    def evaluate_loss_on_piezoelectric_tensor(self, sph_coefs_pred: torch.Tensor, 
                                  piezoelectric_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the model's prediction of the voigt tensor and calculates MAE

        Args:
            sph_coefs_pred: The predicted spherical coeficients that are unnormalized.
            piezoelectric_true: The True piezoelectric tensor

        Returns:
            The MAE and MSE of the piezoelectric tensor.
        """
        assert sph_coefs_pred.shape[-1] == 18, "Did not get proper shape of spherical coefficients."
        piezoelectric_pred = reconstruct_from_harmonic_vec(sph_coefs_pred)
        return l1_loss(piezoelectric_pred, piezoelectric_true), mse_loss(piezoelectric_pred, piezoelectric_true)
    
    def record_best(self, epoch: int, logs: dict):
        if self.best_logs is None:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
        elif logs["Validation/Piezoelectric MAE"] < self.best_logs["Validation/Piezoelectric MAE"]:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x1o + 1x3o + 1x1o + 1x2e")]

if __name__ == "__main__":
    runner = PiezoelectricRunner()
    runner.main()
