from torch_geometric.loader import DataLoader
from Dataset.DielectricDataset import DielectricDataset
from Runner import Runner
from torch.nn.functional import l1_loss, mse_loss
import torch
from contextlib import nullcontext
from tqdm import tqdm
from typing import List, Tuple, Dict
from RecombineDielectricCoefficients import realsphvec2cart
from utils import get_unique_components
from e3nn.o3 import Irreps

class DielectricRunner(Runner):
    @property
    def default_data_path(self) -> str:
        return "Dataset/dielectric"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> DielectricDataset:
        return DielectricDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

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
        dielectric_total_mae = 0
        dielectric_total_mse = 0
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
                dielectric_mae, dielectric_mse = self.evaluate_loss_on_dielectric_tensor(sph_pred, data.dielectric_tensor)
                dielectric_total_mae += dielectric_mae.item()
                dielectric_total_mse += dielectric_mse.item()
            
        return {
            f"{location} MAE": total_loss / num_batches,
            f"{location}/Dielectric MAE": dielectric_total_mae / num_batches,
            f"{location}/Dielectric RMSE": (dielectric_total_mse / num_batches)**(1/2)
        }
    
    def unnormalize(self, sph_coefs: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the spherical coefs.

        Args:
            sph_coefs: The spherical coefs to unnormalize.

        Returns:
            the unnormalized spherical coefficients.
        """
        assert sph_coefs.shape[-1] == 6, "Shape of coefs mismatch expected."
        coef_dict = {"sph": sph_coefs}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["sph"]
    
    def evaluate_loss_on_dielectric_tensor(self, sph_coefs_pred: torch.Tensor, 
                                  dielectric_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the model's prediction of the voigt tensor and calculates MAE

        Args:
            sph_coefs_pred: The predicted spherical coeficients that are unnormalized.
            dielectric_true: The True dielectric tensor

        Returns:
            The MAE and MSE of the dielectric tensor.
        """
        assert sph_coefs_pred.shape[-1] == 6, "Did not get proper shape of spherical coefficients."
        voigt_pred = realsphvec2cart(sph_coefs_pred)
        # Only need unique components from tensor
        dielectric_pred_components = get_unique_components(voigt_pred)
        dielectric_true_components = get_unique_components(dielectric_true)
        return l1_loss(dielectric_pred_components, dielectric_true_components), mse_loss(dielectric_pred_components, dielectric_true_components)
    
    def record_best(self, epoch: int, logs: dict):
        if self.best_logs is None:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
        elif logs["Validation/Dielectric MAE"] < self.best_logs["Validation/Dielectric MAE"]:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e + 1x2e")]

if __name__ == "__main__":
    runner = DielectricRunner()
    runner.main()
    
