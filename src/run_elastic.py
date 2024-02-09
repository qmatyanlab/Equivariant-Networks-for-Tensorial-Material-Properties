from torch_geometric.loader import DataLoader
from Dataset.ElasticDataset import ElasticDataset
from Runner import Runner
from torch.nn.functional import l1_loss, mse_loss
import torch
from contextlib import nullcontext
from tqdm import tqdm
from typing import List, Tuple, Dict
from RecombineElasticCoefficients import realsphvectors2voigt
from utils import get_unique_components
from e3nn.o3 import Irreps

class ElasticRunner(Runner):
    @property
    def default_data_path(self) -> str:
        return "Dataset/elastic"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> ElasticDataset:
        return ElasticDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

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
        symmetric_total_loss = 0
        asymmetric_total_loss = 0
        voigt_total_mae = 0
        voigt_total_mse = 0
        num_batches = len(dataloader)

        with nullcontext() if train else torch.inference_mode():
            for data in tqdm(dataloader, f"{location} Batch", leave=False):
                data = data.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                symmetric_pred, asymmetric_pred = self.model(data)

                # Compute losses
                symmetric_loss = l1_loss(symmetric_pred, data.symmetric_coefs) 
                asymmetric_loss = l1_loss(asymmetric_pred, data.asymmetric_coefs)
                loss = symmetric_loss + asymmetric_loss 

                assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

                # Do training
                if train:
                    loss.backward()
                    self.optimizer.step()

                # Scale loss if normalized
                if self.config["normalize"]:
                    symmetric_pred, asymmetric_pred = self.unnormalize(symmetric_pred, asymmetric_pred)
                    symmetric_coefs, asymmetric_coefs = self.unnormalize(data.symmetric_coefs, data.asymmetric_coefs)
                    symmetric_loss = l1_loss(symmetric_pred, symmetric_coefs) 
                    asymmetric_loss = l1_loss(asymmetric_pred, asymmetric_coefs)
                    loss = symmetric_loss + asymmetric_loss 

                # Deal with accuracy
                total_loss += loss.item()
                symmetric_total_loss += symmetric_loss.item()
                asymmetric_total_loss += asymmetric_loss.item()
                voigt_mae, voigt_mse = self.evaluate_loss_on_voigt_tensor(symmetric_pred, asymmetric_pred, data.voigt_tensor)
                voigt_total_mae += voigt_mae.item()
                voigt_total_mse += voigt_mse.item()
            
        return {
            f"{location} MAE": total_loss / num_batches,
            f"{location}/Symmetric MAE": symmetric_total_loss / num_batches,
            f"{location}/Asymmetric MAE": asymmetric_total_loss / num_batches,
            f"{location}/Voigt MAE": voigt_total_mae / num_batches,
            f"{location}/Voigt RMSE": (voigt_total_mse / num_batches)**(1/2)
        }
    
    def unnormalize(self, symmetric_coefs: torch.Tensor, asymmetric_coefs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unnormalizes the symmetric and asymmetric coefs.

        Args:
            symmetric_coefs: The symmetric coefs to unnormalize
            asymmetric_coefs: The asymmetric coefs to unnormalize

        Returns:
            the unnormalized symmetric and asymmetric coefs
        """
        assert symmetric_coefs.shape[-1] == 15 and asymmetric_coefs.shape[-1] == 6, "Shape of coefs mismatch expected."
        coef_dict = {"symmetric": symmetric_coefs, "asymmetric": asymmetric_coefs}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["symmetric"], unnorm_dict["asymmetric"]
    
    def evaluate_loss_on_voigt_tensor(self, symm_coefs_pred: torch.Tensor, asymm_coefs_pred: torch.Tensor, 
                                  voigt_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the model's prediction of the voigt tensor and calculates MAE

        Args:
            symm_coefs_pred: The predicted symmetric coefficients that are unnormalize.
            asymm_coefs_pred: The predicted asymmetric coefficients that are unnormalized.
            voigt_true: The True voigt tensor

        Returns:
            The MAE and MSE of the voigt tensor.
        """
        if symm_coefs_pred.shape[-1] != 15 or asymm_coefs_pred.shape[-1] != 6:
            raise Exception("Did not get proper symmetric or asymmetric coefficients.")
        voigt_pred = realsphvectors2voigt(asymm_coefs_pred, symm_coefs_pred)
        # Only need unique components from tensor
        voigt_pred_components = get_unique_components(voigt_pred)
        voigt_true_components = get_unique_components(voigt_true)
        return l1_loss(voigt_pred_components, voigt_true_components), mse_loss(voigt_pred_components, voigt_true_components)
    
    def record_best(self, epoch: int, logs: dict):
        if self.best_logs is None:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
        elif logs["Validation/Voigt MAE"] < self.best_logs["Validation/Voigt MAE"]:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e + 1x2e + 1x4e"), Irreps("1x0e + 1x2e")]

if __name__ == "__main__":
    runner = ElasticRunner()
    runner.main()
