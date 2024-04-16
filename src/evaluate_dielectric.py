from Dataset.DielectricDataset import DielectricDataset
from Evaluator import Evaluator
from torch.nn.functional import l1_loss
import torch
from tqdm import tqdm
from typing import List, Tuple
from RecombineDielectricCoefficients import realsphvec2cart
from e3nn.o3 import Irreps

class DielectricEvaluator(Evaluator):
    @property
    def default_data_path(self) -> str:
        return "Dataset/dielectric"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> DielectricDataset:
        return DielectricDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def evaluate_results(self) -> torch.Tensor:
        """
        Runs an epoch on the model using the dataloader

        Returns:
            The component-wise mae
        """
        self.model.eval()

        dielectric_mae = None
        num_batches = len(self.test_loader)

        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"Test Batch", leave=False):
                data = data.to(self.device)

                if dielectric_mae is None:
                    dielectric_mae = torch.zeros(data.dielectric_tensor.shape[1:]).to(self.device)

                sph_pred = self.model(data)[0]

                # Scale loss if normalized
                if self.config["normalize"]:
                    sph_pred = self.unnormalize(sph_pred)

                # Deal with accuracy
                dielectric_mae_batch = self.evaluate_loss_on_dielectric_tensor(sph_pred, data.dielectric_tensor)

                dielectric_mae += dielectric_mae_batch.mean(0)
            
        return dielectric_mae / num_batches
    
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
        Computes the model's prediction of the dielectric tensor and calculates MAE

        Args:
            sph_coefs_pred: The predicted spherical coeficients that are unnormalized.
            dielectric_true: The True dielectric tensor

        Returns:
            The MAE of the dielectric tensor.
        """
        assert sph_coefs_pred.shape[-1] == 6, "Did not get proper shape of spherical coefficients."
        dielectric_pred = realsphvec2cart(sph_coefs_pred)
        # Only need unique components from tensor
        return l1_loss(dielectric_pred, dielectric_true, reduction="none")
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e + 1x2e")]

if __name__ == "__main__":
    evaluator = DielectricEvaluator()
    evaluator.main()
