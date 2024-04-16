from Dataset.PiezoelectricDataset import PiezoelectricDataset
from Evaluator import Evaluator
from torch.nn.functional import l1_loss
import torch
from tqdm import tqdm
from typing import List, Tuple
from RecombinePiezoelectricCoefficients import reconstruct_from_harmonic_vec
from e3nn.o3 import Irreps

class piezoelectricEvaluator(Evaluator):
    @property
    def default_data_path(self) -> str:
        return "Dataset/piezo"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> PiezoelectricDataset:
        return PiezoelectricDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def evaluate_results(self) -> torch.Tensor:
        """
        Runs an epoch on the model using the dataloader

        Returns:
            The component-wise mae
        """
        self.model.eval()

        piezoelectric_mae = None
        num_batches = len(self.test_loader)

        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"Test Batch", leave=False):
                data = data.to(self.device)

                if piezoelectric_mae is None:
                    piezoelectric_mae = torch.zeros(data.piezoelectric_tensor.shape[1:]).to(self.device)

                sph_pred = self.model(data)[0]

                # Scale loss if normalized
                if self.config["normalize"]:
                    sph_pred = self.unnormalize(sph_pred)

                # Deal with accuracy
                piezoelectric_mae_batch = self.evaluate_loss_on_piezoelectric_tensor(sph_pred, data.piezoelectric_tensor)

                piezoelectric_mae += piezoelectric_mae_batch.mean(0)
            
        return piezoelectric_mae / num_batches
    
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
            The MAE of the piezoelectric tensor.
        """
        assert sph_coefs_pred.shape[-1] == 18, "Did not get proper shape of spherical coefficients."
        piezoelectric_pred = reconstruct_from_harmonic_vec(sph_coefs_pred)
        return l1_loss(piezoelectric_pred, piezoelectric_true, reduction="none")
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x1o + 1x3o + 1x1o + 1x2e")]
    
if __name__ == "__main__":
    evaluator = piezoelectricEvaluator()
    evaluator.main()