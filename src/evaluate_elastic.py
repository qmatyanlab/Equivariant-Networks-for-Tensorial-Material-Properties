from Dataset.ElasticDataset import ElasticDataset
from Evaluator import Evaluator
from torch.nn.functional import l1_loss, mse_loss
import torch
from tqdm import tqdm
from typing import List, Tuple
from RecombineElasticCoefficients import realsphvectors2voigt
from e3nn.o3 import Irreps

class ElasticEvaluator(Evaluator):
    @property
    def default_data_path(self) -> str:
        return "Dataset/elastic"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> ElasticDataset:
        return ElasticDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def evaluate_results(self) -> torch.Tensor:
        """
        Runs an epoch on the model using the dataloader

        Returns:
            The component-wise mae
        """
        self.model.eval()

        voigt_mae = None
        num_batches = len(self.test_loader)

        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"Test Batch", leave=False):
                data = data.to(self.device)

                if voigt_mae is None:
                    voigt_mae = torch.zeros(data.voigt_tensor.shape[1:]).to(self.device)

                symmetric_pred, asymmetric_pred = self.model(data)

                # Scale loss if normalized
                if self.config["normalize"]:
                    symmetric_pred, asymmetric_pred = self.unnormalize(symmetric_pred, asymmetric_pred)

                # Deal with accuracy
                voigt_mae_batch = self.evaluate_loss_on_voigt_tensor(symmetric_pred, asymmetric_pred, data.voigt_tensor)

                voigt_mae += voigt_mae_batch.mean(0)
            
        return voigt_mae / num_batches
    
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
                                  voigt_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the model's prediction of the voigt tensor and calculates MAE

        Args:
            symm_coefs_pred: The predicted symmetric coefficients that are unnormalize.
            asymm_coefs_pred: The predicted asymmetric coefficients that are unnormalized.
            voigt_true: The True voigt tensor

        Returns:
            The MAE of the voigt tensor but full rank.
        """
        if symm_coefs_pred.shape[-1] != 15 or asymm_coefs_pred.shape[-1] != 6:
            raise Exception("Did not get proper symmetric or asymmetric coefficients.")
        voigt_pred = realsphvectors2voigt(asymm_coefs_pred, symm_coefs_pred)
        # Only need unique components from tensor
        return l1_loss(voigt_pred, voigt_true, reduction="none")
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e + 1x2e + 1x4e"), Irreps("1x0e + 1x2e")]

if __name__ == "__main__":
    evaluator = ElasticEvaluator()
    evaluator.main()
