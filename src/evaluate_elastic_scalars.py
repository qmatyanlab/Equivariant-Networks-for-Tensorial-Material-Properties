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
        total_k_h_mae = 0
        total_g_h_mae = 0
        total_young_mae = 0
        num_batches = len(self.test_loader)

        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"Test Batch", leave=False):
                data = data.to(self.device)

                symmetric_pred, asymmetric_pred = self.model(data)

                # Scale loss if normalized
                if self.config["normalize"]:
                    symmetric_pred, asymmetric_pred = self.unnormalize(symmetric_pred, asymmetric_pred)

                # Deal with accuracy
                k_h_mae, g_h_mae, young_mae = self.evaluate_loss_on_voigt_tensor(symmetric_pred, asymmetric_pred, data.voigt_tensor)

                total_k_h_mae += k_h_mae.item()
                total_g_h_mae += g_h_mae.item()
                total_young_mae += young_mae.item()

        return total_k_h_mae / num_batches, total_g_h_mae / num_batches, total_young_mae / num_batches
    
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
        # Get scalar predictions
        k_h_pred = self.compute_k_h(voigt_pred)
        g_h_pred = self.compute_g_h(voigt_pred)
        young_pred = self.compute_young_modulus(voigt_pred)
        # Get scalar true
        k_h_true = self.compute_k_h(voigt_true)
        g_h_true = self.compute_g_h(voigt_true)
        young_true = self.compute_young_modulus(voigt_true)
        return l1_loss(k_h_pred, k_h_true), l1_loss(g_h_pred, g_h_true), l1_loss(young_pred, young_true)
        
        
    def compute_k_v(self, voigt_tensor):
        return (1/9) * ((voigt_tensor[:,0,0] + voigt_tensor[:,1,1] + voigt_tensor[:,2,2]) \
                        + (2*(voigt_tensor[:,0,1] + voigt_tensor[:,1,2] + voigt_tensor[:,2,0])))
    
    def compute_g_v(self, voigt_tensor):
        return (1/15) * ((voigt_tensor[:,0,0] + voigt_tensor[:,1,1] + voigt_tensor[:,2,2]) \
                        - (voigt_tensor[:,0,1] + voigt_tensor[:,1,2] + voigt_tensor[:,2,0]) \
                        + (3*(voigt_tensor[:,3,3] + voigt_tensor[:,4,4] + voigt_tensor[:,5,5])))
    
    def compute_k_r(self, voigt_tensor):
        return 1 / ((voigt_tensor[:,0,0] + voigt_tensor[:,1,1] + voigt_tensor[:,2,2]) \
                        + (2*(voigt_tensor[:,1,2] + voigt_tensor[:,2,3] + voigt_tensor[:,3,1])))
    
    def compute_g_r(self, voigt_tensor):
        return 15 / ((4*(voigt_tensor[:,0,0] + voigt_tensor[:,1,1] + voigt_tensor[:,2,2])) \
                        - (4*(voigt_tensor[:,0,1] + voigt_tensor[:,1,2] + voigt_tensor[:,2,0])) \
                        + (3*(voigt_tensor[:,3,3] + voigt_tensor[:,4,4] + voigt_tensor[:,5,5])))
    
    def compute_k_h(self, voigt_tensor):
        return (self.compute_k_v(voigt_tensor) + self.compute_k_r(voigt_tensor)) / 2
    
    def compute_g_h(self, voigt_tensor):
        return (self.compute_g_v(voigt_tensor) + self.compute_g_r(voigt_tensor)) / 2
    
    def compute_young_modulus(self, voigt_tensor):
        K = self.compute_k_h(voigt_tensor)
        G = self.compute_g_h(voigt_tensor)
        return (9 * K * G) / ((3*K) + G)
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e + 1x2e + 1x4e"), Irreps("1x0e + 1x2e")]
    
    def main(self):
        """
        Evaluates a model over the test dataset
        """
        k_h_mae, g_h_mae, young_mae = self.evaluate_results()
        print("K_H mae", k_h_mae)
        print("G_H mae", g_h_mae)
        print("Young mae", young_mae)

if __name__ == "__main__":
    evaluator = ElasticEvaluator()
    evaluator.main()
