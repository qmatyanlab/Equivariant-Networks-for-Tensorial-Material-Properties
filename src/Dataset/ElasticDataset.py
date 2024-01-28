from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data
from typing import Tuple

class ElasticDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of crystal to their elastic tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/elastic/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        symmetric_coefs, asymmetric_coefs = self._get_coefs(idx)
        # Save voigt for special metric check
        voigt_tensor = torch.tensor(self.molecule_data[idx]["elasticity_tensor"], dtype=torch.float32).reshape(1, 6, 6)

        return Data(x=x, node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, symmetric_coefs=symmetric_coefs, asymmetric_coefs=asymmetric_coefs, 
                    voigt_tensor=voigt_tensor)
    
    def _get_coefs(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the coefficients from the certain index in their combined and normalized form

        Args:
            idx: Index of coefficients to get

        Returns:
            The combined and normalized coefficients.
        """
        symmetric_coefs = torch.tensor(self.molecule_data[idx]["symmetric_coefs"], dtype=torch.float32)
        asymmetric_coefs = torch.tensor(self.molecule_data[idx]["asymmetric_coefs"], dtype=torch.float32)
        symmetric_coefs = symmetric_coefs.reshape(1, 15)
        asymmetric_coefs = asymmetric_coefs.reshape(1, 6)
        coef_dict = {"symmetric": symmetric_coefs, "asymmetric": asymmetric_coefs}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["symmetric"], coef_dict["asymmetric"]