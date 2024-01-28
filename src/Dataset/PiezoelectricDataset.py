from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data
from typing import Tuple

class PiezoelectricDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of materials to their piezoelectric tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/piezoelectric/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        sph_coefs = self._get_coefs(idx)
        # Save voigt for special metric check
        dielectric_tensor = torch.tensor(self.molecule_data[idx]["piezoelectric_tensor"], dtype=torch.float32).reshape(1, 3, 3)

        return Data(x=x, node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, sph_coefs=sph_coefs, dielectric_tensor=dielectric_tensor)
    
    def _get_coefs(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the coefficients from the certain index in their combined and normalized form

        Args:
            idx: Index of coefficients to get

        Returns:
            The combined and normalized coefficients.
        """
        sph_coefs = torch.tensor(self.molecule_data[idx]["sph_coefs"])
        sph_coefs = sph_coefs.reshape(1, 6)
        coef_dict = {"sph": sph_coefs}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["sph"]