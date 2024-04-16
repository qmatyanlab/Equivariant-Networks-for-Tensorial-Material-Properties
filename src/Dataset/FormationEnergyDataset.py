from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data

class FormationEnergyDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of materials to their dielectric tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/formation/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        formation_energy = self._get_formation_energy(idx)

        return Data(x=x, node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, formation_energy=formation_energy)
    
    def _get_formation_energy(self, idx: int) -> torch.Tensor:
        """
        Gets the band gap from the certain index in their normalized form

        Args:
            idx: Index of band gap to get

        Returns:
            The normalized formation_energy.
        """
        formation_energy = torch.tensor(self.molecule_data[idx]["formation_energy"])
        formation_energy = formation_energy.reshape(1, 1)
        coef_dict = {"formation_energy": formation_energy}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["formation_energy"]