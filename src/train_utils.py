from Dataset.CrystalSO3Dataset import CrystalSO3Dataset
from typing import Dict, Tuple, Optional
import json
from torch.nn.functional import l1_loss, mse_loss
import torch
from RecombineElasticCoefficients import sphvectors2voigt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from multiprocessing import cpu_count
from contextlib import nullcontext
import wandb
import torch.optim as optim

SEED = 13
NUM_WORKERS = min(cpu_count() - 1, 16)
PROJECT = "Elastic Tensor Prediction"


def setup_datasets(validation: bool, radius: float = 8.0, normalize: bool = True, lmax_h: int = 4,
                   max_neighbors: int = 12) -> Tuple[CrystalSO3Dataset, CrystalSO3Dataset]:
    """
    Sets up the dataset for training

    Args:
        validation: Whether or not to use the validation setup
        radius: How far to look for node connections in the crystal graph in Angstroms
        normalize: Whether or not to normalize the target.
        lmax_h: The maximum l of the spherical harmonics in the model to set the input data to.
        max_neighbors: How many neighbors to include for each atom in the crystal graph.

    Returns:
        The training dataset and the testing dataset
    """
    with open("Dataset/train/data.json") as train_data:
        train_data = json.load(train_data)
    with open("Dataset/validate/data.json") as validation_data:
        if validation:
            test_data = json.load(validation_data)
        else:
            train_data.extend(json.load(validation_data))
            with open("Dataset/test/data.json") as test_data:
                test_data = json.load(test_data)
    with open("Dataset/atom_init.json") as atom_init:
        atom_init = json.load(atom_init)
    return (CrystalSO3Dataset(train_data, atom_init, lmax_h, radius=radius, max_neighbors=max_neighbors, normalize=normalize), 
            CrystalSO3Dataset(test_data, atom_init, lmax_h, radius=radius, max_neighbors=max_neighbors, normalize=normalize))

def run_epoch(model: torch.nn.Module, device: torch.DeviceObjType, dataloader: DataLoader, 
              optimizer: Optional[torch.optim.Optimizer], train: bool, normalize: bool) -> Dict[str, float]:
    """
    Runs an epoch on the model using the dataloader

    Args:
        model: The model to train
        device: The device to run on
        dataloader: The data loader holding the data to run an epoch on
        optimizer: The optimizer to update the model.
        train: Whether or not this is a training run
        normalize: Whether or not the target values are normalized
    
    Returns:
        The logs of the training metrics.
    """
    if train:
        model.train()
        location = "Train"
        if optimizer is None:
            raise Exception("Need an optimizer when training.")
    else:
        model.eval()
        location = "Validation"

    total_loss = 0
    symmetric_total_loss = 0
    asymmetric_total_loss = 0
    voigt_total_mae = 0
    voigt_total_mse = 0
    num_batches = len(dataloader)

    with nullcontext() if train else torch.inference_mode():
        for data in tqdm(dataloader, f"{location} Batch", leave=False):
            data = data.to(device)
            
            if train:
                optimizer.zero_grad()
            symmetric_pred, asymmetric_pred = model(data)

            # Compute losses
            symmetric_loss = l1_loss(symmetric_pred, data.symmetric_coefs) 
            asymmetric_loss = l1_loss(asymmetric_pred, data.asymmetric_coefs)
            loss = symmetric_loss + asymmetric_loss 

            assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

            # Do training
            if train:
                loss.backward()
                optimizer.step()

            # Scale loss if normalized
            if normalize:
                symmetric_pred, asymmetric_pred = unnormalize(symmetric_pred, asymmetric_pred, str(device))
                symmetric_coefs, asymmetric_coefs = unnormalize(data.symmetric_coefs, data.asymmetric_coefs, str(device))
                symmetric_loss = l1_loss(symmetric_pred, symmetric_coefs) 
                asymmetric_loss = l1_loss(asymmetric_pred, asymmetric_coefs)
                loss = symmetric_loss + asymmetric_loss 

            # Deal with accuracy
            total_loss += loss.item()
            symmetric_total_loss += symmetric_loss.item()
            asymmetric_total_loss += asymmetric_loss.item()
            voigt_mae, voigt_mse = evaluate_loss_on_voigt_tensor(symmetric_pred, asymmetric_pred, data.voigt_tensor)
            voigt_total_mae += voigt_mae.item()
            voigt_total_mse += voigt_mse.item()
        
    return {
        f"{location} MAE": total_loss / num_batches,
        f"{location}/Symmetric MAE": symmetric_total_loss / num_batches,
        f"{location}/Asymmetric MAE": asymmetric_total_loss / num_batches,
        f"{location}/Voigt MAE": voigt_total_mae / num_batches,
        f"{location}/Voigt RMSE": (voigt_total_mse / num_batches)**(1/2)
    }

def evaluate_loss_on_voigt_tensor(symm_coefs_pred: torch.Tensor, asymm_coefs_pred: torch.Tensor, 
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
    voigt_pred = sphvectors2voigt(asymm_coefs_pred, symm_coefs_pred)
    # Only need unique components from tensor
    voigt_pred_components = get_unique_components(voigt_pred)
    voigt_true_components = get_unique_components(voigt_true)
    return l1_loss(voigt_pred_components, voigt_true_components), mse_loss(voigt_pred_components, voigt_true_components)

def get_unique_components(voigt_tensors: torch.Tensor) -> torch.Tensor:
    """
    Gets all the unique components from the voigt tensors

    Args:
        voigt_tensors: The voigt tensors to extract unique components from.

    Returns the unique components of the voigt tensors.
    """
    return torch.hstack(
        [voigt_tensors[:, i, i:] for i in range(6)]
    )