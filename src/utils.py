import torch

def get_unique_components(symmetric_tensors: torch.Tensor) -> torch.Tensor:
    """
    Gets all the unique components from the voigt tensors

    Args:
        symmetric_tensors: The symmetric tensors to extract unique components from.

    Returns the unique components of the voigt tensors.
    """
    return torch.hstack(
        [symmetric_tensors[:, i, i:] for i in range(symmetric_tensors.shape[-1])]
    )
