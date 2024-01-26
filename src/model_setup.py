from e3nn.o3 import Irreps
from typing import Literal, Tuple, List, Union
from wandb.sdk.wandb_config import Config
import torch

SYMMETRIC_OUTPUT = Irreps("1x0e + 1x2e + 1x4e")
ASYMMETRIC_OUTPUT = Irreps("1x0e + 1x2e")

def setup_model(config: Union[dict, Config], num_atom_feats: int) -> torch.nn.Module:
    """
    Sets up the models to be run with the right output style

    Args:
        config: The config file to use when setting up the models
        num_atom_feats: The number of atom features to use.

    Returns:
        The requested model.
    """
    # Setup specific model
    output_irreps_lst = [SYMMETRIC_OUTPUT, ASYMMETRIC_OUTPUT]
    if "segnn" == config["model"]:
        model = setup_sempgnn(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                            hidden_irreps=config["hidden_irreps"], num_graph_layers=config["num_graph_layers"],
                            num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                            output_irreps_lst=output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
    elif "seconv" in config["model"]:
        model = setup_seconv(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], hidden_irreps=config["hidden_irreps"], 
                            num_graph_layers=config["num_graph_layers"], num_prepool_layers=config["num_prepool_layers"], 
                            num_postpool_layers=config["num_postpool_layers"], conv_type=config["model"][config["model"].rfind("-") + 1:],
                            output_irreps_lst=output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
    elif "setransformer" == config["model"]:
        model = setup_setransformer(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                            hidden_irreps=config["hidden_irreps"], num_graph_layers=config["num_graph_layers"],
                            num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                            output_irreps_lst=output_irreps_lst, norm=config["layer_norm"], max_radius=config["radius"], 
                            pooling=config["pooling"])
    else:
        raise Exception("No such model of that type")
    return model
        
def setup_irreps(num_atom_feats: int, lmax_h: int, hidden_irreps: str) -> Tuple[Irreps, Irreps, Irreps, Irreps, Irreps, Irreps, Irreps]:
    """
    Sets up the irreps for the model input, hidden layers, and output

    Args:
        num_atom_feats: The number of atom scalar features provided.
        lmax_h: The maximum l for the hidden layers
        hidden_irreps: The hidden irreps string

    Returns all the required irreps
    """
    # Scalar one hot encoding
    input_irreps = Irreps([(num_atom_feats, (0, 1))])
    # Irreps from relative position between atoms based on spherical harmonics
    edge_attr_irreps = Irreps.spherical_harmonics(lmax_h)
    node_attr_irreps = Irreps.spherical_harmonics(lmax_h)
    # Distance is a singular scalar
    additional_message_irreps = Irreps("1x0e")
    # Create hidden irreps
    hidden_irreps = Irreps(hidden_irreps)
    return input_irreps, edge_attr_irreps, node_attr_irreps, additional_message_irreps, hidden_irreps

def setup_sempgnn(num_atom_feats: int, lmax_h: int, hidden_irreps: str, num_graph_layers: int, 
                  num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                  norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> torch.nn.Module:
    """
    Sets up a SEGNN model based on the parameters provided.

    Args:
        num_atom_feats: How many atomic features will be provided
        lmax_h: The maximum l for the hidden layer
        hidden_irreps: The hidden irreps string
        num_graph_layers: How many hidden layers of graph interaction
        num_prepool_layers: How many linear layers on just node values
        num_postpool_layers: How many layers before final output after pooling
        output_irreps_lst: The list of outputs of the model
        norm: The layer normalization the model may employ
        pooling: How to pool the nodes together
        subspace_type: How to allocate the irreps used in the hidden layers.

    Returns the specified SEMPNN model
    """
    from Model.SEMPNN import SEMPNN
    (input_irreps, 
     edge_attr_irreps, 
     node_attr_irreps, 
     additional_message_irreps, 
     hidden_irreps) = setup_irreps(num_atom_feats, lmax_h, hidden_irreps)

    return SEMPNN(input_irreps,
        hidden_irreps,
        output_irreps_lst,
        edge_attr_irreps,
        node_attr_irreps,
        num_graph_layers=num_graph_layers,
        num_prepool_layers=num_prepool_layers,
        num_postpool_layers=num_postpool_layers,
        additional_message_irreps=additional_message_irreps,
        norm=norm,
        pooling=pooling)

def setup_seconv(num_atom_feats: int, lmax_h: int, hidden_irreps: str, num_graph_layers: int, 
                 num_prepool_layers: int, num_postpool_layers: int, 
                 conv_type: Literal["linear", "nonlinear"], output_irreps_lst: List[Irreps],
                 norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"],
                 subspace_type: Literal["balanced", "weightbalanced"]) -> torch.nn.Module:
    """
    Sets up a SEConv model based on the parameters provided.

    Args:
        num_atom_feats: How many atomic features will be provided
        lmax_h: The maximum l for the hidden layer
        hidden_irreps: How many features in the hidden layers
        num_graph_layers: How many hidden layers of graph interaction
        num_prepool_layers: How many linear layers on just node values
        num_postpool_layers: How many layers before final output after pooling
        conv_type: Either linear or nonlinear convolutions
        output_irreps_lst: The list of outputs of the model
        norm: The layer normalization the model may employ
        pooling: How to pool the nodes together
        subspace_type: How to allocate the irreps used in the hidden layers.

    Returns the specified SEConv model.
    """
    from Model.SEConvNN import SEConv
    (input_irreps, 
     edge_attr_irreps, 
     node_attr_irreps, 
     additional_message_irreps, 
     hidden_irreps) = setup_irreps(num_atom_feats, lmax_h, hidden_irreps)

    return SEConv(input_irreps,
        hidden_irreps,
        output_irreps_lst,
        edge_attr_irreps,
        node_attr_irreps,
        num_graph_layers=num_graph_layers,
        num_prepool_layers=num_prepool_layers,
        num_postpool_layers=num_postpool_layers,
        additional_message_irreps=additional_message_irreps,
        conv_type=conv_type,
        norm=norm,
        pooling=pooling)

def setup_setransformer(num_atom_feats: int, lmax_h: int, hidden_irreps: str, num_graph_layers: int, 
                  num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                  norm: Literal["none", "batch"], max_radius: float, pooling: Literal["mean", "add", "max"]) -> torch.nn.Module:
    """
    Sets up a SETransformer model based on the parameters provided.

    Args:
        num_atom_feats: How many atomic features will be provided
        lmax_h: The maximum l for the hidden layer
        hidden_irreps: How many features in the hidden layers
        num_graph_layers: How many hidden layers of graph interaction
        num_prepool_layers: How many linear layers on just node values
        num_postpool_layers: How many layers before final output after pooling
        output_irreps_lst: The list of outputs of the model
        norm: The layer normalization the model may employ
        max_radius: The maximum radius for cutoff function
        pooling: How to pool the nodes together
        subspace_type: How to allocate the irreps used in the hidden layers.

    Returns the SETransformer model as specified
    """
    from Model.SETransformer import SETransformer
    (input_irreps, 
     edge_attr_irreps, 
     node_attr_irreps, 
     additional_message_irreps, 
     hidden_irreps) = setup_irreps(num_atom_feats, lmax_h, hidden_irreps)

    return SETransformer(input_irreps,
        hidden_irreps,
        output_irreps_lst,
        edge_attr_irreps,
        node_attr_irreps,
        num_graph_layers=num_graph_layers,
        num_prepool_layers=num_prepool_layers,
        num_postpool_layers=num_postpool_layers,
        additional_message_irreps=additional_message_irreps,
        norm=norm,
        max_radius=max_radius,
        pooling=pooling)